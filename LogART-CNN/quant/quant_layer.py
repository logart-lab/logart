from re import X
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from torch.autograd import Variable
import math
from skfuzzy.image import nmse


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


@torch.no_grad()
def log_quantize(x, scale, zero, method, code_map, level_map, asym_map, hardware_approx):
    if 'sqrt2' in method:
        x_log = torch.div(torch.log(torch.abs(x / scale) + 1e-32), torch.log(code_map))
        x_int = torch.round(x_log)
        # Clamp Low
        x_clamp_1 = torch.clamp(x_int - zero, -1 - asym_map / 2, math.inf * torch.ones_like(x))
        del x_int, x_log
        torch.cuda.empty_cache()
        # Clamp High
        x_clamp = torch.clamp(x_clamp_1 - level_map, -math.inf, -1)

        x_quant = torch.pow(code_map, x_clamp + level_map + zero)

        if hardware_approx:
            approx_factor = 1.5 / math.sqrt(2)
            # approx_factor = (1 + 1/2 - 2**(-4) ) / math.sqrt(2)
            # approx_factor = (1 + 1/2 - 2**(-4) - 2**(-6)) / math.sqrt(2)
            sqrt2_flag = (code_map == math.sqrt(2)) & ((x_clamp + level_map 
                                                + zero) % 2 != 0)
            x_quant = torch.where(sqrt2_flag, x_quant * approx_factor , x_quant)    
        
        del x_clamp, code_map, zero, level_map
        torch.cuda.empty_cache()

        # Set the minimum value to 0
        s = torch.sign(x.detach())
        zero_flag = (x_clamp_1 <= -1 - asym_map / 2)
        x_quant[zero_flag] = 0

        x_dequant = x_quant * s * scale
    
        return x_dequant
    else:
        x_log = torch.div(torch.log(torch.abs(x / scale) + 1e-32), torch.log(code_map))
        x_int = torch.round(x_log)
        # Clamp Low
        x_clamp_1 = torch.clamp(x_int - zero, 0 - (1 + asym_map / 2) * code_map, math.inf * torch.ones_like(x))
        del x_int, x_log
        torch.cuda.empty_cache()
        # Clamp High
        x_clamp = torch.clamp(x_clamp_1 - level_map, -math.inf, -1)

        x_quant = torch.pow(code_map, x_clamp + level_map + zero)

        if hardware_approx:
            approx_factor = 1.5 / math.sqrt(2)
            # approx_factor = (1 + 1/2 - 2**(-4) ) / math.sqrt(2)
            # approx_factor = (1 + 1/2 - 2**(-4) - 2**(-6)) / math.sqrt(2)
            sqrt2_flag = (code_map == math.sqrt(2)) & ((x_clamp + level_map 
                                                    + zero) % 2 != 0)
            x_quant = torch.where(sqrt2_flag, x_quant * approx_factor , x_quant)

        del x_clamp, zero, level_map
        torch.cuda.empty_cache()

        # Set the minimum value to 0
        s = torch.sign(x.detach())
        zero_flag = (x_clamp_1 <= -1 - asym_map / 2) & (code_map == 2)
        x_quant[zero_flag] = 0

        x_dequant = x_quant * s * scale

        return x_dequant


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    :param hardware_apprx: if True, use 1+1/2 hardware approximation for quantization
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, 
                 scale_method: str = 'max', hardware_apprx: bool = False,
                 leaf_param: bool = False):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits

        self.delta = None
        self.zero_point = None
        self.code_map = None
        self.level_map = None
        self.asym_map = None

        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        self.hardware_approx = hardware_apprx

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            if self.leaf_param:
                delta, self.zero_point, self.code_map, self.level_map, self.asym_map = self.init_quantization_scale(x, self.channel_wise) 
                # self.delta = torch.nn.Parameter(delta)
                # self.zero_point = torch.nn.Parameter(self.zero_point)
            else:
                self.delta, self.zero_point, self.code_map, self.level_map, self.asym_map = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        if self.scale_method in {'log_2', 'log_sqrt2', 'log_dynamic'}:
            x_dequant = log_quantize(x, self.delta, self.zero_point, self. scale_method, self.code_map, self.level_map, self.asym_map, self.hardware_approx)

        else:
            raise NotImplementedError('scale_method {} not supported'.format(self.scale_method))

        return x_dequant


    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        if not channel_wise:
            assert False, 'Per-layer quantization is currently not supported!'
        else:
            if 'log' in self.scale_method:
                # get the channel-wise max of the weight
                weight_reshape = x.view(x.shape[0], -1)
                tmp = torch.zeros(x.shape[0], device=x.device)
                xmin = torch.minimum(weight_reshape.min(1)[0], tmp)
                xmax = torch.maximum(weight_reshape.max(1)[0], tmp)
                if self.sym:
                    xmax = torch.maximum(torch.abs(xmin), xmax)
                    xmin = -xmax
                
                tmp = (xmin == 0) & (xmax == 0)
                # assert that xmin and xmax are not both 0
                assert not torch.any(tmp), "xmin and xmax are both 0"

                # Initialize Scale, Zero, Code Map, and Level Map
                dim_diff = x.dim() - (xmin.unsqueeze(1)).dim()
                target_shape = (xmin.unsqueeze(1)).shape + (1,) * dim_diff

                best = torch.full_like(xmin.unsqueeze(1), 1e10, device=x.device) # best shape: [num_channels, 1] 
                scale = torch.zeros_like((xmin.unsqueeze(1)).view(target_shape), device=x.device) # scale shape: [num_channels, 1, 1, 1]
                zero = torch.zeros_like(x, device=x.device) # zero shape: [num_channels, 1/in_channels, kernel_size/1, kernel_size/1]
                code_map = torch.zeros_like(x, device=x.device) # code_map shape: [num_channels, 1/in_channels, kernel_size/1, kernel_size/1]
                level_map = torch.zeros_like(x, device=x.device) # level_map shape: [num_channels, 1/in_channels, kernel_size/1, kernel_size/1]
                asym_map = torch.zeros_like((xmin.unsqueeze(1)).view(target_shape), device=x.device) # asym_map shape: [num_channels, 1, 1, 1]

                if self.scale_method == 'log_2':                    
                    for i in range(int(.01 * 100)): # The scaling factor search is disabled here
                        p = 1 - i / 100

                        code_map1, level_map1, zero1, asym_map1 = self.get_code_map(x, xmax.unsqueeze(1), xmin.unsqueeze(1), 2 ** (self.n_bits - 1), self.n_bits, self.sym)
                        
                        scale1 = torch.ones([x.shape[0], 1], device=x.device) * p
                        scale1 = scale1.view(target_shape) # scale1 shape: [num_channels, 1, 1, 1]

                        # Quantize
                        q = log_quantize(x, scale=scale1, zero=zero1, method=self.scale_method, code_map=code_map1, level_map=level_map1, asym_map=asym_map1, hardware_approx=self.hardware_approx)
                        q -= x
                        q.abs_()
                        q.pow_(2)
                        err = torch.sum(q.view(q.shape[0], -1), dim=-1, keepdim=True)

                        tmp = (err < best)
                        tmp = tmp.view(target_shape)

                        scale = torch.where(tmp, scale1, scale)
                        zero = torch.where(tmp, zero1, zero)
                        code_map = torch.where(tmp, code_map1, code_map)
                        level_map = torch.where(tmp, level_map1, level_map)
                        asym_map = torch.where(tmp, asym_map1, asym_map)
                        best = torch.minimum(err, best)

                        del p, code_map1, level_map1, zero1, asym_map1, scale1, q, err

                    del x, xmax, xmin, best
                    torch.cuda.empty_cache()
                
                elif self.scale_method == 'log_sqrt2':
                    for i in range(int(.01 * 100)): # The scaling factor search is disabled here
                        p = 1 - i / 100

                        code_map1, level_map1, zero1, asym_map1 = self.get_code_map(x, xmax.unsqueeze(1), xmin.unsqueeze(1), 0, self.n_bits, self.sym)
                        
                        scale1 = torch.ones([x.shape[0], 1], device=x.device) * p
                        scale1 = scale1.view(target_shape) # scale1 shape: [num_channels, 1, 1, 1]

                        # Quantize
                        q = log_quantize(x, scale=scale1, zero=zero1, method=self.scale_method, code_map=code_map1, level_map=level_map1, asym_map=asym_map1, hardware_approx=self.hardware_approx)
                        q -= x
                        q.abs_()
                        q.pow_(2)
                        err = torch.sum(q.view(q.shape[0], -1), dim=-1, keepdim=True)

                        tmp = (err < best)
                        tmp = tmp.view(target_shape)

                        scale = torch.where(tmp, scale1, scale)
                        zero = torch.where(tmp, zero1, zero)
                        code_map = torch.where(tmp, code_map1, code_map)
                        level_map = torch.where(tmp, level_map1, level_map)
                        asym_map = torch.where(tmp, asym_map1, asym_map)
                        best = torch.minimum(err, best)

                        del p, code_map1, level_map1, zero1, asym_map1, scale1, q, err

                    del x, xmax, xmin, best
                    torch.cuda.empty_cache()

                elif self.scale_method == 'log_dynamic':
                    for i in range(int(.01 * 100)): # The scaling factor search is disabled here
                        for code_no_2 in range(2 ** (self.n_bits - 1), -1, -2):                        
                            p = 1 - i / 100

                            code_map1, level_map1, zero1, asym_map1 = self.get_code_map(x, xmax.unsqueeze(1), xmin.unsqueeze(1), code_no_2, self.n_bits, self.sym)
                            
                            scale1 = torch.ones([x.shape[0], 1], device=x.device) * p
                            scale1 = scale1.view(target_shape) # scale1 shape: [num_channels, 1, 1, 1]

                            # Quantize
                            q = log_quantize(x, scale=scale1, zero=zero1, method=self.scale_method, code_map=code_map1, level_map=level_map1, asym_map=asym_map1, hardware_approx=self.hardware_approx)
                            q -= x
                            q.abs_()
                            q.pow_(2)
                            err = torch.sum(q.view(q.shape[0], -1), dim=-1, keepdim=True)

                            tmp = (err < best)
                            tmp = tmp.view(target_shape)

                            scale = torch.where(tmp, scale1, scale)
                            zero = torch.where(tmp, zero1, zero)
                            code_map = torch.where(tmp, code_map1, code_map)
                            level_map = torch.where(tmp, level_map1, level_map)
                            asym_map = torch.where(tmp, asym_map1, asym_map)
                            best = torch.minimum(err, best)

                            del p, code_map1, level_map1, zero1, asym_map1, scale1, q, err

                        del x, xmax, xmin, best
                        torch.cuda.empty_cache()

                else:
                    raise NotImplementedError

        return scale, zero, code_map, level_map, asym_map

    def get_code_map(self, x:torch.Tensor, x_max:torch.Tensor, x_min:torch.Tensor, code_no_2:int, n_bits:int, sym:bool):
        if code_no_2 >= math.pow(2, n_bits - 1):
            max_val = torch.floor(torch.log2(x_max + 1e-32) + 0.5)
            neg_max_val = torch.floor(torch.log2(-x_min + 1e-32) + 0.5)

            if sym:
                asym_map = torch.zeros_like(x_max)
            else:
                asym_map = torch.clamp((torch.abs(max_val - neg_max_val)), max=math.pow(2, n_bits))
            
            max_val = torch.maximum(max_val, neg_max_val)
            min_val = max_val - math.pow(2, (n_bits - 1)) + 1   

            c_map = torch.ones_like(x, dtype=torch.bool)
            zero_point = min_val
            zero_point = torch.stack([zero_point, zero_point], dim=0)
            l_map = torch.ones_like(x_max) * math.pow(2, (n_bits - 1))     
            l_map = torch.stack([l_map, l_map], dim=0)

        elif code_no_2 <= 0:
            max_val = torch.floor(2 * (torch.log2(x_max + 1e-32)) + 0.5)
            neg_max_val = torch.floor(2 * (torch.log2(-x_min + 1e-32)) + 0.5)

            if sym:
                asym_map = torch.zeros_like(x_max)
            else:
                asym_map = torch.clamp((torch.abs(max_val - neg_max_val)), max=math.pow(2, n_bits))

            max_val = torch.maximum(max_val, neg_max_val)
            min_val = max_val - math.pow(2, (n_bits - 1)) + 1

            c_map = torch.zeros_like(x, dtype=torch.bool) 
            zero_point = min_val
            zero_point = torch.stack([zero_point, zero_point], dim=0)
            l_map = torch.ones_like(x_max) * math.pow(2, (n_bits - 1))
            l_map = torch.stack([l_map, l_map], dim=0)

        else:
            code_no_root2 = math.pow(2, (n_bits - 1)) - code_no_2

            pos_max_val_root2 = torch.floor(2 * (torch.log2(x_max + 1e-32)) + 0.5)
            neg_max_val_root2 = torch.floor(2 * (torch.log2(-x_min + 1e-32)) + 0.5)
            max_val_root2 = torch.maximum(pos_max_val_root2, neg_max_val_root2)

            min_val_root2 = max_val_root2 - code_no_root2 + 1
            max_val_2 = (max_val_root2 - code_no_root2) // 2
            min_val_2 = max_val_2 - code_no_2 + 1
            sl_index = (min_val_root2 / 2 + max_val_2) / 2
            sl = torch.pow(2, sl_index)

            input_abs = torch.abs(x)
            dim_diff = x.dim() - sl.dim()
            target_shape = sl.shape + (1,) * dim_diff
            c_map = input_abs <= sl.view(target_shape)

            zero_point_2 = min_val_2
            zero_point_root2 = min_val_root2
            zero_point = torch.stack([zero_point_2, zero_point_root2], dim=0)

            l_map_2 = torch.ones_like(x_max) * code_no_2
            l_map_root2 = torch.ones_like(x_max) * code_no_root2
            l_map = torch.stack([l_map_2, l_map_root2], dim=0)

            if sym:
                asym_map = torch.zeros_like(x_max)
            else:
                small_max = torch.minimum(torch.abs(x_min), x_max)
                small_max_val = torch.where(small_max > sl, torch.floor(2 * (torch.log2(small_max + 1e-32)) + 0.5), torch.floor(torch.log2(small_max + 1e-32) + 0.5))
                asym_map = torch.where(small_max > sl, (torch.abs(max_val_root2 - small_max_val)), (code_no_root2 + torch.abs(max_val_2 - small_max_val)))
                asym_map = torch.clamp(asym_map, max=math.pow(2, n_bits))

        # Different from memory intense LLM, Reshape the code map, zero point, and level map to speedup 
        dim_diff = x.dim() - asym_map.dim()
        target_shape = asym_map.shape + (1,) * dim_diff        
        zero_point = torch.where(c_map, zero_point[0].view(target_shape), zero_point[1].view(target_shape)) # zero_point shape: [num_channels, 1/in_channels, kernel_size/1, kernel_size/1]
        l_map = torch.where(c_map, l_map[0].view(target_shape), l_map[1].view(target_shape)) # l_map shape: [num_channels, 1/in_channels, kernel_size/1, kernel_size/1]
        c_map = torch.where(c_map, 2, math.sqrt(2)) # c_map shape: [num_channels, 1/in_channels, kernel_size/1, kernel_size/1]
        asym_map = asym_map.view(target_shape) # asym_map shape: [num_channels, 1, 1, 1]

        return c_map, l_map, zero_point, asym_map


    def linear_quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q


    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits


    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)


class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, se_module=None):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
            self.static_padding = None
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
            self.static_padding = None
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.se_module = se_module
        self.extra_repr = org_module.extra_repr

    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias

        if self.static_padding is not None:
            input = self.static_padding(input)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        # disable act quantization is designed for convolution before elemental-wise operation,
        # in that case, we apply activation function and quantization after ele-wise op.
        if self.se_module is not None:
            out = self.se_module(out)

        out = self.activation_function(out)
        if self.disable_act_quant:
            return out
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

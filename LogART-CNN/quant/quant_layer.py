import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from torch.autograd import Variable
from efficientnet_pytorch.utils import Conv2dStaticSamePadding
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

        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        self.hardware_approx = hardware_apprx

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            if self.leaf_param:
                delta, self.zero_point, self.code_map, self.level_map = self.init_quantization_scale(x, self.channel_wise) 
                # self.delta = torch.nn.Parameter(delta)
                # self.zero_point = torch.nn.Parameter(self.zero_point)
            else:
                self.delta, self.zero_point, self.code_map, self.level_map = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        if self.scale_method in {'linear_mse', 'linear_minmax'}:
            x_int = round_ste(x / self.delta) + self.zero_point
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
            x_dequant = (x_quant - self.zero_point) * self.delta

        elif self.scale_method in {'log_2', 'log_sqrt2', 'dynamic'}:
            # start quantization
            s = torch.sign(x.detach())
            x_log = torch.div(torch.log(torch.abs(x) + 1e-32), torch.log(self.code_map))
            x_int = round_ste(x_log)
            # Clamp Low
            x_clamp = torch.clamp(x_int - self.zero_point, 0, self.n_levels/2 - 1)
            # Clamp High
            x_clamp = torch.clamp(x_clamp - self.level_map, -self.n_levels/2, -1)
            x_quant = torch.pow(self.code_map, x_clamp + self.level_map + self.zero_point)

            if self.hardware_approx:
                approx_factor = 1.5 / math.sqrt(2)
                sqrt2_flag = (self.code_map == math.sqrt(2)) & ((x_clamp + self.level_map 
                                                                 + self.zero_point) % 2 != 0)
                x_quant = torch.where(sqrt2_flag, x_quant * approx_factor , x_quant)

            x_dequant = x_quant * s

        else:
            raise NotImplementedError('scale_method {} not supported'.format(self.scale_method))

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]

            delta = torch.zeros_like(x_clone)
            zero_point = torch.zeros_like(x_clone)
            code_map = torch.zeros_like(x_clone)
            level_map = torch.zeros_like(x_clone)

            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c], code_map[c], level_map[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
        else:
            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax

                delta_val = float(x_max - x_min) / (self.n_levels - 1)
                if delta_val < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta_val = 1e-8

                zero_point_val = round(-x_min / delta_val)
                
                zero_point = torch.ones_like(x) * zero_point_val
                delta = torch.ones_like(x) * delta_val
                code_map = torch.zeros_like(x)
                level_map = torch.zeros_like(x)

            elif self.scale_method == 'linear_mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(80):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.linear_quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta_val = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point_val = (- new_min / delta_val).round()

                zero_point = torch.ones_like(x) * zero_point_val
                delta = torch.ones_like(x) * delta_val
                code_map = torch.zeros_like(x)
                level_map = torch.zeros_like(x)

            elif self.scale_method == 'log_2':
                code_map, level_map, zero_point = self.get_code_map(x, code_no_2=2 ** (self.n_bits - 1))
                delta = zero_point
            
            elif self.scale_method == 'log_sqrt2':
                code_map, level_map, zero_point = self.get_code_map(x, code_no_2=0)
                delta = zero_point

            elif self.scale_method == 'dynamic':
                nmse_list = []
                for code2 in range(0, 2 ** (self.n_bits - 1) + 1):
                    qw = self.log_quantize(x, code_no_2=code2)
                    v_nmse = nmse(x.detach().cpu().numpy(), qw.detach().cpu().numpy())
                    nmse_list.append(v_nmse)

                nmse_min = min(nmse_list)
                code_no_2 = nmse_list.index(nmse_min)
                code_map, level_map, zero_point = self.get_code_map(x, code_no_2=code_no_2)
                delta = zero_point
            else:
                raise NotImplementedError

        return delta, zero_point, code_map, level_map

    def get_code_map(self, x, code_no_2):
        abs_value = (x.detach()).abs().view(-1)
        v = abs_value.max()

        if code_no_2 >= math.pow(2, (self.n_bits - 1)):
            max_val = torch.floor(torch.log2(v + 1e-32) + 0.5)
            min_val = max_val - math.pow(2, (self.n_bits - 1)) + 1

            c_map = torch.ones_like(x) * 2
            zero_point = torch.ones_like(x) * min_val
            l_map = torch.ones_like(x) * self.n_levels/2

        elif code_no_2 <= 0:
            max_val = torch.floor(2 * (torch.log2(v + 1e-32)) + 0.5)
            min_val = max_val - math.pow(2, (self.n_bits - 1)) + 1

            c_map = torch.ones_like(x) * math.sqrt(2)
            zero_point = torch.ones_like(x) * min_val
            l_map = torch.ones_like(x) * self.n_levels/2

        else:
            code_no_root2 = math.pow(2, (self.n_bits - 1)) - code_no_2

            max_val_root2 = torch.floor(2 * (torch.log2(v + 1e-32)) + 0.5)
            min_val_root2 = max_val_root2 - code_no_root2 + 1
            max_val_2 = (max_val_root2 - code_no_root2) // 2
            min_val_2 = max_val_2 - code_no_2 + 1
            sl_index = (min_val_root2 / 2 + max_val_2) / 2
            sl = torch.pow(2, sl_index)

            input_abs = torch.abs(x)
            flag_2 = input_abs <= sl

            c_map_2 = torch.ones_like(x) * 2
            c_map_root2 = torch.ones_like(x) * math.sqrt(2)
            c_map_2[~flag_2] = 0
            c_map_root2[flag_2] = 0
            c_map = c_map_2 + c_map_root2

            zero_point_2 = torch.ones_like(x) * min_val_2
            zero_point_root2 = torch.ones_like(x) * min_val_root2
            zero_point_2[~flag_2] = 0
            zero_point_root2[flag_2] = 0
            zero_point = zero_point_2 + zero_point_root2

            l_map_2 = torch.ones_like(x) * code_no_2
            l_map_root2 = torch.ones_like(x) * code_no_root2
            l_map_2[~flag_2] = 0
            l_map_root2[flag_2] = 0
            l_map = l_map_2 + l_map_root2

        return c_map, l_map, zero_point

    def linear_quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q
    
    def log_quantize(self, x, code_no_2):
        if isinstance(code_no_2, Variable):
            code_no_2 = code_no_2.data.cpu().numpy()
    
        c_map, l_map, zero_point = self.get_code_map(x, code_no_2)
        s = torch.sign(x.detach())
        x_log = torch.div(torch.log(torch.abs(x) + 1e-32), torch.log(c_map))
        x_int = round_ste(x_log)
        # Clamp Low
        x_clamp = torch.clamp(x_int - zero_point, 0, self.n_levels/2 - 1)
        # Clamp High
        x_clamp = torch.clamp(x_clamp - l_map, -self.n_levels/2, -1)
        x_quant = torch.pow(c_map, x_clamp + l_map + zero_point)

        if self.hardware_approx:
            approx_factor = 1.5 / math.sqrt(2)
            sqrt2_flag = (c_map == math.sqrt(2)) & ((x_clamp + l_map + zero_point) % 2 != 0)
            x_quant = torch.where(sqrt2_flag, x_quant * approx_factor , x_quant)
            
        x_dequant = x_quant * s

        return x_dequant


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
            if isinstance(org_module, Conv2dStaticSamePadding):
                self.static_padding = org_module.static_padding
            else:
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

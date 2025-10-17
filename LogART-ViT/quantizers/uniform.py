import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from quantizers._ste import *
from torch.autograd import Variable
from skfuzzy.image import nmse
import math


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
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


def get_similarity(tensor_raw, tensor_sim, metric=None, raw_grad=None):
    if len(tensor_raw.shape) == 1:
        reduction =' all'
    else:
        reduction = 'none'

    if metric == "mae":
        similarity = -torch.abs(tensor_raw - tensor_sim).mean()
    elif metric == "mse":
        similarity = -lp_loss(tensor_sim, tensor_raw, p=2, reduction=reduction)
    else:
        raise NotImplementedError(f"metric {metric} not implemented!")
    return similarity


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
        if len(x.shape) != 2:
            raise NotImplementedError("Only support 2D weights for now!")
            
        x_log = torch.div(torch.log(torch.abs(x / scale) + 1e-32), torch.log(code_map))
        x_int = torch.round(x_log)
        # Clamp Low
        code_map_bool = code_map == 2
        x_clamp_1 = torch.clamp(x_int - zero, 0 - (1 + asym_map / 2) * code_map_bool, math.inf * torch.ones_like(x))
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


class UniformQuantizer(nn.Module):
    def __init__(self, n_bits: int = 8, symmetric: bool = True, channel_wise: bool = False, 
                 scale_method: str = 'log_dynamic', hardware_approx: bool = False):
        super().__init__()
        self.sym = symmetric
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.channel_wise = channel_wise
        self.drop_prob = 1.0
        self.inited = False
        self.training_mode = False
        self.scale_method = scale_method
        self.hardware_approx = hardware_approx

        self.scale = None
        self.zero_point = None
        self.code_map = None
        self.level_map = None
        self.asym_map = None

    def init_training(self):
        self.training_mode = True

    def end_training(self):
        self.training_mode = False
        
    def forward(self, x):
        if self.n_bits == 32:
            return x
               
        if self.inited == False:
            self.init_quantization_scale(x, self.channel_wise)
            self.inited = True
        assert self.inited
        
        if self.training_mode and self.drop_prob < 1.0:
            x_orig = x

        if self.scale_method in {'log_2', 'log_sqrt2', 'log_dynamic'}:
            if len(x.shape) == 2:
                temp_weight = x.view(x.shape[0], -1)
            elif len(x.shape) == 3:
                temp_weight = x.flatten(0, 1)
            elif len(x.shape) == 4:
                temp_weight = x.flatten(1, 3)
            else:
                raise NotImplementedError("Only support 2D, 3D and 4D weights for now!")

            x_dequant = log_quantize(temp_weight, self.scale, self.zero_point, self. scale_method, self.code_map, self.level_map, self.asym_map, self.hardware_approx)

            if len(x.shape) == 2:
                pass
            elif len(x.shape) == 3:
                x_dequant.reshape(x.shape)
            elif len(x.shape) == 4:
                x_dequant.reshape(x.shape)
            else:
                raise NotImplementedError("Only support 2D, 3D and 4D weights for now!")

        if self.training_mode and self.drop_prob < 1.0:
            x_prob = torch.where(torch.rand_like(x) < self.drop_prob, x_dequant, x_orig)
            return x_prob
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        if not channel_wise:
            raise NotImplementedError("Only support channel-wise quantization for now!")
            
        else:
            if 'log' in self.scale_method:
                # Handle different dimensions of the weight
                if len(x.shape) == 2:
                    weight_reshape = x.view(x.shape[0], -1)
                elif len(x.shape) == 3:
                    weight_reshape = x.flatten(0, 1)
                elif len(x.shape) == 4:
                    weight_reshape = x.flatten(1, 3)
                else:
                    raise NotImplementedError("Only support 2D and 4D weights for now!")

                # get the channel-wise max of the weight
                tmp = torch.zeros(weight_reshape.shape[0], device=x.device)
                xmin = torch.minimum(weight_reshape.min(1)[0], tmp)
                xmax = torch.maximum(weight_reshape.max(1)[0], tmp)
                if self.sym:
                    xmax = torch.maximum(torch.abs(xmin), xmax)
                    xmin = -xmax
                
                tmp = (xmin == 0) & (xmax == 0)
                # assert that xmin and xmax are not both 0
                assert not torch.any(tmp), "xmin and xmax are both 0"

                # Initialize Scale, Zero, Code Map, and Level Map
                dim_diff = weight_reshape.dim() - (xmin.unsqueeze(1)).dim()
                target_shape = (xmin.unsqueeze(1)).shape + (1,) * dim_diff

                best = torch.full_like(xmin.unsqueeze(1), 1e10, device=x.device) # best shape: [num_channels, 1] 
                delta = torch.zeros_like((xmin.unsqueeze(1)).view(target_shape), device=x.device) # delta shape: [num_channels, 1, 1, 1]
                zero_point = torch.zeros_like(weight_reshape, device=x.device) # zero_point shape: [num_channels, 1/in_channels, kernel_size/1, kernel_size/1]
                code_map = torch.zeros_like(weight_reshape, device=x.device) # code_map shape: [num_channels, 1/in_channels, kernel_size/1, kernel_size/1]
                level_map = torch.zeros_like(weight_reshape, device=x.device) # level_map shape: [num_channels, 1/in_channels, kernel_size/1, kernel_size/1]
                asym_map = torch.zeros_like((xmin.unsqueeze(1)).view(target_shape), device=x.device) # asym_map shape: [num_channels, 1, 1, 1]

                if self.scale_method == 'log_2':
                    for i in range(int(.01 * 100)): # The scaling factor search is disabled here
                        p = 1 - i / 100

                        code_map1, level_map1, zero_point1, asym_map1 = self.get_code_map(weight_reshape, xmax.unsqueeze(1), xmin.unsqueeze(1), 2 ** (self.n_bits - 1), self.n_bits, self.sym)
                        
                        scale1 = torch.ones([weight_reshape.shape[0], 1], device=x.device) * p
                        scale1 = scale1.view(target_shape) # scale1 shape: [num_channels, 1, 1, 1]

                        # Quantize
                        q = log_quantize(weight_reshape, scale=scale1, zero=zero_point1, method=self.scale_method, code_map=code_map1, level_map=level_map1, asym_map=asym_map1, hardware_approx=self.hardware_approx)
                        q -= weight_reshape
                        q.abs_()
                        q.pow_(2)
                        err = torch.sum(q.view(q.shape[0], -1), dim=-1, keepdim=True)

                        tmp = (err < best)
                        tmp = tmp.view(target_shape)

                        delta = torch.where(tmp, scale1, delta)
                        zero_point = torch.where(tmp, zero_point1, zero_point)
                        code_map = torch.where(tmp, code_map1, code_map)
                        level_map = torch.where(tmp, level_map1, level_map)
                        asym_map = torch.where(tmp, asym_map1, asym_map)
                        best = torch.minimum(err, best)

                        del p, code_map1, level_map1, zero_point1, asym_map1, scale1, q, err

                    del xmax, xmin, best
                    torch.cuda.empty_cache()

                elif self.scale_method == 'log_sqrt2':
                    for i in range(int(.01 * 100)): # The scaling factor search is disabled here
                        p = 1 - i / 100

                        code_map1, level_map1, zero_point1, asym_map1 = self.get_code_map(weight_reshape, xmax.unsqueeze(1), xmin.unsqueeze(1), 0, self.n_bits, self.sym)
                        
                        scale1 = torch.ones([weight_reshape.shape[0], 1], device=x.device) * p
                        scale1 = scale1.view(target_shape) # scale1 shape: [num_channels, 1, 1, 1]

                        # Quantize
                        q = log_quantize(weight_reshape, scale=scale1, zero=zero_point1, method=self.scale_method, code_map=code_map1, level_map=level_map1, asym_map=asym_map1, hardware_approx=self.hardware_approx)
                        q -= weight_reshape
                        q.abs_()
                        q.pow_(2)
                        err = torch.sum(q.view(q.shape[0], -1), dim=-1, keepdim=True)

                        tmp = (err < best)
                        tmp = tmp.view(target_shape)

                        delta = torch.where(tmp, scale1, delta)
                        zero_point = torch.where(tmp, zero_point1, zero_point)
                        code_map = torch.where(tmp, code_map1, code_map)
                        level_map = torch.where(tmp, level_map1, level_map)
                        asym_map = torch.where(tmp, asym_map1, asym_map)
                        best = torch.minimum(err, best)

                        del p, code_map1, level_map1, zero_point1, asym_map1, scale1, q, err

                    del xmax, xmin, best
                    torch.cuda.empty_cache()

                elif self.scale_method == 'log_dynamic':
                    for i in range(int(0.01 * 100)): # The scaling factor search is disabled here
                        for code_no_2 in range(2 ** (self.n_bits - 1), 2 ** (self.n_bits - 1)-1, -2):                        
                            p = 1 - i / 100

                            code_map1, level_map1, zero_point1, asym_map1 = self.get_code_map(weight_reshape, xmax.unsqueeze(1), xmin.unsqueeze(1), code_no_2, self.n_bits, self.sym)
                            
                            scale1 = torch.ones([weight_reshape.shape[0], 1], device=x.device) * p
                            scale1 = scale1.view(target_shape) # scale1 shape: [num_channels, 1, 1, 1]

                            # Quantize
                            q = log_quantize(weight_reshape, scale=scale1, zero=zero_point1, method=self.scale_method, code_map=code_map1, level_map=level_map1, asym_map=asym_map1, hardware_approx=self.hardware_approx)
                            q -= weight_reshape
                            q.abs_()
                            q.pow_(2)
                            err = torch.sum(q.view(q.shape[0], -1), dim=-1, keepdim=True)

                            tmp = (err < best)
                            tmp = tmp.view(target_shape)

                            delta = torch.where(tmp, scale1, delta)
                            zero_point = torch.where(tmp, zero_point1, zero_point)
                            code_map = torch.where(tmp, code_map1, code_map)
                            level_map = torch.where(tmp, level_map1, level_map)
                            asym_map = torch.where(tmp, asym_map1, asym_map)
                            best = torch.minimum(err, best)

                            del p, code_map1, level_map1, zero_point1, asym_map1, scale1, q, err

                    del xmax, xmin, best
                    torch.cuda.empty_cache()
                
                else:
                    raise NotImplementedError(f"Scale method {self.scale_method} not implemented!")
                    
            else:
                raise NotImplementedError

        self.scale = delta
        self.zero_point = zero_point
        self.code_map = code_map
        self.level_map = level_map
        self.asym_map = asym_map
        self.inited = True


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


    def __repr__(self):
        return f'{self.__class__.__name__}(n_bits={self.n_bits}, sym={self.sym}, channel_wise={self.channel_wise})'

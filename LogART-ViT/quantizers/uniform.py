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

class UniformQuantizer(nn.Module):
    def __init__(self, n_bits: int = 8, symmetric: bool = True, channel_wise: bool = False, 
                 scale_method: str = 'linear', hardware_approx: bool = False):
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

    def init_training(self):
        self.training_mode = True

    def end_training(self):
        self.training_mode = False
        
    def forward(self, x):
        if self.n_bits == 32:
            return x
               
        if self.inited == False:
            self.scale, self.zero_point, self.code_map, self.level_map = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True
        assert self.inited
        if self.training_mode and self.drop_prob < 1.0:
            x_orig = x

        if self.scale_method in {'linear_mse', 'linear_minmax'}:
            x_int = round_ste(x / self.scale) + self.zero_point
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
            x_dequant = (x_quant - self.zero_point) * self.scale
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
        if self.training_mode and self.drop_prob < 1.0:
            x_prob = torch.where(torch.rand_like(x) < self.drop_prob, x_dequant, x_orig)
            return x_prob
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            delta = torch.zeros_like(x_clone)
            zero_point = torch.zeros_like(x_clone)
            code_map = torch.zeros_like(x_clone)
            level_map = torch.zeros_like(x_clone)

            if len(x_clone.shape) == 2:
                n_channels = x_clone.shape[0]           
                # determine the scale and zero point channel-by-channel
                for c in range(n_channels):
                    delta[c], zero_point[c], code_map[c], level_map[c] = self.init_quantization_scale(x_clone[c],
                                                                                                  channel_wise=False)
            if len(x_clone.shape) == 3:
                n_channels_1 = x_clone.shape[0]
                n_channels_2 = x_clone.shape[1]
                for c1 in range(n_channels_1):
                    for c2 in range(n_channels_2):
                        delta[c1, c2, :], zero_point[c1, c2, :], code_map[c1, c2, :], level_map[c1, c2, :] = self.init_quantization_scale(x_clone[c1, c2, :],channel_wise=False)

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

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax

                best_score = 1e+10
                for i in range(80): #80
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
                    score = get_similarity(x.detach(), qw, 'mse')
                    nmse_list.append(score)

                nmse_min = max(nmse_list)
                code_no_2 = nmse_list.index(nmse_min)
                code_map, level_map, zero_point = self.get_code_map(x, code_no_2=code_no_2)
                delta = zero_point
            else:
                raise NotImplementedError

        return delta, zero_point, code_map, level_map

    def get_code_map(self, x, code_no_2):
        abs_value = (x.detach()).abs().reshape(-1)
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
        x_clamp = torch.clamp(x_int - zero_point, 0, self.n_levels / 2 - 1)
        # Clamp High
        x_clamp = torch.clamp(x_clamp - l_map, -self.n_levels / 2, -1)

        if self.hardware_approx:
            approx_factor = 1.5 / math.sqrt(2)
            sqrt2_flag = (c_map == math.sqrt(2)) & ((x_clamp + l_map 
                                                        + zero_point) % 2 != 0)
            x_quant = torch.where(sqrt2_flag, x_quant * approx_factor , x_quant)

        x_quant = torch.pow(c_map, x_clamp + l_map + zero_point)
        x_dequant = x_quant * s

        return x_dequant



    def __repr__(self):
        return f'{self.__class__.__name__}(n_bits={self.n_bits}, sym={self.sym}, channel_wise={self.channel_wise})'

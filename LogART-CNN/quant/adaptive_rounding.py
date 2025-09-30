import torch
from torch import nn
from quant.quant_layer import UniformAffineQuantizer, round_ste

import math


class AdaRoundQuantizer(nn.Module):
    """
    Adaptive Rounding Quantizer, used to optimize the rounding policy
    by reconstructing the intermediate output.
    Based on
     Up or Down? Adaptive Rounding for Post-Training Quantization: https://arxiv.org/abs/2004.10568

    :param uaq: UniformAffineQuantizer, used to initialize quantization parameters in this quantizer
    :param round_mode: controls the forward pass in this quantizer
    :param weight_tensor: initialize alpha
    """

    def __init__(self, uaq: UniformAffineQuantizer, weight_tensor: torch.Tensor, 
                 round_mode='learned_round_sigmoid'):
        super(AdaRoundQuantizer, self).__init__()
        # copying all attributes from UniformAffineQuantizer
        self.n_bits = uaq.n_bits
        self.sym = uaq.sym
        self.delta = uaq.delta
        self.zero_point = uaq.zero_point
        self.n_levels = uaq.n_levels
        self.code_map = uaq.code_map
        self.level_map = uaq.level_map
        self.asym_map = uaq.asym_map
        self.scale_method = uaq.scale_method
        self.hardware_approx = uaq.hardware_approx

        self.round_mode = round_mode
        self.alpha = None
        self.soft_targets = False

        # params for sigmoid function
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3
        self.init_alpha(x=weight_tensor.clone())


    def forward(self, x):
        if self.round_mode == 'learned_hard_sigmoid':
            # Flooring
            if self.scale_method in {'log_2', 'log_sqrt2', 'log_dynamic'}:
                x_log = torch.div(torch.log(torch.abs(x / self.delta) + 1e-32), torch.log(self.code_map))
                x_floor = torch.floor(x_log)
            else:
                raise NotImplementedError('scale_method {} not supported'.format(self.scale_method))
            
            # Add the learnable rounding
            if self.soft_targets:
                x_int = x_floor + self.get_soft_targets()
            else:
                x_int = x_floor + (self.alpha >= 0).float()
        
        else:
            raise ValueError('Wrong rounding mode')

        # Clamping and Dequantization
        if self.scale_method in {'log_sqrt2'}:
            # Clamp Low
            x_clamp_1 = torch.clamp(x_int - self.zero_point, -1 - self.asym_map / 2, math.inf * torch.ones_like(x))
            # Clamp High
            x_clamp = torch.clamp(x_clamp_1 - self.level_map, -math.inf, -1)

            x_quant = torch.pow(self.code_map, x_clamp + self.level_map + self.zero_point)

            if self.hardware_approx:
                approx_factor = 1.5 / math.sqrt(2)
                # approx_factor = (1 + 1/2 - 2**(-4) ) / math.sqrt(2)
                # approx_factor = (1 + 1/2 - 2**(-4) - 2**(-6)) / math.sqrt(2)
                sqrt2_flag = (self.code_map == math.sqrt(2)) & ((x_clamp + self.level_map 
                                                    + self.zero_point) % 2 != 0)
                x_quant = torch.where(sqrt2_flag, x_quant * approx_factor , x_quant)   

            # Set the minimum value to 0
            s = torch.sign(x.detach())
            zero_flag = (x_clamp_1 <= -1 - self.asym_map / 2)
            x_quant = torch.where(zero_flag, torch.zeros_like(x_quant), x_quant)

            x_float_q = x_quant * s * self.delta

        elif self.scale_method in {'log_2', 'log_dynamic'}:
            # Clamp Low
            x_clamp_1 = torch.clamp(x_int - self.zero_point, 0 - (1 + self.asym_map / 2) * self.code_map, math.inf * torch.ones_like(x))
            # Clamp High
            x_clamp = torch.clamp(x_clamp_1 - self.level_map, -math.inf, -1)

            x_quant = torch.pow(self.code_map, x_clamp + self.level_map + self.zero_point)

            if self.hardware_approx:
                approx_factor = 1.5 / math.sqrt(2)
                # approx_factor = (1 + 1/2 - 2**(-4) ) / math.sqrt(2)
                # approx_factor = (1 + 1/2 - 2**(-4) - 2**(-6)) / math.sqrt(2)
                sqrt2_flag = (self.code_map == math.sqrt(2)) & ((x_clamp + self.level_map 
                                                        + self.zero_point) % 2 != 0)
                x_quant = torch.where(sqrt2_flag, x_quant * approx_factor , x_quant)

            # Set the minimum value to 0
            s = torch.sign(x.detach())
            zero_flag = (x_clamp_1 <= -1 - self.asym_map / 2) & (self.code_map == 2)
            x_quant = torch.where(zero_flag, torch.zeros_like(x_quant), x_quant)

            x_float_q = x_quant * s * self.delta

        else:
            raise NotImplementedError('scale_method {} not supported'.format(self.scale_method))

        return x_float_q


    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)


    def init_alpha(self, x: torch.Tensor):
        if self.scale_method in {'log_2', 'log_sqrt2', 'log_dynamic'}:
            x_log = torch.div(torch.log(torch.abs(x / self.delta) + 1e-32), torch.log(self.code_map))
            x_floor = torch.floor(x_log)
            if self.round_mode == 'learned_hard_sigmoid':
                print('Init alpha to be FP32')
                rest = x_log - x_floor  # rest of rounding [0, 1)
                alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  
                self.alpha = nn.Parameter(alpha)
            else:
                raise NotImplementedError
        
        else:
            raise NotImplementedError('scale_method {} not supported'.format(self.scale_method))

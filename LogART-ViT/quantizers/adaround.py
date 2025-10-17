from huggingface_hub import webhook_endpoint
import torch
import math
from torch import nn
from quantizers.uniform import UniformQuantizer
from quantizers._ste import round_ste


class AdaRoundQuantizer(nn.Module):
    """
    Adaptive Rounding Quantizer, used to optimize the rounding policy
    by reconstructing the intermediate output.
    Based on
     Up or Down? Adaptive Rounding for Post-Training Quantization: https://arxiv.org/abs/2004.10568

    :param uq: UniformQuantizer, used to initialize quantization parameters in this quantizer
    :param round_mode: controls the forward pass in this quantizer
    :param weight_tensor: initialize alpha
    """

    def __init__(self, uq: UniformQuantizer, weight_tensor: torch.Tensor, round_mode='learned_hard_sigmoid'):
        super().__init__()
        # copying all attributes from UniformQuantizer
        self.n_bits = uq.n_bits
        self.n_levels = uq.n_levels
        self.channel_wise = uq.channel_wise
        self.sym = uq.sym
        self.scale = uq.scale
        self.zero_point = uq.zero_point
        self.scale_method = uq.scale_method
        self.code_map = uq.code_map
        self.level_map = uq.level_map
        self.asym_map = uq.asym_map
        self.round_mode = round_mode
        self.alpha = None
        self.soft_targets = False
        self.hardware_approx = uq.hardware_approx
        # params for sigmoid function
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3
        self.init_alpha(x=weight_tensor.clone())

    def forward(self, x):
        if self.round_mode == 'learned_hard_sigmoid':
            if len(x.shape) == 2:
                weight_reshape = x.view(x.shape[0], -1)
            elif len(x.shape) == 3:
                weight_reshape = x.flatten(0, 1)
            elif len(x.shape) == 4:
                weight_reshape = x.flatten(1, 3)
            else:
                raise NotImplementedError("Only support 2D, 3D and 4D weights for now!")

            if self.scale_method in {'log_2', 'log_sqrt2', 'log_dynamic'}:
                x_log = torch.div(torch.log(torch.abs(weight_reshape / self.scale) + 1e-32), torch.log(self.code_map))
                x_floor = torch.floor(x_log)
            else:
                raise NotImplementedError('scale_method {} not supported'.format(self.scale_method))
            
            if self.soft_targets:
                x_int = x_floor + self.get_soft_targets()
            else:
                x_int = x_floor + (self.alpha >= 0).float()
        else:
            raise ValueError('Wrong rounding mode')
        
        if self.scale_method in {'log_sqrt2'}:
            # Clamp Low
            x_clamp_1 = torch.clamp(x_int - self.zero_point, -1 - self.asym_map / 2, math.inf * torch.ones_like(weight_reshape))
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
            s = torch.sign(weight_reshape.detach())
            zero_flag = (x_clamp_1 <= -1 - self.asym_map / 2)
            x_quant = torch.where(zero_flag, torch.zeros_like(x_quant), x_quant)

            x_float_q = x_quant * s * self.scale

        elif self.scale_method in {'log_2', 'log_dynamic'}:
            # Clamp Low
            code_map_bool = self.code_map == 2
            x_clamp_1 = torch.clamp(x_int - self.zero_point, 0 - (1 + self.asym_map / 2) * code_map_bool, math.inf * torch.ones_like(weight_reshape))
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
            s = torch.sign(weight_reshape.detach())
            zero_flag = (x_clamp_1 <= -1 - self.asym_map / 2) & (self.code_map == 2)
            x_quant = torch.where(zero_flag, torch.zeros_like(x_quant), x_quant)

            x_float_q = x_quant * s * self.scale

        else:
            raise NotImplementedError('scale_method {} not supported'.format(self.scale_method))

        if len(x.shape) == 2:
            pass
        elif len(x.shape) == 3:
            x_float_q = x_float_q.reshape(x.shape)
        elif len(x.shape) == 4:
            x_float_q = x_float_q.reshape(x.shape)
        else:
            raise NotImplementedError("Only support 2D, 3D and 4D weights for now!")
        
        return x_float_q

    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def init_alpha(self, x: torch.Tensor):  
        if len(x.shape) == 2:
            weight_reshape = x.view(x.shape[0], -1)
        elif len(x.shape) == 3:
            weight_reshape = x.flatten(0, 1)
        elif len(x.shape) == 4:
            weight_reshape = x.flatten(1, 3)
        else:
            raise NotImplementedError("Only support 2D, 3D and 4D weights for now!")

        if self.scale_method in {'log_2', 'log_sqrt2', 'log_dynamic'}:
            x_log = torch.div(torch.log(torch.abs(weight_reshape / self.scale) + 1e-32), torch.log(self.code_map))
            x_floor = torch.floor(x_log)
            if self.round_mode == 'learned_hard_sigmoid':
                print('Init alpha to be FP32')
                rest = x_log - x_floor  # rest of rounding [0, 1)
                alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
                self.alpha = nn.Parameter(alpha)
            else:
                raise NotImplementedError

    def get_hard_value(self, x):
        raise NotImplementedError("Be careful when using this functions!")
        init_shape = x.shape
        return ((torch.floor(x.reshape_as(self.alpha) / self.scale) + (self.alpha >= 0).float()) * self.scale).reshape(*init_shape)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_bits={self.n_bits}, sym={self.sym}, channel_wise={self.channel_wise}, round_mode={self.round_mode})'

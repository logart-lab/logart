from sympy.sets.sets import false
import torch
import torch.nn as nn
import math

def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x

def get_code_map(x:torch.Tensor, x_max:torch.Tensor, x_min:torch.Tensor, code_no_2:int, n_bits:int, sym:bool):
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
        c_map = input_abs <= sl.expand(x.shape)

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

    return c_map, l_map, zero_point, asym_map

def quantize(x, scale, zero, method, group_size=-1, code_map=None, level_map=None, asym_map=None, set_zero=1, hardware_approx=False, maxq=None, sym=True, search_scale=False):
    if maxq < 0:
        assert False, "maxq < 0"
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    if 'log' in method:
        original_shape = x.shape
        if group_size > 0:
            assert x.shape[-1] % group_size == 0
            x = x.reshape(-1, group_size)

        if 'sqrt2' in method:
            real_code_map = torch.where(code_map, 2, math.sqrt(2))
            x_log = torch.div(torch.log(torch.abs(x / scale.expand(x.shape)) + 1e-32), torch.log(real_code_map))
            
            real_zero = torch.where(code_map, zero[0].expand(x.shape), zero[1].expand(x.shape))
            real_level_map = torch.where(code_map, level_map[0].expand(x.shape), level_map[1].expand(x.shape))
            x_int = round_ste(x_log)
            # Clamp Low
            x_clamp_1 = torch.clamp(x_int - real_zero, -1 - asym_map.expand(x.shape) / 2, math.inf * torch.ones_like(x))
            del x_int, x_log
            torch.cuda.empty_cache()
            # Clamp High
            x_clamp = torch.clamp(x_clamp_1 - real_level_map, -math.inf, -1)

            x_quant = torch.pow(real_code_map, x_clamp + real_level_map + real_zero)
            
            if hardware_approx:
                approx_factor = 1.5 / math.sqrt(2)
                # approx_factor = (1 + 1/2 - 2**(-4) ) / math.sqrt(2)
                # approx_factor = (1 + 1/2 - 2**(-4) - 2**(-6)) / math.sqrt(2)
                sqrt2_flag = (code_map == 0) & ((x_clamp + real_level_map 
                                                + real_zero) % 2 != 0)
                x_quant = torch.where(sqrt2_flag, x_quant * approx_factor , x_quant)
            # else:
            #     raise ValueError("Only Hardware approximation is supported")
            
            del x_clamp, real_code_map, real_zero, real_level_map
            torch.cuda.empty_cache()

            # Set the minimum value to 0
            s = torch.sign(x.detach())
            zero_flag = (x_clamp_1 <= -1 - asym_map.expand(x.shape) / 2)
            # hazard_flag = (x_clamp_1 == 0 - asym_map.expand(x.shape) // 2) & (s == set_zero) & (asym_map.expand(x.shape) % 2 == 0)
            x_quant[zero_flag] = 0
            # x_quant[hazard_flag] = 0

            x_dequant = x_quant * s * scale.expand(x.shape)

            if group_size > 0:
                x_dequant = x_dequant.reshape(original_shape)

            return x_dequant

        else:
            real_code_map = torch.where(code_map, 2, math.sqrt(2))
            x_log = torch.div(torch.log(torch.abs(x / scale.expand(x.shape)) + 1e-32), torch.log(real_code_map))

            real_zero = torch.where(code_map, zero[0].expand(x.shape), zero[1].expand(x.shape))
            real_level_map = torch.where(code_map, level_map[0].expand(x.shape), level_map[1].expand(x.shape))
            x_int = round_ste(x_log)
            # Clamp Low
            x_clamp_1 = torch.clamp(x_int - real_zero, 0 - (1 + asym_map.expand(x.shape) / 2) * code_map, math.inf * torch.ones_like(x))
            del x_int, x_log
            torch.cuda.empty_cache()
            # Clamp High
            x_clamp = torch.clamp(x_clamp_1 - real_level_map, -math.inf, -1)

            x_quant = torch.pow(real_code_map, x_clamp + real_level_map + real_zero)

            if hardware_approx:
                approx_factor = 1.5 / math.sqrt(2)
                # approx_factor = (1 + 1/2 - 2**(-4) ) / math.sqrt(2)
                # approx_factor = (1 + 1/2 - 2**(-4) - 2**(-6)) / math.sqrt(2)
                sqrt2_flag = (code_map == False) & ((x_clamp + real_level_map 
                                                    + real_zero) % 2 != 0)
                x_quant = torch.where(sqrt2_flag, x_quant * approx_factor , x_quant)
            # else:
            #     raise ValueError("Only Hardware approximation is supported")
            
            del x_clamp, real_code_map, real_zero, real_level_map
            torch.cuda.empty_cache()

            # Set the minimum value to 0
            s = torch.sign(x.detach())
            zero_flag = (x_clamp_1 <= -1 - asym_map.expand(x.shape) / 2) & code_map
            # hazard_flag = (x_clamp_1 == 0 - asym_map.expand(x.shape) // 2) & code_map & (s == set_zero) & (asym_map.expand(x.shape) % 2 == 0)
            x_quant[zero_flag] = 0
            # x_quant[hazard_flag] = 0

            x_dequant = x_quant * s * scale.expand(x.shape)

            if group_size > 0:
                x_dequant = x_dequant.reshape(original_shape)

            return x_dequant
    else:
        assert False, "Quantization method not supported"

class MinMaxQuantizer(nn.Module):
    def __init__(self, shape=1):
        super(MinMaxQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))
        self.register_buffer('zeta', torch.zeros(shape))
        self.register_buffer('code_map', torch.zeros(shape))
        self.register_buffer('level_map', torch.zeros(shape))
        self.register_buffer('asym_map', torch.zeros(shape))
        self.register_buffer('set_zero', torch.ones(shape))

    def configure(self, bits, per_channel=False, sym=True, scale_search=False, method='log_2', group_size=-1, hardware_approx=False):
        self.nbits = bits
        self.maxq = torch.tensor(2 ** bits - 1)
        self.per_channel = per_channel
        self.sym = sym
        self.scale_search = scale_search
        self.method = method
        self.hardware_approx = hardware_approx
        self.group_size = group_size

    def find_params(self, x):
        dev = x.device
        org_x = x
        d_in = org_x.flatten(1).shape[-1]
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.per_channel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)
        
        if 'log' in self.method:
            # Group the weight
            original_shape = x.shape
            if self.group_size > 0:
                assert x.shape[-1] % self.group_size == 0
                x = x.reshape(-1, self.group_size)
            
            # get the max of the weight
            tmp = torch.zeros(x.shape[0], device=dev)
            xmin = torch.minimum(x.min(1)[0], tmp)
            xmax = torch.maximum(x.max(1)[0], tmp)
            if self.sym:
                xmax = torch.maximum(torch.abs(xmin), xmax)
                xmin = -xmax

            tmp = (xmin == 0) & (xmax == 0)
            # assert that xmin and xmax are not both 0
            assert not torch.any(tmp), "xmin and xmax are both 0"

            # Initialize Scale, Zero, Code Map, and Level Map
            scale = torch.zeros([x.shape[0], 1], device=dev)
            zero = torch.zeros([2, x.shape[0], 1], device=dev)
            code_map = torch.zeros_like(x, device=dev, dtype=torch.bool)
            level_map = torch.zeros([2, x.shape[0], 1], device=dev)
            asym_map = torch.zeros([x.shape[0], 1], device=dev)
            best = torch.full([x.shape[0]], float('inf'), device=dev)

            if self.method == 'log_2':    
                for i in range(int(.01 * 100)):
                    p = 1 - i / 100
                    # Get Temporary Scale and Zero
                    code_map1, level_map1, zero1, asym_map1 = get_code_map(x, xmax.unsqueeze(1), xmin.unsqueeze(1), 2 ** (self.nbits - 1), self.nbits, self.sym)
                    scale1 = torch.ones([x.shape[0], 1], device=dev) * p
                    # Quantize
                    q = quantize(x, scale=scale1, zero=zero1, code_map=code_map1, level_map=level_map1, asym_map=asym_map1, hardware_approx=self.hardware_approx, maxq=self.maxq, method=self.method, sym=self.sym)
                    q -= x
                    q.abs_()
                    q.pow_(2)
                    err = torch.sum(q, 1)
                    tmp = err < best
                    if torch.any(tmp):
                        best[tmp] = err[tmp]
                        scale[tmp] = scale1[tmp]
                        zero = torch.where(tmp.unsqueeze(0).unsqueeze(-1), zero1, zero)
                        code_map[tmp] = code_map1[tmp]
                        level_map = torch.where(tmp.unsqueeze(0).unsqueeze(-1), level_map1, level_map)
                        asym_map[tmp] = asym_map1[tmp]
                        
                        del i, p, code_map1, level_map1, zero1, asym_map1, scale1, q, err

                del x, xmax, xmin, tmp, best
                torch.cuda.empty_cache()

            elif self.method == 'log_sqrt2':
                for i in range(int(.01 * 100)):
                    p = 1 - i / 100
                    # Get Temporary Scale and Zero
                    code_map1, level_map1, zero1, asym_map1 = get_code_map(x, xmax.unsqueeze(1), xmin.unsqueeze(1), 0, self.nbits, self.sym)
                    scale1 = torch.ones([x.shape[0], 1], device=dev) * p
                    # Quantize
                    q = quantize(x, scale=scale1, zero=zero1, code_map=code_map1, level_map=level_map1, asym_map=asym_map1, hardware_approx=self.hardware_approx, maxq=self.maxq, method=self.method, sym=self.sym)
                    q -= x
                    q.abs_()
                    q.pow_(2)
                    err = torch.sum(q, 1)
                    tmp = err < best
                    if torch.any(tmp):
                        best[tmp] = err[tmp]
                        scale[tmp] = scale1[tmp]
                        zero = torch.where(tmp.unsqueeze(0).unsqueeze(-1), zero1, zero)
                        code_map[tmp] = code_map1[tmp]
                        level_map = torch.where(tmp.unsqueeze(0).unsqueeze(-1), level_map1, level_map)
                        asym_map[tmp] = asym_map1[tmp]

                        del i, p, code_map1, level_map1, zero1, asym_map1, scale1, q, err

                del x, xmax, xmin, tmp, best
                torch.cuda.empty_cache()

            elif self.method == 'log_dynamic':
                for code_no_2 in range(2 ** (self.nbits - 1), -1, -2):
                    # for i in range(int(.8 * 100)):
                    for i in range(int(0.01 * 100)):
                        p = 1 - i / 100
                        # Get Temporary Scale and Zero
                        code_map1, level_map1, zero1, asym_map1 = get_code_map(x, xmax.unsqueeze(1), xmin.unsqueeze(1), code_no_2, self.nbits, self.sym)
                        scale1 = torch.ones([x.shape[0], 1], device=dev) * p
                        # Quantize
                        q = quantize(x, scale=scale1, zero=zero1, code_map=code_map1, level_map=level_map1, asym_map=asym_map1, hardware_approx=self.hardware_approx, maxq=self.maxq, method=self.method, sym=self.sym)
                        q -= x
                        q.abs_()
                        q.pow_(2)
                        err = torch.sum(q, 1)
                        tmp = err < best
                        if torch.any(tmp):
                            best[tmp] = err[tmp]
                            scale[tmp] = scale1[tmp]
                            zero = torch.where(tmp.unsqueeze(0).unsqueeze(-1), zero1, zero)
                            code_map[tmp] = code_map1[tmp]
                            level_map = torch.where(tmp.unsqueeze(0).unsqueeze(-1), level_map1, level_map)
                            asym_map[tmp] = asym_map1[tmp]
                    
                        del tmp, i, p, code_map1, level_map1, zero1, asym_map1, scale1, q, err
                        torch.cuda.empty_cache()

                del x, xmax, xmin, code_no_2, best
                torch.cuda.empty_cache()

            else:
                assert False, "Quantization method not supported"

        else:
            assert False, "Quantization method not supported"

        zeta = torch.ones((1, d_in), device=dev)
        if self.group_size > 0:
            set_zero = torch.ones([shape[0] * (shape[-1] // self.group_size), 1], device=dev)
        else:
            set_zero = torch.ones([shape[0], 1], device=dev)       
        
        self.scale = nn.Parameter(scale, requires_grad=False)
        self.zero = nn.Parameter(zero, requires_grad=False)
        self.code_map = nn.Parameter(code_map, requires_grad=False)
        self.level_map = nn.Parameter(level_map, requires_grad=False)
        self.asym_map = nn.Parameter(asym_map, requires_grad=False)
        self.set_zero = nn.Parameter(set_zero, requires_grad=False)
        self.zeta = nn.Parameter(zeta, requires_grad=False)

    def quantize(self, x):
        if self.ready():
            return quantize(x, scale=self.scale, zero=self.zero, method=self.method, group_size=self.group_size, code_map=self.code_map, level_map=self.level_map, asym_map=self.asym_map, set_zero=self.set_zero, hardware_approx=self.hardware_approx, maxq=self.maxq, sym=self.sym, search_scale=self.scale_search)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)
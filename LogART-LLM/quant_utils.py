import torch
from torch import nn, Tensor
import math

from torch._inductor.ir import NoneAsConstantBuffer

@torch.no_grad()
def uniform_quantize(x, scale, zero_point, nbits):    
    x_int = torch.clamp(torch.round(x / scale) + zero_point, 0, 2**nbits - 1)
    return (x_int - zero_point) * scale

@torch.no_grad()
def log_quantize(x, scale, zero, method, group_size, code_map, level_map, asym_map, set_zero, hardware_approx, nbits):
    original_shape = x.shape
    if group_size > 0:
        assert x.shape[-1] % group_size == 0
        if len(x.shape) == 2:
            x = x.reshape(-1, group_size)
        elif len(x.shape) == 3:
            x = x.reshape(original_shape[0], -1, group_size)

    if 'sqrt2' in method:
        real_code_map = torch.where(code_map, 2, math.sqrt(2))
        x_log = torch.div(torch.log(torch.abs(x / scale) + 1e-32), torch.log(real_code_map))

        real_zero = torch.where(code_map, zero[0].expand(x.shape), zero[1].expand(x.shape))
        real_level_map = torch.where(code_map, level_map[0].expand(x.shape), level_map[1].expand(x.shape))
        x_int = torch.round(x_log)
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
            sqrt2_flag = (code_map == False) & ((x_clamp + real_level_map 
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
        x_log = torch.div(torch.log(torch.abs(x / scale) + 1e-32), torch.log(real_code_map))

        real_zero = torch.where(code_map, zero[0].expand(x.shape), zero[1].expand(x.shape))
        real_level_map = torch.where(code_map, level_map[0].expand(x.shape), level_map[1].expand(x.shape))
        x_int = torch.round(x_log)
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

def quantize_zfold(weight, scale, zero, zeta, code_map, level_map, asym_map, set_zero, hardware_approx, method, group_size, nbits):
    if 'log' in method:
        if group_size > 0:
            if len(weight.shape) == 2:
                mul = scale.reshape(-1, weight.shape[0], 1) * zeta.reshape(-1, 1, group_size)
                mul = mul.view(-1, group_size)
            elif len(weight.shape) == 3:
                mul = scale.reshape(1, -1, weight.shape[0], weight.shape[1], 1) * zeta.reshape(1, -1, 1, 1, group_size)
                mul = mul.view(1, -1, group_size)
        else:
            mul = scale * zeta

        return log_quantize(weight, mul, zero, method, group_size, code_map, level_map, asym_map, set_zero, hardware_approx, nbits)
    else:
        assert False, "Quantization method not supported"

@torch.no_grad()
def compute_loss_perturb(weight, scale, zero, zeta, code_map, level_map, asym_map, set_zero, hardware_approx, method, group_size, nbits, H):
    delta_w = quantize_zfold(weight, scale, zero, zeta, code_map, level_map, asym_map, set_zero, hardware_approx, method, group_size, nbits) - weight
    if len(H.shape) == 3:
        num_heads = H.shape[0]
        delta_w = delta_w.view(num_heads, -1, delta_w.shape[-1])
    
    return ((delta_w @ H) * delta_w).sum()

def damping(H, percdamp=.01):  
    # Calculate the mean of diagonals across all heads  
    mean_diags = torch.mean(torch.diagonal(H, dim1=-2, dim2=-1), dim=-1)  

    # Add the damping values back into the original tensor along the diagonals  
    H.diagonal(dim1=-2, dim2=-1).add_(mean_diags.view(-1, *[1]*(len(H.shape)-2)), alpha=percdamp)  

    return H

def filter_dead_neuron(W, H, replace=1/2048, percdamp=.01, apply_damping=True):
    if len(H.shape) == 2:  
        H = H.unsqueeze(0)  
    num_heads, in_features = H.shape[0], H.shape[-1]  
    W = W.view(num_heads, -1, in_features)  

    # Extract the diagonals of H and find indices where they are equal to 0  
    diagonals = torch.diagonal(H, dim1=-2, dim2=-1)  
    idx_dead = (diagonals == 0)  

    # Set the corresponding columns of W to 0 and replace the dead neurons in H with the given value  
    mask = ~idx_dead.unsqueeze(-2)
    W *= mask  
    H.diagonal(dim1=-2, dim2=-1)[idx_dead] = replace  

    if apply_damping:
        H = damping(H, percdamp)

    W = W.view(-1, in_features)  
    H = H.squeeze()

    return W, H

@torch.jit.script
def _linear_grid_search_impl_H(
    w_flat: Tensor,
    zeta: Tensor,
    w_min: Tensor,
    w_max: Tensor,
    n_bits: int,
    symmetric: bool,
    H: Tensor
):
    # torchscriptable code
    best_score = torch.full_like(w_min, 1e10)
    best_scale = torch.ones_like(best_score)
    if symmetric:
        best_zero_point = torch.full_like(best_score, 2**(n_bits-1))
    else:
        best_zero_point = torch.zeros_like(best_score)

    for clip_ratio in torch.arange(1.0, 0.0, -0.01):
        new_max, new_min = w_max * clip_ratio, w_min * clip_ratio
        new_scale = (new_max - new_min) / (2**n_bits - 1)

        if symmetric:
            w_hat = new_scale * (torch.clamp((w_flat / new_scale + best_zero_point).round(), 0, 2**n_bits - 1) - best_zero_point)
            delta_w = (w_hat - w_flat) * zeta
            score = torch.sum((delta_w @ H) * delta_w, dim=-1, keepdim=True)

            best_scale = torch.where(score < best_score, new_scale, best_scale)
            best_score = torch.minimum(score, best_score)

        else:
            for round in ("floor", "ceil"):
                new_zeropoint = torch.floor(-new_min / new_scale) if round == "floor" else torch.ceil(-new_min / new_scale)
                # new_zeropoint = torch.round(-new_min / new_scale)
                w_hat = new_scale * (torch.clamp((w_flat / new_scale + new_zeropoint).round(), 0, 2**n_bits - 1) - new_zeropoint)
                delta_w = (w_hat - w_flat) * zeta
                score = torch.sum((delta_w @ H) * delta_w, dim=-1, keepdim=True)

                best_scale = torch.where(score < best_score, new_scale, best_scale)
                best_zero_point = torch.where(score < best_score, new_zeropoint, best_zero_point)
                best_score = torch.minimum(score, best_score)
    
    if len(w_flat.shape) == 2:
        best_scale = best_scale.expand((-1, w_flat.shape[-1]))
        best_zero_point = best_zero_point.expand((-1, w_flat.shape[-1]))
    if len(w_flat.shape) == 3:
        best_scale = best_scale.expand((-1, -1, w_flat.shape[-1]))
        best_zero_point = best_zero_point.expand((-1, -1, w_flat.shape[-1]))

    return best_scale, best_zero_point

@torch.jit.script
def _log_grid_search_impl_H( 
    w_flat: Tensor,
    zeta: Tensor,
    w_min: Tensor,
    w_max: Tensor,
    n_bits: int,
    symmetric: bool,
    scale_search: bool,
    method: str,
    group_size: int,
    code_map: Tensor,
    level_map: Tensor,
    asym_map: Tensor,
    set_zero: Tensor,
    hardware_approx: bool,
    H: Tensor
):
    # torchscriptable code
    best_score = torch.full_like(w_min, 1e10)
    if len(w_flat.shape) == 2:
        best_scale = torch.zeros_like(w_min)
        best_zero_point = torch.zeros([2, w_flat.shape[0], 1], device=w_flat.device)
        best_code_map = torch.zeros_like(w_flat, dtype=torch.bool)
        best_level_map = torch.zeros([2, w_flat.shape[0], 1], device=w_flat.device)
        best_asym_map = torch.zeros_like(w_min)
        best_set_zero = torch.zeros_like(w_min)
    else:
        best_scale = torch.zeros_like(w_min)
        best_zero_point = torch.zeros([2, w_flat.shape[0], w_flat.shape[1], 1], device=w_flat.device)
        best_code_map = torch.zeros_like(w_flat, dtype=torch.bool)
        best_level_map = torch.zeros([2, w_flat.shape[0], w_flat.shape[1], 1], device=w_flat.device)
        best_asym_map = torch.zeros_like(w_min)
        best_set_zero = torch.zeros_like(w_min)

    if method == 'log_2': 
        max_val = torch.floor(torch.log2(w_max + 1e-32) + 0.5)
        neg_max_val = torch.floor(torch.log2(-w_min + 1e-32) + 0.5)

        if symmetric:
            asym_map = torch.zeros_like(w_max)
        else:
            asym_map = torch.clamp((torch.abs(max_val - neg_max_val)), max=math.pow(2, n_bits))

        max_val = torch.maximum(max_val, neg_max_val)
        min_val = max_val - math.pow(2, (n_bits - 1)) + 1

        c_map = torch.ones_like(w_flat, dtype=torch.bool)
        zero_point = min_val
        zero_point = torch.stack([zero_point, zero_point], dim=0)
        l_map = torch.ones_like(w_max) * math.pow(2, (n_bits - 1))     
        l_map = torch.stack([l_map, l_map], dim=0)

        best_zero_point = zero_point
        best_code_map = c_map
        best_level_map = l_map
        best_asym_map = asym_map

        search_th = 0.0 if scale_search else 0.999
        
        for clip_ratio in torch.arange(1.0, search_th, -0.01):
            # Get Temporary Scale and Zero
            scale1 = torch.ones_like(w_min) * clip_ratio
            zero1 = zero_point
            code_map1 = c_map
            level_map1 = l_map
            asym_map1 = asym_map

            for round in ([-1, 1]):
                set_zero1 = torch.ones_like(w_min) * round

                # Quantize
                real_code_map = torch.where(code_map1, 2, math.sqrt(2))
                x_log = torch.div(torch.log(torch.abs(w_flat / scale1) + 1e-32), torch.log(real_code_map))

                real_zero = torch.where(code_map1, zero1[0].expand(w_flat.shape), zero1[1].expand(w_flat.shape))
                real_level_map = torch.where(code_map1, level_map1[0].expand(w_flat.shape), level_map1[1].expand(w_flat.shape))
                x_int = torch.round(x_log)
                # Clamp Low
                x_clamp_1 = torch.clamp(x_int - real_zero, (0 - (1 + asym_map1.expand(w_flat.shape) / 2) * code_map1), math.inf * torch.ones_like(w_flat))
                # Clamp High
                x_clamp = torch.clamp(x_clamp_1 - real_level_map, -math.inf, -1)

                x_quant = torch.pow(real_code_map, x_clamp + real_level_map + real_zero)
                
                if hardware_approx:
                    approx_factor = 1.5 / math.sqrt(2)
                    # approx_factor = (1 + 1/2 - 2**(-4) ) / math.sqrt(2)
                    # approx_factor = (1 + 1/2 - 2**(-4) - 2**(-6)) / math.sqrt(2)
                    sqrt2_flag = (code_map1 == False) & ((x_clamp + real_level_map 
                                                        + real_zero) % 2 != 0)
                    x_quant = torch.where(sqrt2_flag, x_quant * approx_factor , x_quant)
                # else:
                #     raise ValueError("Only Hardware approximation is supported")   

                # Set the minimum value to 0
                s = torch.sign(w_flat.detach())
                zero_flag = (x_clamp_1 <= -1 - asym_map1.expand(w_flat.shape) / 2) & code_map1
                # hazard_flag = (x_clamp_1 == 0 - asym_map1.expand(w_flat.shape) // 2) & code_map1 & (s == set_zero1) & (asym_map1.expand(w_flat.shape) % 2 == 0)
                x_quant[zero_flag] = 0
                # x_quant[hazard_flag] = 0

                q = x_quant * s * scale1.expand(w_flat.shape)

                # 初始化 new_delta_w
                if group_size > 0:
                    score = torch.full_like(w_min, 1e10)
                    increase = H.shape[-1] // group_size
                    delta_w = (q.reshape(1, -1, H.shape[-1]) - w_flat.reshape(1, -1, H.shape[-1])) * zeta

                    if len(w_flat.shape) == 2:                        
                        # new_delta_w = torch.zeros(delta_w.shape[0]*increase, delta_w.shape[1])
                        delta_arrange = delta_w.reshape(-1, group_size)
                        
                        for i in range(increase):
                            small_w = delta_arrange[torch.arange(i, delta_w.shape[0]*increase, increase), :]
                            small_H = H[i*group_size:(i+1)*group_size, i*group_size:(i+1)*group_size]
                            score[torch.arange(i, delta_w.shape[0]*increase, increase), 1] = torch.sum((small_w @ small_H) * small_w, dim=-1, keepdim=True)
                            # new_delta_w[torch.arange(i, delta_w.shape[0]*increase, increase), i*group_size:(i+1)*group_size] += delta_arrange[torch.arange(i, delta_w.shape[0]*increase, increase), :]
                        
                    elif len(w_flat.shape) == 3:
                        # new_delta_w = torch.zeros([delta_w.shape[0], delta_w.shape[1]*increase, delta_w.shape[2]], device=delta_w.device)
                        delta_arrange = delta_w.reshape(delta_w.shape[0], -1, group_size)

                        for i in range(increase):
                            small_w = delta_arrange[:, torch.arange(i, delta_w.shape[1]*increase, increase), :]
                            small_H = H[:, i*group_size:(i+1)*group_size, i*group_size:(i+1)*group_size]
                            score[:, torch.arange(i, delta_w.shape[1]*increase, increase), :] = torch.sum((small_w @ small_H) * small_w, dim=-1, keepdim=True)
                            
                            # new_delta_w[:, torch.arange(i, delta_w.shape[1]*increase, increase), i*group_size:(i+1)*group_size] += delta_arrange[:, torch.arange(i, delta_w.shape[1]*increase, increase), :]
                    else:
                        delta_arrange = delta_w
                        score = torch.sum((delta_arrange @ H) * delta_arrange, dim=-1, keepdim=True)
                else:
                    delta_w = (q - w_flat) * zeta
                    delta_arrange = delta_w
                    score = torch.sum((delta_arrange @ H) * delta_arrange, dim=-1, keepdim=True)
                
                

                best_scale = torch.where(score < best_score, scale1, best_scale)
                best_set_zero = torch.where(score < best_score, set_zero1, best_set_zero)
                best_score = torch.minimum(score, best_score)

    elif method == 'log_sqrt2':
        search_th = 0.0 if scale_search else 0.999

        for clip_ratio in torch.arange(1.0, search_th, -0.01):
            max_val = torch.floor(2 * (torch.log2(w_max + 1e-32)) + 0.5)
            neg_max_val = torch.floor(2 * (torch.log2(-w_min + 1e-32)) + 0.5)

            if symmetric:
                asym_map = torch.zeros_like(w_max)
            else:
                asym_map = torch.clamp((torch.abs(max_val - neg_max_val)), max=math.pow(2, n_bits))

            max_val = torch.maximum(max_val, neg_max_val)
            min_val = max_val - math.pow(2, (n_bits - 1)) + 1

            c_map = torch.zeros_like(w_flat, dtype=torch.bool) 
            zero_point = min_val
            zero_point = torch.stack([zero_point, zero_point], dim=0)
            l_map = torch.ones_like(w_max) * math.pow(2, (n_bits - 1))
            l_map = torch.stack([l_map, l_map], dim=0)      

            # Get Temporary Scale and Zero
            scale1 = torch.ones_like(w_min) * clip_ratio
            zero1 = zero_point
            code_map1 = c_map
            level_map1 = l_map
            asym_map1 = asym_map

            for round in ([-1, 1]):
                set_zero1 = torch.ones_like(w_min) * round

                # Quantize
                real_code_map = torch.where(code_map1, 2, math.sqrt(2))
                x_log = torch.div(torch.log(torch.abs(w_flat / scale1) + 1e-32), torch.log(real_code_map))

                real_zero = torch.where(code_map1, zero1[0].expand(w_flat.shape), zero1[1].expand(w_flat.shape))
                real_level_map = torch.where(code_map1, level_map1[0].expand(w_flat.shape), level_map1[1].expand(w_flat.shape))
                x_int = torch.round(x_log)
                # Clamp Low
                x_clamp_1 = torch.clamp(x_int - real_zero, -1 - asym_map1.expand(w_flat.shape) / 2, math.inf * torch.ones_like(w_flat))
                # Clamp High
                x_clamp = torch.clamp(x_clamp_1 - real_level_map, -math.inf, -1)

                x_quant = torch.pow(real_code_map, x_clamp + real_level_map + real_zero)
                
                if hardware_approx:
                    approx_factor = 1.5 / math.sqrt(2)
                    # approx_factor = (1 + 1/2 - 2**(-4) ) / math.sqrt(2)
                    # approx_factor = (1 + 1/2 - 2**(-4) - 2**(-6)) / math.sqrt(2)
                    sqrt2_flag = (code_map1 == False) & ((x_clamp + real_level_map 
                                                        + real_zero) % 2 != 0)
                    x_quant = torch.where(sqrt2_flag, x_quant * approx_factor , x_quant)
                # else:
                #     raise ValueError("Only Hardware approximation is supported")   

                # Set the minimum value to 0
                s = torch.sign(w_flat.detach())
                zero_flag = (x_clamp_1 <= -1 - asym_map1.expand(w_flat.shape) / 2)
                # hazard_flag = (x_clamp_1 == 0 - asym_map1.expand(w_flat.shape) // 2) & (s == set_zero1) & (asym_map1.expand(w_flat.shape) % 2 == 0)
                x_quant[zero_flag] = 0
                # x_quant[hazard_flag] = 0

                q = x_quant * s * scale1.expand(w_flat.shape)

                delta_w = (q - w_flat) * zeta
                score = torch.sum((delta_w @ H) * delta_w, dim=-1, keepdim=True)

                best_scale = torch.where(score < best_score, scale1, best_scale)
                best_zero_point = torch.where((score < best_score).unsqueeze(0), zero1, best_zero_point)
                best_code_map = torch.where(score < best_score, code_map1, best_code_map)
                best_level_map = torch.where((score < best_score).unsqueeze(0), level_map1, best_level_map)
                best_asym_map = torch.where(score < best_score, asym_map1, best_asym_map)
                best_set_zero = torch.where(score < best_score, set_zero1, best_set_zero)
                best_score = torch.minimum(score, best_score)
                
    elif method == 'log_dynamic':
        search_th = 0.0 if scale_search else 0.999
        for code_no_2 in range(int(2 ** (n_bits - 1)), -1, -2):
            for clip_ratio in torch.arange(1.0, search_th, -0.01):
                code_no_root2 = math.pow(2, (n_bits - 1)) - code_no_2

                pos_max_val_root2 = torch.floor(2 * (torch.log2(w_max + 1e-32)) + 0.5)
                neg_max_val_root2 = torch.floor(2 * (torch.log2(-w_min + 1e-32)) + 0.5)
                max_val_root2 = torch.maximum(pos_max_val_root2, neg_max_val_root2)

                min_val_root2 = max_val_root2 - code_no_root2 + 1
                max_val_2 = (max_val_root2 - code_no_root2) // 2
                min_val_2 = max_val_2 - code_no_2 + 1
                sl_index = (min_val_root2 / 2 + max_val_2) / 2
                sl = torch.pow(2, sl_index)

                input_abs = torch.abs(w_flat)
                c_map = input_abs <= sl.expand(w_flat.shape)

                zero_point_2 = min_val_2
                zero_point_root2 = min_val_root2
                zero_point = torch.stack([zero_point_2, zero_point_root2], dim=0)

                l_map_2 = torch.ones_like(w_max) * code_no_2
                l_map_root2 = torch.ones_like(w_max) * code_no_root2
                l_map = torch.stack([l_map_2, l_map_root2], dim=0)

                if symmetric:
                    asym_map = torch.zeros_like(w_max)
                else:
                    small_max = torch.minimum(torch.abs(w_min), w_max)
                    small_max_val = torch.where(small_max > sl, torch.floor(2 * (torch.log2(small_max + 1e-32)) + 0.5), torch.floor(torch.log2(small_max + 1e-32) + 0.5))
                    asym_map = torch.where(small_max > sl, (torch.abs(max_val_root2 - small_max_val)), (code_no_root2 + torch.abs(max_val_2 - small_max_val)))
                    asym_map = torch.clamp(asym_map, max=math.pow(2, n_bits))

                # Get Temporary Scale and Zero
                scale1 = torch.ones_like(w_min) * clip_ratio
                zero1 = zero_point
                code_map1 = c_map
                level_map1 = l_map
                asym_map1 = asym_map

                for round in ([-1, 1]):
                    set_zero1 = torch.ones_like(w_min) * round

                    # Quantize
                    real_code_map = torch.where(code_map1, 2, math.sqrt(2))
                    x_log = torch.div(torch.log(torch.abs(w_flat / scale1) + 1e-32), torch.log(real_code_map))

                    real_zero = torch.where(code_map1, zero1[0].expand(w_flat.shape), zero1[1].expand(w_flat.shape))
                    real_level_map = torch.where(code_map1, level_map1[0].expand(w_flat.shape), level_map1[1].expand(w_flat.shape))
                    x_int = torch.round(x_log)
                    # Clamp Low 
                    x_clamp_1 = torch.clamp(x_int - real_zero, (0 - (1 + asym_map1.expand(w_flat.shape) / 2) * code_map1), math.inf * torch.ones_like(w_flat))
                    # Clamp High
                    x_clamp = torch.clamp(x_clamp_1 - real_level_map, -math.inf, -1)

                    x_quant = torch.pow(real_code_map, x_clamp + real_level_map + real_zero)
                    
                    if hardware_approx:
                        approx_factor = 1.5 / math.sqrt(2)
                        # approx_factor = (1 + 1/2 - 2**(-4) ) / math.sqrt(2)
                        # approx_factor = (1 + 1/2 - 2**(-4) - 2**(-6)) / math.sqrt(2)
                        sqrt2_flag = (code_map1 == False) & ((x_clamp + real_level_map 
                                                            + real_zero) % 2 != 0)
                        x_quant = torch.where(sqrt2_flag, x_quant * approx_factor , x_quant)
                    # else:
                    #     raise ValueError("Only Hardware approximation is supported")   

                    # Set the minimum value to 0
                    s = torch.sign(w_flat.detach())
                    zero_flag = (x_clamp_1 <= -1 - asym_map1.expand(w_flat.shape) / 2) & code_map1
                    # hazard_flag = (x_clamp_1 == 0 - asym_map1.expand(w_flat.shape) // 2) & code_map1 & (s == set_zero1) & (asym_map1.expand(w_flat.shape) % 2 == 0)
                    x_quant[zero_flag] = 0
                    # x_quant[hazard_flag] = 0

                    q = x_quant * s * scale1.expand(w_flat.shape)

                    delta_w = (q - w_flat) * zeta
                    score = torch.sum((delta_w @ H) * delta_w, dim=-1, keepdim=True)

                    best_scale = torch.where(score < best_score, scale1, best_scale)
                    best_zero_point = torch.where((score < best_score).unsqueeze(0), zero1, best_zero_point)
                    best_code_map = torch.where(score < best_score, code_map1, best_code_map)
                    best_level_map = torch.where((score < best_score).unsqueeze(0), level_map1, best_level_map)
                    best_asym_map = torch.where(score < best_score, asym_map1, best_asym_map)
                    best_set_zero = torch.where(score < best_score, set_zero1, best_set_zero)
                    best_score = torch.minimum(score, best_score)

    else:
        assert False, "Quantization method not supported"

    return best_scale, best_zero_point, best_code_map, best_level_map, best_asym_map, best_set_zero

@torch.no_grad()
def find_quant_params(weight, zeta, n_bits, method, group_size=-1, code_map=None, level_map=None, asym_map=None, set_zero=None, hardware_approx=False, symmetric=True, scale_search=False, H=None):
    assert H is not None, "Hessian should be given."

    if group_size > 0:
        target_dim = [-1, *[1] * (len(weight.shape) - 1)]
        target_dim_zero = [2, -1, *[1] * (len(weight.shape) - 1)]
        target_dim_code = [-1, *[1*group_size] * (len(weight.shape) - 1)]
        w_flat = weight.flatten(1)
        in_features = w_flat.shape[-1]
    else:
        target_dim = [-1, *[1] * (len(weight.shape) - 1)]
        target_dim_zero = [2, -1, *[1] * (len(weight.shape) - 1)]
        target_dim_code = [-1, *[1*weight.shape[-1]] * (len(weight.shape) - 1)]
        w_flat = weight.flatten(1)
        in_features = w_flat.shape[-1]
    
    w_flat = w_flat / zeta
    if len(H.shape) == 2:
        H = H.unsqueeze(0)
    num_heads = H.shape[0]
    zeta = zeta.view(1, 1, in_features)
    if group_size > 0:
        w_flat = w_flat.view(num_heads, -1, group_size)
    else:
        w_flat = w_flat.view(num_heads, -1, in_features)

    tmp = torch.zeros((*w_flat.shape[:-1], 1), device=w_flat.device)
    w_max = torch.maximum(torch.max(w_flat, dim=-1, keepdim=True).values, tmp)
    w_min = torch.minimum(torch.min(w_flat, dim=-1, keepdim=True).values, tmp)

    if symmetric:
        w_max = torch.maximum(torch.abs(w_min), w_max)
        tmp = w_min < 0
        if torch.any(tmp):
            w_min[tmp] = -w_max[tmp]

    tmp = (w_min == 0) & (w_max == 0)

    # assert that xmin and xmax are not both 0
    assert not torch.any(tmp), "w_min and w_max are both 0"

    w_max[tmp] = 1
    w_min[tmp] = -1

    if 'log' in method:
        scale, zero_point, code_map, level_map, asym_map, set_zero = _log_grid_search_impl_H(w_flat, zeta, w_min, w_max, n_bits, symmetric, scale_search, method, group_size, code_map, level_map, asym_map, set_zero, hardware_approx, H)
    else:
        assert False, "Quantization method not supported"
    
    return scale.view(target_dim), zero_point.view(target_dim_zero), code_map.view(target_dim_code), level_map.view(target_dim_zero), asym_map.view(target_dim), set_zero.view(target_dim)


def refine_qparams_zfold(wrappers: dict, zeta_share_list: list, hyperparams: dict):
    import time 
    tick = time.time()
    
    for i, name in enumerate(zeta_share_list):
        if i > 0:
            break
        in_features = wrappers[name].layer.weight.data.shape[-1]
        n_bits = wrappers[name].quantizer.nbits
        sym = wrappers[name].quantizer.sym

    # pre-processing
    W_group, H_group = {}, {}
    for name in zeta_share_list:
        W, H = wrappers[name].layer.weight.data.clone(), wrappers[name].H.clone()
        W, H = filter_dead_neuron(W, H, replace=hyperparams['replace'], percdamp=hyperparams['percdamp'], apply_damping=True)
        W_group[name], H_group[name] = W, H
     
    # initialize qparams
    zeta = torch.ones([1, in_features], device=W.device)
    scale_group, zero_group = {}, {}
    for name in zeta_share_list:
        quantizer = wrappers[name].quantizer
        scale_group[name], zero_group[name] = quantizer.scale.view([-1, 1]), quantizer.zero.view([-1, 1])
        
    # compute initial loss perturbation incurred by quantization
    loss_perturb_initial = 0
    for name in zeta_share_list:
        loss_perturb_initial += compute_loss_perturb(W_group[name], scale_group[name], zero_group[name], zeta, n_bits, H_group[name])

    loss_perturb_before = loss_perturb_initial
    best_scale_group, best_zero_group, best_zeta = scale_group.copy(), zero_group.copy(), zeta

    # update scale/zero and zeta alternatively
    count_update = 0
    while count_update < 30:
        count_update += 1

        # update zeta
        zeta = find_zeta(W_group, scale_group, zero_group, zeta, n_bits).view(zeta.shape)
        zeta = torch.where(zeta==0., torch.ones(1).cuda(), zeta)

        # update scale and zero-point
        for name in zeta_share_list:
            scale_group[name], zero_group[name] = find_quant_params(W_group[name], zeta, n_bits, sym, H_group[name])
        
        # compute loss perturbation after update
        loss_perturb_after = 0
        for name in zeta_share_list:
            loss_perturb_after += compute_loss_perturb(W_group[name], scale_group[name], zero_group[name], zeta, n_bits, H_group[name])
        
        if loss_perturb_after > loss_perturb_before:
            break
        else:
            loss_perturb_before = loss_perturb_after
            best_scale_group, best_zero_group, best_zeta = scale_group.copy(), zero_group.copy(), zeta
    
    delta_loss_improvement = loss_perturb_initial - loss_perturb_before
    num_updates = count_update - 1

    for name in zeta_share_list:
        quantizer = wrappers[name].quantizer
        quantizer.scale.data = best_scale_group[name].view(quantizer.scale.shape)
        quantizer.zero.data = best_zero_group[name].view(quantizer.zero.shape)
        quantizer.zeta.data = best_zeta
    
    return delta_loss_improvement, num_updates, time.time() - tick


def find_zeta(W_group: dict, scale_group: dict, zero_group: dict, zeta, n_bits, eps=1e-10):
    W_stack = torch.cat(list(W_group.values()), dim=0)
    scale = torch.cat(list(scale_group.values()), dim=0)
    zero = torch.cat(list(zero_group.values()), dim=0)

    W_hat = uniform_quantize(W_stack / zeta, scale, zero, n_bits)
    p = torch.bmm(W_hat.T.unsqueeze(-2), W_stack.T.unsqueeze(-1))
    q = torch.bmm(W_hat.T.unsqueeze(-2), W_hat.T.unsqueeze(-1))
    q = eps * torch.ones_like(q) + q

    return p / q


def refine_qparams_with_hessian(wrappers, idx_block, model_type, use_zfold, hyperparams, llm_config):
    if use_zfold:
        from model_utils import set_zfold_layers
        zfold_list, zeta_share_lists = set_zfold_layers(model_type, llm_config)  # define layers where Z-Fold can be applied
    else:
        zfold_list, zeta_share_lists = [], {}

    print('+---------------------------+---------------------------+--------+-----------+')
    print('|           Layer           |   delta-loss-improvement  |  time  | num_iters |')
    print('+===========================+===========================+========+===========+')

    compute_zeta_list = []
    # compute shared zeta first
    for share_name, zeta_share_list in zeta_share_lists.items():
        delta_loss_improvement, num_updates, refine_time = refine_qparams_zfold(wrappers, zeta_share_list, hyperparams)
        print(f'|{idx_block}: {share_name : <24}| {delta_loss_improvement:.3f}\t| {refine_time:.2f}\t| {num_updates : <2}|')
        compute_zeta_list += zeta_share_list

    for name, wrapper in wrappers.items():
        if name in compute_zeta_list:
            continue
        else:
            delta_loss_improvement, num_updates, refine_time = wrapper.refine_quant_params(name in zfold_list, hyperparams)
        
        print(f'|{idx_block}: {name : <24}| {delta_loss_improvement:.3f}\t| {refine_time:.2f}\t| {num_updates : <2}|')
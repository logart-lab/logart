import time
import math
import torch
import torch.nn as nn

from quant_utils import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class Aespa:
    def __init__(self, layer):
        self.layer = layer        
        self.quantizer = None
        self.H = None
        self.cov_G = None

    def compute_cov_in_batch(self, _, inp, out):
        if self.H is None:
            self.H = 0
            self.n_data_in = 0

        inp = inp[0].data
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))  # shape = [BL, d]
        inp = inp.t()  # shape = [d, BL]

        n_current = inp.shape[-1]
        self.H *= self.n_data_in / (self.n_data_in + n_current)
        self.n_data_in += n_current
        inp = math.sqrt(2 / self.n_data_in) * inp.float()
        self.H += inp.matmul(inp.t())


    def compute_cov_out_batch(self, _, inps, outs, n_heads):
        if not hasattr(self, "cov_out"):
            self.cov_out = 0
            self.n_data_out = 0
        
        head_dim = outs.shape[-1] // n_heads
        outs = outs.view(outs.shape[0], outs.shape[1], n_heads, head_dim).transpose(1, 2).contiguous()  # [B, H, L, d_h]
        outs = outs.transpose(0, 1).view(n_heads, -1, head_dim).transpose(-1, -2).contiguous()  # [H, d_h, BL]

        n_current = outs.shape[-1]
        self.cov_out *= self.n_data_out / (self.n_data_out + n_current)
        self.n_data_out += n_current
        outs = math.sqrt(2 / self.n_data_out) * outs.float()
        self.cov_out += outs @ outs.transpose(-1, -2)


    def refine_quant_params(self, use_zfold: bool, hyperparams: dict):
        assert self.quantizer is not None, "Quantizer should be defined first."
        assert self.H is not None, "Hessian should be computed first."

        W = self.layer.weight.data.clone()
        if not self.quantizer.ready():
            self.quantizer.find_params(W)
        
        W, H = W.float(), self.H.clone()
        W, H = filter_dead_neuron(W, H, replace=hyperparams['replace'], percdamp=hyperparams['percdamp'], apply_damping=True)
        
        tick = time.time()
        if use_zfold:
            return refine_qparams_zfold({self.name: self}, [self.name], hyperparams)
        else:
            scale, zero, zeta = self.quantizer.scale, self.quantizer.zero, self.quantizer.zeta
            code_map, level_map, asym_map, set_zero, hardware_approx, method = self.quantizer.code_map, self.quantizer.level_map, self.quantizer.asym_map, self.quantizer.set_zero, self.quantizer.hardware_approx, self.quantizer.method
            scale_search = self.quantizer.scale_search
            n_bits = self.quantizer.nbits
            group_size = self.quantizer.group_size

            # compute initial loss perturbation incurred by quantization
            loss_perturb_before = compute_loss_perturb(W, scale, zero, zeta, code_map, level_map, asym_map, set_zero, hardware_approx, method, group_size, n_bits, H)

            # update scale and zero-point
            scale, zero, code_map, level_map, asym_map, set_zero = find_quant_params(W, zeta, n_bits, method, group_size, code_map, level_map, asym_map, set_zero, hardware_approx, self.quantizer.sym, scale_search, H)

            # compute loss perturbation after update
            loss_perturb_after = compute_loss_perturb(W, scale, zero, zeta, code_map, level_map, asym_map, set_zero, hardware_approx, method, group_size, n_bits, H)    

            delta_loss_improvement = loss_perturb_before - loss_perturb_after
            
            self.quantizer.scale.data = scale.view(self.quantizer.scale.shape)
            self.quantizer.zero.data = zero.view(self.quantizer.zero.shape)
            self.quantizer.code_map.data = code_map.view(self.quantizer.code_map.shape)
            self.quantizer.level_map.data = level_map.view(self.quantizer.level_map.shape)
            self.quantizer.set_zero.data = set_zero.view(self.quantizer.set_zero.shape)
            self.quantizer.asym_map.data = asym_map.view(self.quantizer.asym_map.shape)

            return delta_loss_improvement, 1, time.time() - tick

    def quant(self, opts:dict, hyperparams: dict):
        assert self.quantizer is not None, "Quantizer should be defined first."
        assert self.H is not None, "Hessian should be computed first."

        W = self.layer.weight.data.clone()
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)
        orig_w_shape, orig_w_dtype = W.shape, W.dtype
        W = W.float()

        # Quant. Params.
        scale, zero, zeta = self.quantizer.scale, self.quantizer.zero, self.quantizer.zeta
        code_map, level_map, asym_map, set_zero, hardware_approx, method = self.quantizer.code_map, self.quantizer.level_map, self.quantizer.asym_map, self.quantizer.set_zero, self.quantizer.hardware_approx, self.quantizer.method
        n_bits = self.quantizer.nbits
        group_size = self.quantizer.group_size

        H, cov_G = self.H.clone(), self.cov_G.clone() if self.cov_G is not None else None

        # pre-processing
        W, H = filter_dead_neuron(W, H, replace=hyperparams['replace'], percdamp=hyperparams['percdamp'], apply_damping=True)
        if len(H.shape) == 2:  # common Hessian for all heads
            H = H.unsqueeze(0)
        num_heads = H.shape[0] if cov_G is None else cov_G.shape[0]
        hidden_size = W.shape[-1]
        original_shape = W.shape
        head_dim = W.shape[0] // num_heads
        W = W.view(num_heads, head_dim, hidden_size)

        if group_size > 0:
            scale = scale.view(num_heads, head_dim * (original_shape[-1] // group_size), 1)
            zero = zero.view(2, num_heads, head_dim * (original_shape[-1] // group_size), 1)
            code_map = code_map.view(num_heads, head_dim * (original_shape[-1] // group_size), group_size)
            level_map = level_map.view(2, num_heads, head_dim * (original_shape[-1] // group_size), 1)
            asym_map = asym_map.view(num_heads, head_dim * (original_shape[-1] // group_size), 1)
            set_zero = set_zero.view(num_heads, head_dim* (original_shape[-1] // group_size), 1)
            zeta = zeta.view([1, (original_shape[-1] // group_size), -1])
        else:
            scale = scale.view(num_heads, head_dim, 1)
            zero = zero.view(2, num_heads, head_dim, 1)
            code_map = code_map.view(num_heads, head_dim, hidden_size)
            level_map = level_map.view(2, num_heads, head_dim, 1)
            asym_map = asym_map.view(num_heads, head_dim, 1)
            set_zero = set_zero.view(num_heads, head_dim, 1)
            zeta = zeta.view([1, 1, -1])

        # initialize weight-rounding policy
        if opts['optq_init']:
            W_update = self.optq(W, H, scale, zero, zeta, code_map, level_map, asym_map, set_zero, hardware_approx, method, group_size, n_bits, act_order=opts['act_order'])
            if not opts['learn_rounding']:
                print(f'|{self.i}: {self.name : <24}|GPU memory: {torch.cuda.max_memory_allocated("cuda") / 1024**3:.3f}\t|')
        else:
            W_update = W
        
        # weight-rounding optimization via learning
        if opts['learn_rounding']:
            Q = self.adaround(W, W_update, H, cov_G, scale, zero, zeta, code_map, level_map, asym_map, set_zero, hardware_approx, method, group_size, n_bits, opts)
        else:
            Q = quantize_zfold(W_update, scale, zero, zeta, code_map, level_map, asym_map, set_zero, hardware_approx, method, group_size, n_bits)

        del W, H, scale, zero, zeta, code_map, level_map, asym_map, set_zero
        torch.cuda.empty_cache()

        # assign quantized (fake-quant) weights
        self.layer.weight.data = Q.reshape(orig_w_shape).to(orig_w_dtype)


    def optq(self, W, H, scale, zero, zeta, code_map, level_map, asym_map, set_zero, hardware_approx, method, group_size, n_bits, act_order):
        W, H = W.clone(), H.clone()
        n_columns = W.shape[-1]

        if act_order:
            num_heads, hidden_size = W.shape[0], H.shape[-1]
            if H.shape[0] == 1:  # Common Hessian for all heads
                W = W.view(1, -1, hidden_size)
            perm_multi_head = torch.zeros((H.shape[0], hidden_size), dtype=torch.int64, device=H.device)
            invperm_multi_head = torch.zeros_like(perm_multi_head)
            zeta_multi_head = torch.zeros((H.shape[0], *zeta.shape[1:]), device=zeta.device)
            for idx_head in range(H.shape[0]):
                perm = torch.argsort(torch.diag(H[idx_head]), descending=True)
                invperm_multi_head[idx_head] = torch.argsort(perm)
                W[idx_head] = W[idx_head][:, perm]
                H[idx_head] = H[idx_head][perm][:, perm]
                zeta_multi_head[idx_head] = zeta[0][:, perm]
                perm_multi_head[idx_head] = perm

            W = W.view(num_heads, -1, hidden_size)
            zeta = zeta_multi_head

        # Cholesky Decomposition
        U = torch.zeros_like(H)
        for idx_head in range(H.shape[0]):
            U[idx_head] = torch.linalg.cholesky(
                torch.cholesky_inverse(torch.linalg.cholesky(H[idx_head])), upper=True
            )
        
        W_update = torch.zeros_like(W)
        blocksize = 128
        for i1 in range(0, n_columns, blocksize):
            i2 = min(i1 + blocksize, n_columns)
            count = i2 - i1

            W1 = W[..., i1:i2].clone()
            Err1 = torch.zeros_like(W1)
            U1 = U[..., i1:i2, i1:i2]

            for i in range(count):
                W_update[..., i1+i] = W1[..., i]
                
                q = quantize_zfold(W1[..., i].unsqueeze(-1), scale, zero, zeta[..., i1+i].unsqueeze(-1), code_map[..., i].unsqueeze(-1), level_map, asym_map, set_zero, hardware_approx, method, group_size, n_bits).squeeze(-1)
                err1 = (W1[..., i] - q) / U1[..., i, i].unsqueeze(-1)
                W1[..., i:] -= torch.matmul(err1.unsqueeze(-1), U1[..., i, i:].unsqueeze(-2))
                Err1[..., i] = err1
            W[..., i2:] -= torch.matmul(Err1, U[..., i1:i2, i2:])
        
        if act_order:
            if H.shape[0] == 1:
                W_update = W_update.view(1, -1, hidden_size)
            for idx_head in range(H.shape[0]):
                W_update[idx_head] = W_update[idx_head][:, invperm_multi_head[idx_head]]
            W_update = W_update.view(num_heads, -1, hidden_size)

        return W_update
    
    def uniform_quantize(self, W, rd, scale, zero, n_bits):
        q = torch.clamp(torch.floor(W / scale) + rd + zero, 0, 2**n_bits-1)
        q = scale * (q - zero)
        return q
    
    def log_quantize(self, W, rd, scale, zero, method, group_size, code_map, level_map, asym_map, set_zero, hardware_approx, n_bits):
        if 'sqrt2' in method:
            real_code_map = torch.where(code_map, 2, math.sqrt(2))
            x_log = torch.div(torch.log(torch.abs(W / scale.expand(W.shape)) + 1e-32), torch.log(real_code_map))

            real_zero = torch.where(code_map, zero[0].expand(W.shape), zero[1].expand(W.shape))
            real_level_map = torch.where(code_map, level_map[0].expand(W.shape), level_map[1].expand(W.shape))
            x_int = torch.floor(x_log) + rd
            # Clamp Low
            x_clamp_1 = torch.clamp(x_int - real_zero, -1 - asym_map.expand(W.shape) / 2, math.inf)
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
            
            del x_clamp, real_code_map, real_zero, real_level_map, x_log, x_int, x_clamp, x_quant
            torch.cuda.empty_cache()

            # Set the minimum value to 0
            s = torch.sign(W.detach())
            zero_flag = (x_clamp_1 <= -1 - asym_map.expand(W.shape) / 2) 
            # hazard_flag = (x_clamp_1 == 0 - asym_map.expand(W.shape) // 2) & (s == set_zero) & (asym_map.expand(W.shape) % 2 == 0)
            x_quant = torch.where(zero_flag, torch.zeros_like(x_quant), x_quant)
            # x_quant = torch.where(hazard_flag, torch.zeros_like(x_quant), x_quant)

            q = x_quant * s * scale.expand(W.shape)
        else:
            original_shape = W.shape
            if group_size > 0:
                assert W.shape[-1] % group_size == 0
                if len(original_shape) == 2:
                    W = W.reshape(-1, group_size)
                else:
                    W = W.reshape(original_shape[0], -1, group_size)
                
            real_code_map = torch.where(code_map, 2, math.sqrt(2))
            x_log = torch.div(torch.log(torch.abs(W / scale.expand(W.shape)) + 1e-32), torch.log(real_code_map))

            real_zero = torch.where(code_map, zero[0].expand(W.shape), zero[1].expand(W.shape))
            real_level_map = torch.where(code_map, level_map[0].expand(W.shape), level_map[1].expand(W.shape))
            x_int = torch.floor(x_log) + rd
            # Clamp Low
            x_clamp_1 = torch.clamp(x_int - real_zero, 0 - (1 + asym_map.expand(W.shape) / 2)  * code_map, math.inf * torch.ones_like(W))
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
            s = torch.sign(W.detach())
            zero_flag = (x_clamp_1 <= -1 - asym_map.expand(W.shape) / 2) & code_map
            # hazard_flag = (x_clamp_1 == 0 - asym_map.expand(W.shape) // 2) & code_map & (s == set_zero) & (asym_map.expand(W.shape) % 2 == 0)
            x_quant = torch.where(zero_flag, torch.zeros_like(x_quant), x_quant)
            # x_quant = torch.where(hazard_flag, torch.zeros_like(x_quant), x_quant)

            q = x_quant * s * scale.expand(W.shape)

            if group_size > 0:
                q = q.reshape(original_shape)

        return q
    
    def quantize_adaround(self, W, rd, scale, zero, zeta, code_map, level_map, asym_map, set_zero, hardware_approx, method, group_size, n_bits):
        if 'log' in method:
            return self.log_quantize(W, rd, scale, zero, method, group_size, code_map, level_map, asym_map, set_zero, hardware_approx, n_bits)
        else:
            assert False, "Quantization method not supported"

    def adaround(self, W_org, W_update, H, cov_G, scale, zero, zeta, code_map, level_map, asym_map, set_zero, hardware_approx, method, group_size, n_bits, opts: dict):
        lr, num_iters = opts['lr'], opts['num_iters']
        round_weight = opts['round_weight_qkv'] if self.name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] else opts['round_weight']

        # if group_size > 0:
        #     if len(W_update.shape) == 2:
        #         mul = scale.reshape(-1, W_update.shape[0], 1) * zeta.reshape(-1, 1, group_size)
        #         scale = mul.view(-1, group_size)
        #     elif len(W_update.shape) == 3:
        #         mul = scale.reshape(1, -1, W_update.shape[0], W_update.shape[1], 1) * zeta.reshape(1, -1, 1, 1, group_size)
        #         scale = mul.view(1, -1, group_size)
        # else:
        #     scale = scale * zeta

        print_period = int(num_iters * 0.2)
        with torch.enable_grad():
            sigm = RectifiedSigmoid(-0.1, 1.1)

            # Initialize the learnable rounding sb
            if 'log' in method:
                if group_size > 0:
                    original_shape = W_update.shape
                    if len(original_shape) == 2:
                        W_update = W_update.reshape(-1, group_size)
                    else:
                        W_update = W_update.reshape(original_shape[0], -1, group_size)

                real_code_map = torch.where(code_map, 2, math.sqrt(2))
                x_log = torch.div(torch.log(torch.abs(W_update / scale) + 1e-32), torch.log(real_code_map))
                x_floor = torch.floor(x_log)
                rest = x_log - x_floor
                sb = nn.Parameter(sigm.inverse(rest)) 

                if group_size > 0:
                    W_update = W_update.reshape(original_shape)
            else:
                assert False, "Quantization method not supported"

            optimizer = torch.optim.Adam([sb], lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters, eta_min=lr*0.3)

            round_loss_func = RoundLoss(max_count=num_iters, b_range=(20, 2), decay_start=0.0, warmup=0.2, p_norm=2.0)

            for i in range(num_iters):
                q = self.quantize_adaround(W_update, sigm(sb), scale, zero, zeta, code_map, level_map, asym_map, set_zero, hardware_approx, method, group_size, n_bits)

                e = q - W_org
                recon_loss = ((e @ H) * e).sum() if cov_G is None else (cov_G * (e @ H @ e.transpose(-1, -2))).sum()
                round_loss = round_loss_func(i, sigm(sb))
                total_loss = recon_loss + round_weight * round_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                del q, e, total_loss
                torch.cuda.empty_cache()

                if i == 0 or (i + 1) % print_period == 0:
                    if self.i is None:
                        print(f'|{self.name : <27}| {i+1: <2}\t| {float(recon_loss):.3f}\t| {float(round_loss):.3f}\t| {torch.cuda.max_memory_allocated("cuda") / 1024**3: .3f}\t|')
                    else:
                        print(f'|{self.i}: {self.name : <24}| {i+1: <2}\t| {float(recon_loss):.3f}\t| {float(round_loss):.3f}\t|{torch.cuda.max_memory_allocated("cuda") / 1024**3: .3f}\t|')
            print('+===========================+================+=================+=================+')

        if opts['test_with_hardware_approx']:
            hardware_approx = True
            
        Q = self.quantize_adaround(W_update, (sb >= 0).float(), scale, zero, zeta, code_map, level_map, asym_map, set_zero, hardware_approx, method, group_size, n_bits)
        
        return Q

    def free(self):
        self.H = None
        self.cov_G = None

        torch.cuda.empty_cache()


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay, start_b, end_b):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t-self.start_decay) / (self.t_max-self.start_decay)
            return self.end_b + (self.start_b-self.end_b)*max(0.0, 1 - rel_t)
        
class RoundLoss(nn.Module):
    def __init__(self, max_count, b_range, decay_start, warmup, p_norm):
        super(RoundLoss, self).__init__()
        self.loss_start = max_count * warmup
        # NOTE: cosine temp decay does not improve accuracy.
        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1-warmup)*decay_start, start_b=b_range[0], end_b=b_range[1])
        self.p_norm = p_norm
        self.b = 0

    def forward(self, iter_count, sb):
        """Compute regularization term to optimize the rounding policy"""
        if iter_count < self.loss_start:
            return 0
        else:
            self.b = self.temp_decay(iter_count)
            return (1 - (2*sb - 1).abs().pow(self.b)).sum()
        
class RectifiedSigmoid(nn.Module):
    """
    Implementation of Rectified Sigmoid Function
    Based on https://arxiv.org/pdf/1712.01312
    """

    def __init__(self, gamma, zeta):
        super(RectifiedSigmoid, self).__init__()
        self.gamma = gamma
        self.zeta = zeta

    def forward(self, x):
        return torch.clamp(torch.sigmoid(x)*(self.zeta-self.gamma) + self.gamma, 0, 1)

    def inverse(self, y):
        """return x that satisfies y = RectifiedSigmoid(x)"""
        return -torch.log((self.zeta-self.gamma)/(y-self.gamma) - 1)
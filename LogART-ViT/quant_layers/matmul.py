import numpy as np
import math
import torch
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
from itertools import product
from quantizers.uniform import *
from datetime import datetime


class MinMaxQuantMatMul(nn.Module):
    """Matrix Multiplication base class"""
    def __init__(self, A_bit=8, B_bit=8, mode="raw"):
        super().__init__()
        self.mode = mode
        self.A_quantizer = UniformQuantizer(n_bits = A_bit, symmetric = True, channel_wise = False)
        self.B_quantizer = UniformQuantizer(n_bits = B_bit, symmetric = True, channel_wise = False)
        self.raw_input = None
        self.raw_out = None
        self.tmp_input = None
        self.tmp_out = None
        self.calibrated = True
    
    def forward(self, A, B):
        if self.mode == 'raw':
            out = A @ B
        elif self.mode == 'debug_only_quant_weight':
            out = A @ B
        elif self.mode == "quant_forward":
            out = self.quant_forward(A, B)
        else:
            raise NotImplementedError
        return out
    
    def quant_input_A(self, x):
        return self.A_quantizer(x)
    
    def quant_input_B(self, x):
        return self.B_quantizer(x)
    
    def quant_forward(self, A, B):
        assert self.calibrated, f"Module should be calibrated before run quant_forward for {self}"
        return self.quant_input_A(A) @ self.quant_input_B(B)
    
    
class PTQSLQuantMatMul(MinMaxQuantMatMul):
    """
    - Q @ K:
        - A's shape: B,H,S,C
        - B's shape: B,H,C,S
    - scores @ V:
        - A's shape: B,H,S,S
        - B's shape: B,H,S,C
    """
    def __init__(self, A_bit=8, B_bit=8, mode="raw", metric="mse", search_round=1, scale_method='linear', 
                 hardware_approx=False, eq_n=100, head_channel_wise=True, token_channel_wise=False, num_heads=12):
        super().__init__(A_bit, B_bit, mode)
        self.A_quantizer = UniformQuantizer(n_bits = A_bit, symmetric = True, channel_wise = head_channel_wise)
        self.B_quantizer = UniformQuantizer(n_bits = B_bit, symmetric = True, channel_wise = head_channel_wise)
        self.metric = metric
        self.search_round = search_round
        self.eq_n = eq_n
        # the head dim is always dim-1
        self.head_channel_wise = head_channel_wise
        self.token_channel_wise = token_channel_wise
        self.num_heads = num_heads
        
        if not self.head_channel_wise:
            target_shape = [1, 1, 1, 1]
            self.A_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
            self.B_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
        else:
            target_shape = [1, self.num_heads, 1, 1]
            self.A_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
            self.B_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
    
    def _get_similarity(self, tensor_raw, tensor_sim, metric=None):
        if metric == "mae":
            similarity = -torch.abs(tensor_raw - tensor_sim)
        elif metric == "mse":
            similarity = -(tensor_raw - tensor_sim) ** 2
        else:
            raise NotImplementedError(f"metric {metric} not implemented!")
        return similarity
        
    
class PTQSLBatchingQuantMatMul(PTQSLQuantMatMul):
    def __init__(self, A_bit=8, B_bit=8, mode="raw", metric="mse", calib_batch_size=32, 
                 search_round=1, scale_method='linear', hardware_approx=False, eq_n=100, 
                 head_channel_wise=True, token_channel_wise=False, num_heads=12):
        super().__init__(A_bit, B_bit, mode, metric, search_round, scale_method, hardware_approx, 
                        eq_n, head_channel_wise, token_channel_wise, num_heads)
        self.calib_batch_size = calib_batch_size
        
    def _initialize_calib_parameters(self):
        """ 
        set parameters for feeding calibration data
        """
        self.calib_size = self.raw_input[0].shape[0]
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            memory = props.total_memory // 2
        else:
            raise EnvironmentError("CUDA is not available on this system")
        numel = (4 * self.raw_input[0][:self.calib_size].numel()+
                 4 * self.raw_input[1][:self.calib_size].numel()+
                 8 * self.raw_out[:self.calib_batch_size].numel()) # number of parameters on GPU
        self.parallel_eq_n = int((memory / 4) // numel)
        self.parallel_eq_n = math.ceil(self.eq_n * 1.0 / math.ceil(self.eq_n * 1.0 / self.parallel_eq_n))
        
        
class AsymmetricallyBatchingQuantMatMul(PTQSLBatchingQuantMatMul):
    def __init__(self, A_bit=8, B_bit=8, mode="raw", metric="mse", calib_batch_size=32, 
                 search_round=1, scale_method='linear', hardware_approx=False, eq_n=100, 
                 head_channel_wise=True, token_channel_wise=False, num_heads=12):
        super().__init__(A_bit, B_bit, mode, metric, calib_batch_size, search_round, scale_method, 
                         hardware_approx, eq_n, head_channel_wise, token_channel_wise, num_heads)
        del self.A_quantizer, self.B_quantizer
        self.A_quantizer = UniformQuantizer(n_bits = A_bit, symmetric = False, channel_wise = head_channel_wise)
        self.B_quantizer = UniformQuantizer(n_bits = B_bit, symmetric = False, channel_wise = head_channel_wise)
        if not self.head_channel_wise:
            target_shape = [1, 1, 1, 1]
            self.A_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
            self.B_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
            self.A_quantizer.zero_point = nn.Parameter(torch.zeros(*target_shape))
            self.B_quantizer.zero_point = nn.Parameter(torch.zeros(*target_shape))
        else:
            target_shape = [1, self.num_heads, 1, 1]
            self.A_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
            self.B_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
            self.A_quantizer.zero_point = nn.Parameter(torch.zeros(*target_shape))
            self.B_quantizer.zero_point = nn.Parameter(torch.zeros(*target_shape))
    
    def _search_best_A_scale(self, A_scale_candidates, A_zero_point_candidates):
        batch_similarities = [] # similarities, need to concatenate and calculate sum
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            A = self.raw_input[0][b_st:b_ed].cuda()
            B = self.raw_input[1][b_st:b_ed].cuda()
            B_sim = self.quant_input_B(B).unsqueeze(0) # shape: 1,b,*,dim2,dim3
            raw_out = self.raw_out[b_st:b_ed].unsqueeze(0).cuda()
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                # quantize A
                cur_A_scale = A_scale_candidates[p_st:p_ed]
                cur_A_zero_point = A_zero_point_candidates[p_st:p_ed]
                A_sim = A.squeeze(0)
                A_quant = ((A_sim / cur_A_scale).round_() + cur_A_zero_point).clamp(0, 2 * self.A_quantizer.n_levels - 1)
                A_sim = (A_quant - cur_A_zero_point).mul_(cur_A_scale) # shape: (parallel_eq_n,b,*,dim1,dim2)
                out_sim = A_sim @ B_sim # shape: parallel_eq_n,b,*,dim1,dim3
                similarity = self._get_similarity(raw_out, out_sim, self.metric) # shape: parallel_eq_n,b,*,dim1,dim3
                if self.head_channel_wise:
                    similarity = torch.mean(similarity, dim=list(range(3, len(similarity.shape)))) # shape: parallel_eq_n,b,heads
                else:
                    similarity = torch.mean(similarity, dim=list(range(2, len(similarity.shape)))) # shape: parallel_eq_n,b
                similarity = similarity.sum(dim=1, keepdim=True) # shape: (parallel_eq_n,1) or (parallel_eq_n,1,heads)
                similarities.append(similarity)
            # calculate best similarity for this block
            similarities = torch.cat(similarities, 0) # shape: (eq_n,1) or (eq_n,1,heads)
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=1).sum(dim=1, keepdim=False) #shape: eq_n or (eq_n,heads)
        best_index = torch.argmax(batch_similarities, dim=0, keepdim=False).view(1, 1, -1, 1, 1)
        tmp_A_scale = torch.gather(A_scale_candidates, dim=0, index=best_index)
        tmp_A_zero_point = torch.gather(A_zero_point_candidates, dim=0, index=best_index)
        self.A_quantizer.scale.data.copy_(tmp_A_scale.view(self.A_quantizer.scale.shape))
        self.A_quantizer.zero_point.copy_(tmp_A_zero_point.view(self.A_quantizer.zero_point.shape))
        return best_index
        
    def _search_best_B_scale(self, B_scale_candidates, B_zero_point_candidates):
        batch_similarities = [] # similarities, need to concatenate and calculate sum
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            A = self.raw_input[0][b_st:b_ed].cuda()
            B = self.raw_input[1][b_st:b_ed].cuda()
            A_sim = self.quant_input_A(A).unsqueeze(0) # shape: 1,b,*,dim1,dim2
            raw_out = self.raw_out[b_st:b_ed].unsqueeze(0).cuda()
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                # quantize B
                cur_B_scale = B_scale_candidates[p_st:p_ed]
                cur_B_zero_point = B_zero_point_candidates[p_st:p_ed]
                B_sim = B.squeeze(0)
                B_quant = ((B_sim / cur_B_scale).round_() + cur_B_zero_point).clamp(0, 2 * self.B_quantizer.n_levels - 1)
                B_sim = (B_quant - cur_B_zero_point).mul_(cur_B_scale) # shape: (parallel_eq_n,b,*,dim2,dim3)
                out_sim = A_sim @ B_sim # shape: parallel_eq_n,b,*,dim1,dim3
                similarity = self._get_similarity(raw_out, out_sim, self.metric) # shape: parallel_eq_n,b,*,dim1,dim3
                if self.head_channel_wise:
                    similarity = torch.mean(similarity, dim=list(range(3, len(similarity.shape)))) # shape: parallel_eq_n,b,heads
                else:
                    similarity = torch.mean(similarity, dim=list(range(2, len(similarity.shape)))) # shape: parallel_eq_n,b
                similarity = similarity.sum(dim=1, keepdim=True) # shape: (parallel_eq_n,1) or (parallel_eq_n,1,heads)
                similarities.append(similarity)
            # calculate best similarity for this block
            similarities = torch.cat(similarities, 0) # shape: (eq_n,1) or (eq_n,1,heads)
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=1).sum(dim=1, keepdim=False) #shape: eq_n or (eq_n,heads)
        best_index = torch.argmax(batch_similarities, dim=0, keepdim=False).view(1, 1, -1, 1, 1)
        tmp_B_scale = torch.gather(B_scale_candidates, dim=0, index=best_index)
        tmp_B_zero_point = torch.gather(B_zero_point_candidates, dim=0, index=best_index)
        self.B_quantizer.scale.data.copy_(tmp_B_scale.view(self.B_quantizer.scale.shape))
        self.B_quantizer.zero_point.copy_(tmp_B_zero_point.view(self.B_quantizer.zero_point.shape))
        return best_index
    
    def calculate_percentile_candidates(self, x, l=0.99, r=0.99999):
        percentiles_uppers, percentiles_lowers = [], []
        pct = torch.tensor([l, r])
        tensor_too_large = True
        mini_batch_size = 1
        if self.head_channel_wise:
            x_ = x.transpose(0, 1).contiguous() # shape: heads,b,*,dim1,dim2
            x_ = x_.view(x_.shape[0], mini_batch_size, -1) 
        else:
            x_ = x.view(1, mini_batch_size, -1)
        while tensor_too_large:
            try:
                uppers_candidates = torch.quantile(x_, pct.to(x_.device), dim=-1).mean(dim=-1, keepdim=False) # shape: 2,(heads or 1)
                lowers_candidates = torch.quantile(x_, (1 - pct).to(x_.device), dim=-1).mean(dim=-1, keepdim=False) # shape: 2,(heads or 1)
                tensor_too_large = False
            except:
                mini_batch_size *= 2
                x_ = x_.view(x_.shape[0], mini_batch_size, -1) if self.head_channel_wise else x_.view(1, mini_batch_size, -1)
        u_splits = torch.linspace(0, 1, steps=self.eq_n+1).cuda()[:, None, None, None, None] * (uppers_candidates[1] - uppers_candidates[0]).view(1, 1, -1, 1, 1)
        d_splits = torch.linspace(0, 1, steps=self.eq_n+1).cuda()[:, None, None, None, None] * (lowers_candidates[0] - lowers_candidates[1]).view(1, 1, -1, 1, 1)
        upper_candidates = uppers_candidates[0].view(1, 1, -1, 1, 1) + u_splits
        lower_candidates = lowers_candidates[1].view(1, 1, -1, 1, 1) + d_splits
        return upper_candidates, lower_candidates
        
    def hyperparameter_searching(self):
        self._initialize_calib_parameters()
        A_uppers_candidates, A_lowers_candidates = self.calculate_percentile_candidates(self.raw_input[0].cuda(), l=0.99, r=0.99999)
        B_uppers_candidates, B_lowers_candidates = self.calculate_percentile_candidates(self.raw_input[1].cuda(), l=0.99, r=0.99999)
        A_scale_candidates = ((A_uppers_candidates - A_lowers_candidates) / (2 * self.A_quantizer.n_levels - 1)).contiguous().cuda()
        A_zero_point_candidates = -(A_lowers_candidates / A_scale_candidates).round().contiguous().cuda()
        B_scale_candidates = ((B_uppers_candidates - B_lowers_candidates) / (2 * self.B_quantizer.n_levels - 1)).contiguous().cuda()
        B_zero_point_candidates = -(B_lowers_candidates / B_scale_candidates).round().contiguous().cuda()
        self.A_quantizer.scale.data.copy_(A_scale_candidates[-2])
        self.A_quantizer.zero_point.data.copy_(A_zero_point_candidates[-2])
        self.B_quantizer.scale.data.copy_(B_scale_candidates[-2])
        self.B_quantizer.zero_point.data.copy_(B_zero_point_candidates[-2])
        self.A_quantizer.inited = True
        self.B_quantizer.inited = True
        
        A_best_index = self._search_best_A_scale(A_scale_candidates, A_zero_point_candidates)
        B_best_index = self._search_best_B_scale(B_scale_candidates, B_zero_point_candidates)
        for e in range(self.search_round):
            if self.A_quantizer.n_bits < 32:
                for ee in range(2):
                    if ee % 2 == 0:
                        A_uppers_candidates_ = torch.gather(A_uppers_candidates, dim=0, index=A_best_index)
                        A_lowers_candidates_ = A_lowers_candidates
                    else:
                        A_uppers_candidates_ = A_uppers_candidates
                        A_lowers_candidates_ = torch.gather(A_lowers_candidates, dim=0, index=A_best_index)
                    A_scale_candidates = ((A_uppers_candidates_ - A_lowers_candidates_) / (2 * self.A_quantizer.n_levels - 1)).contiguous().cuda()
                    A_zero_point_candidates = -(A_lowers_candidates_ / A_scale_candidates).round().contiguous().cuda()
                    A_best_index = self._search_best_A_scale(A_scale_candidates, A_zero_point_candidates)
            if self.B_quantizer.n_bits < 32:
                for ee in range(2):
                    if ee % 2 == 0:
                        B_uppers_candidates_ = torch.gather(B_uppers_candidates, dim=0, index=B_best_index)
                        B_lowers_candidates_ = B_lowers_candidates
                    else:
                        B_uppers_candidates_ = B_uppers_candidates
                        B_lowers_candidates_ = torch.gather(B_lowers_candidates, dim=0, index=B_best_index)
                    B_scale_candidates = ((B_uppers_candidates_ - B_lowers_candidates_) / (2 * self.B_quantizer.n_levels - 1)).contiguous().cuda()
                    B_zero_point_candidates = -(B_lowers_candidates_ / B_scale_candidates).round().contiguous().cuda()
                    B_best_index = self._search_best_B_scale(B_scale_candidates, B_zero_point_candidates)
        
        if self.token_channel_wise:
            BA, HA, NA, MA = self.raw_input[0].shape
            BB, HB, NB, MB = self.raw_input[1].shape
            assert BA == BB and HA == HB and MA == NB
            A_token_wise_scale = self.A_quantizer.scale.expand(-1, -1, NA, -1)
            B_token_wise_scale = self.B_quantizer.scale.expand(-1, -1, -1, MB)
            del self.A_quantizer.scale, self.B_quantizer.scale
            self.A_quantizer.scale = nn.Parameter(A_token_wise_scale.clone())
            self.B_quantizer.scale = nn.Parameter(B_token_wise_scale.clone())
        
        self.calibrated = True
        del self.raw_input, self.raw_out
        return None

import torch
from quantizers.uniform import *
from timm.layers.patch_embed import PatchEmbed


def search_qparams(block, wrappers, last_block, inps, search_method, batch_size=32):
    zfold_list = ['attn.proj', 'mlp.fc1', 'mlp.fc2']

    channel_wise = True

    if isinstance(block, PatchEmbed):
        for name in wrappers:
            best_scale, zero_point, level_map, code_map, asym_map = layer_wise(wrappers, name, channel_wise, search_method, batch_size)
            wrappers[name].quantizer.zero_point = zero_point
            wrappers[name].quantizer.code_map = code_map
            wrappers[name].quantizer.level_map = level_map
            wrappers[name].quantizer.scale = best_scale
            wrappers[name].quantizer.asym_map = asym_map

    elif last_block:
        for name in wrappers:
            best_scale, zero_point, level_map, code_map, asym_map = layer_wise(wrappers, name, channel_wise, search_method, batch_size)
            wrappers[name].quantizer.zero_point = zero_point
            wrappers[name].quantizer.code_map = code_map
            wrappers[name].quantizer.level_map = level_map
            wrappers[name].quantizer.scale = best_scale
            wrappers[name].quantizer.asym_map = asym_map

    else:
        # # W（QKV） Separate
        # for i in range(3):
        #     # qkv weight shape: [3*oc, ic]
        #     channel_num = int(wrappers['attn.qkv'].layer.weight.shape[0] / 3)

        #     best_scale, zero_point, level_map, code_map, asym_map = block_wise(block, wrappers, inps, channel_wise, i, search_method, batch_size)
        #     wrappers['attn.qkv'].quantizer.zero_point[(i * channel_num):((1 + i) * channel_num), :] = zero_point
        #     wrappers['attn.qkv'].quantizer.code_map[(i * channel_num):((1 + i) * channel_num), :] = code_map
        #     wrappers['attn.qkv'].quantizer.level_map[(i * channel_num):((1 + i) * channel_num), :] = level_map
        #     wrappers['attn.qkv'].quantizer.scale[(i * channel_num):((1 + i) * channel_num), :] = best_scale
        #     wrappers['attn.qkv'].quantizer.asym_map[(i * channel_num):((1 + i) * channel_num), :] = asym_map

        best_scale, zero_point, level_map, code_map, asym_map = layer_wise(wrappers, 'attn.qkv', channel_wise, search_method, batch_size)
        wrappers['attn.qkv'].quantizer.zero_point = zero_point
        wrappers['attn.qkv'].quantizer.code_map = code_map
        wrappers['attn.qkv'].quantizer.level_map = level_map
        wrappers['attn.qkv'].quantizer.scale = best_scale
        wrappers['attn.qkv'].quantizer.asym_map = asym_map

        # best_scale, zero_point, level_map, code_map, asym_map = block_wise(block, wrappers, inps, channel_wise, search_method, batch_size)
        # wrappers['attn.qkv'].quantizer.zero_point = zero_point
        # wrappers['attn.qkv'].quantizer.code_map = code_map
        # wrappers['attn.qkv'].quantizer.level_map = level_map
        # wrappers['attn.qkv'].quantizer.scale = best_scale
        # wrappers['attn.qkv'].quantizer.asym_map = asym_map

        for name in zfold_list:
            best_scale, zero_point, level_map, code_map, asym_map = layer_wise(wrappers, name, channel_wise, search_method, batch_size)
            wrappers[name].quantizer.zero_point = zero_point
            wrappers[name].quantizer.code_map = code_map
            wrappers[name].quantizer.level_map = level_map
            wrappers[name].quantizer.scale = best_scale
            wrappers[name].quantizer.asym_map = asym_map


def block_wise(transform_block, wrappers, inps, channel_wise=False, qkv_i=0, search_method='tensor_wise', batch_size=32):
    if channel_wise:
        if search_method == 'block_wise':
            block_inps = inps
            block_outs = wrappers['attn.proj'].inps

        ori_weight = wrappers['attn.qkv'].layer.weight.data.clone().detach()
        num_channels = int(ori_weight.shape[0] / 3)
        ori_weight_q_k_v = wrappers['attn.qkv'].layer.weight.data[
            (qkv_i * num_channels):((1 + qkv_i) * num_channels), :].clone().detach()

        layer_quantizer = wrappers['attn.qkv'].quantizer
        scale_method = layer_quantizer.scale_method

        # get the channel-wise max of the weight
        tmp = torch.zeros(ori_weight_q_k_v.shape[0], device=ori_weight.device)
        xmin = torch.minimum(ori_weight_q_k_v.min(1)[0], tmp)
        xmax = torch.maximum(ori_weight_q_k_v.max(1)[0], tmp)
        if layer_quantizer.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            xmin = -xmax
        
        tmp = (xmin == 0) & (xmax == 0)
        # assert that xmin and xmax are not both 0
        assert not torch.any(tmp), "xmin and xmax are both 0"

        # Initialize Scale, Zero, Code Map, and Level Map
        dim_diff = ori_weight_q_k_v.dim() - (xmin.unsqueeze(1)).dim()
        target_shape = (xmin.unsqueeze(1)).shape + (1,) * dim_diff

        best = torch.full_like(xmin.unsqueeze(1), 1e10, device=ori_weight.device) # best shape: [num_channels, 1] 
        scale = torch.zeros_like((xmin.unsqueeze(1)).view(target_shape), device=ori_weight.device) # delta shape: [num_channels, 1, 1, 1]
        zero_point = torch.zeros_like(ori_weight_q_k_v, device=ori_weight.device) # zero_point shape: [num_channels, 1/in_channels, kernel_size/1, kernel_size/1]
        code_map = torch.zeros_like(ori_weight_q_k_v, device=ori_weight.device) # code_map shape: [num_channels, 1/in_channels, kernel_size/1, kernel_size/1]
        level_map = torch.zeros_like(ori_weight_q_k_v, device=ori_weight.device) # level_map shape: [num_channels, 1/in_channels, kernel_size/1, kernel_size/1]
        asym_map = torch.zeros_like((xmin.unsqueeze(1)).view(target_shape), device=ori_weight.device) # asym_map shape: [num_channels, 1, 1, 1]
        err = torch.zeros_like((xmin.unsqueeze(1)).view(target_shape), device=ori_weight.device) # err shape: [num_channels, 1]
        
        if scale_method == 'log_2':
            raise NotImplementedError("Only support log_dynamic for now!")
        elif scale_method == 'log_sqrt2':
            raise NotImplementedError("Only support log_dynamic for now!")
        elif scale_method == 'log_dynamic':
            search_threshold = 0.0 # search_threshold = 0.999 if disable scaling factor search
            for clip_ratio in torch.arange(1.0, search_threshold, -0.01):
                for code_no_2 in range(int(2 ** (layer_quantizer.n_bits - 1)), -1, -2):
                    code_map1, level_map1, zero1, asym_map1 = layer_quantizer.get_code_map(ori_weight_q_k_v, 
                                                                                        xmax.unsqueeze(1), xmin.unsqueeze(1), 
                                                                                        code_no_2, layer_quantizer.n_bits, 
                                                                                        layer_quantizer.sym)
                    
                    scale1 = torch.ones([ori_weight_q_k_v.shape[0], 1], device=ori_weight.device) * clip_ratio
                    scale1 = scale1.view(target_shape) # scale1 shape: [num_channels, 1, 1, 1]

                    q = log_quantize(ori_weight_q_k_v, scale=scale1, zero=zero1, method=scale_method, code_map=code_map1, level_map=level_map1, asym_map=asym_map1, hardware_approx=layer_quantizer.hardware_approx)
                    
                    if search_method == 'tensor_wise':
                        err = q - ori_weight_q_k_v
                        err.abs_()
                        err.pow_(2)
                        err = torch.sum(err.view(err.shape[0], -1), dim=-1, keepdim=True)
                    
                    elif search_method == 'block_wise':
                        temp_qkv = wrappers['attn.qkv'].layer.weight.data.clone().detach()
                        temp_qkv[(qkv_i * num_channels):((1 + qkv_i) * num_channels), :] = q
                        wrappers['attn.qkv'].layer.weight.data = temp_qkv
                        # qkv_module = dict(transform_block.named_modules())['attn.qkv']
                        # qkv_module.weight.data[(qkv_i * num_channels):((1 + qkv_i) * num_channels), :] = q

                        # Find the target module and register the hook
                        # proj_module = dict(transform_block.named_modules())['attn.proj']
                        proj_module = transform_block.attn.proj
                        # quantized_out = torch.zeros_like(block_outs)
                        err.zero_()
                        
                        # Perform the forward pass. The hook will capture the input to attn.proj
                        for i in range(int(block_inps.size(0) / batch_size)): 
                            def capture_hook(module, input, output):
                                # The input to the module is a tuple, we are interested in the first element
                                captured_input['in'] = input[0].detach()

                            captured_input = {}
                            handle = proj_module.register_forward_hook(capture_hook)

                            with torch.no_grad():
                                _ = transform_block(block_inps[i * batch_size:(i + 1) * batch_size])

                            # Remove the hook and get the captured tensor
                            handle.remove()

                            # shape of quantized_output: [batch_size, num_heads, seq_len, head_dim]????
                            quantized_out = captured_input['in']
                        
                            temp_err = quantized_out - block_outs[i * batch_size:(i + 1) * batch_size]

                            del quantized_out, captured_input
                            torch.cuda.empty_cache()

                            temp_err.abs_()
                            temp_err.pow_(2)
                            if len(temp_err.shape) == 4: # err shape: [batch_size, num_out_channels, out_w?, out_h?]
                                temp_err = torch.sum(temp_err, dim=[0, 2, 3]).unsqueeze(1) # err shape: [num_out_channels, 1]
                            elif len(temp_err.shape) == 3: # err shape: [batch_size, num_out_channels, ]
                                temp_err = torch.sum(temp_err, dim=[0, 1]).unsqueeze(1) # err shape: [num_out_channels, 1]
                            else:
                                assert False, 'Wrong shape of err'
                            
                            err += temp_err.detach()

                            del temp_err
                            torch.cuda.empty_cache()
                        
                        # restore the original weight
                        # qkv_module.weight.data = ori_weight
                        wrappers['attn.qkv'].layer.weight.data = ori_weight
                    
                    else:
                        raise NotImplementedError("Only support tensor_wise and block_wise for now!")
                    
                    tmp = (err < best)
                    tmp = tmp.view(target_shape)

                    scale = torch.where(tmp, scale1, scale)
                    zero_point = torch.where(tmp, zero1, zero_point)
                    code_map = torch.where(tmp, code_map1, code_map)
                    level_map = torch.where(tmp, level_map1, level_map)
                    asym_map = torch.where(tmp, asym_map1, asym_map)
                    best = torch.minimum(err, best)
                
                # wrappers['attn.qkv'].layer.weight.data = ori_weight
        
        # wrappers['attn.qkv'].layer.weight.data = ori_weight
        # transform_block.attn.qkv.weight.data = ori_weight

    else:
        raise NotImplementedError("Only support per-channel quantization for now!")

    return scale, zero_point, level_map, code_map, asym_map


def layer_wise(wrappers: dict, name: str, channel_wise: bool = False, search_method='tensor_wise', batch_size=32):
    if channel_wise:
        if search_method == 'block_wise':
            layer_inps = wrappers[name].inps
            layer_outs = wrappers[name].outs
        
        ori_weight = wrappers[name].layer.weight.clone().detach()
        ori_shape = wrappers[name].layer.weight.data.shape

        if len(ori_weight.shape) == 2:
            temp_weight = ori_weight.view(ori_weight.shape[0], -1)
        elif len(ori_weight.shape) == 4:
            temp_weight = ori_weight.flatten(1, 3)
        else:
            raise NotImplementedError("Only support 2D and 4D weights for now!")

        layer_quantizer = wrappers[name].quantizer
        scale_method = layer_quantizer.scale_method


        # get the channel-wise max of the weight
        tmp = torch.zeros(temp_weight.shape[0], device=ori_weight.device)
        xmin = torch.minimum(temp_weight.min(1)[0], tmp)
        xmax = torch.maximum(temp_weight.max(1)[0], tmp)
        if layer_quantizer.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            xmin = -xmax
        
        tmp = (xmin == 0) & (xmax == 0)
        # assert that xmin and xmax are not both 0
        assert not torch.any(tmp), "xmin and xmax are both 0"

        # Initialize Scale, Zero, Code Map, and Level Map
        dim_diff = temp_weight.dim() - (xmin.unsqueeze(1)).dim()
        target_shape = (xmin.unsqueeze(1)).shape + (1,) * dim_diff

        best = torch.full_like(xmin.unsqueeze(1), 1e10, device=ori_weight.device) # best shape: [num_channels, 1] 
        scale = torch.zeros_like((xmin.unsqueeze(1)).view(target_shape), device=ori_weight.device) # delta shape: [num_channels, 1, 1, 1]
        zero_point = torch.zeros_like(temp_weight, device=ori_weight.device) # zero_point shape: [num_channels, 1/in_channels, kernel_size/1, kernel_size/1]
        code_map = torch.zeros_like(temp_weight, device=ori_weight.device) # code_map shape: [num_channels, 1/in_channels, kernel_size/1, kernel_size/1]
        level_map = torch.zeros_like(temp_weight, device=ori_weight.device) # level_map shape: [num_channels, 1/in_channels, kernel_size/1, kernel_size/1]
        asym_map = torch.zeros_like((xmin.unsqueeze(1)).view(target_shape), device=ori_weight.device) # asym_map shape: [num_channels, 1, 1, 1]
        err = torch.zeros_like((xmin.unsqueeze(1)).view(target_shape), device=ori_weight.device) # err shape: [num_channels, 1]

        if scale_method == 'log_2':
            raise NotImplementedError("Only support log_dynamic for now!")
        elif scale_method == 'log_sqrt2':
            raise NotImplementedError("Only support log_dynamic for now!")
        elif scale_method == 'log_dynamic':
            search_threshold = 0.0 # search_threshold = 0.999 if disable scaling factor search
            for clip_ratio in torch.arange(1.0, search_threshold, -0.01):
                for code_no_2 in range(int(2 ** (layer_quantizer.n_bits - 1)), -1, -2):
                    code_map1, level_map1, zero1, asym_map1 = layer_quantizer.get_code_map(temp_weight, 
                                                                                        xmax.unsqueeze(1), xmin.unsqueeze(1), 
                                                                                        code_no_2, layer_quantizer.n_bits, 
                                                                                        layer_quantizer.sym)
                    scale1 = torch.ones([temp_weight.shape[0], 1], device=ori_weight.device) * clip_ratio
                    scale1 = scale1.view(target_shape) # scale1 shape: [num_channels, 1, 1, 1]

                    q = log_quantize(temp_weight, scale=scale1, zero=zero1, method=scale_method, code_map=code_map1, level_map=level_map1, asym_map=asym_map1, hardware_approx=layer_quantizer.hardware_approx)
                    if len(ori_weight.shape) == 2:
                        q = q.view(ori_weight.shape[0], -1) 
                    elif len(ori_weight.shape) == 4:
                        q = q.view(ori_shape)
                    else:
                        raise NotImplementedError("Only support 2D and 4D weights for now!")      
                    if search_method == 'tensor_wise':
                        err = q - ori_weight
                        err.abs_()
                        err.pow_(2)
                        if len(err.shape) == 2:
                            err = torch.sum(err.view(err.shape[0], -1), dim=-1, keepdim=True)
                        elif len(err.shape) == 4:
                            err = torch.sum(err, dim=[1, 2, 3]).unsqueeze(1) # err shape: [num_out_channels, 1]
                        else:
                            assert False, 'Wrong shape of err'

                    elif search_method == 'block_wise':
                        wrappers[name].layer.weight.data = q
                        # quantized_out = torch.zeros_like(layer_outs)
                        err.zero_()
                        for i in range(int(layer_inps.size(0) / batch_size)): 
                            quantized_out = wrappers[name].layer(layer_inps[i * batch_size:(i + 1) * batch_size])
                            temp_err = quantized_out - layer_outs[i * batch_size:(i + 1) * batch_size]

                            del quantized_out
                            torch.cuda.empty_cache()

                            temp_err.abs_()
                            temp_err.pow_(2)
                            
                            if len(temp_err.shape) == 4: # err shape: [batch_size, num_out_channels, out_w?, out_h?]
                                temp_err = torch.sum(temp_err, dim=[0, 2, 3]).unsqueeze(1) # err shape: [num_out_channels, 1]
                            elif len(temp_err.shape) == 3:
                                temp_err = torch.sum(temp_err, dim=[0, 1]).unsqueeze(1)
                            elif len(temp_err.shape) == 2:
                                temp_err = torch.sum(temp_err, dim=[0]).unsqueeze(1) # err shape: [num_out_channels, 1]????????
                            else:
                                assert False, 'Wrong shape of err'
                            
                            err += temp_err.detach()
                            
                            del temp_err
                            torch.cuda.empty_cache()

                        wrappers[name].layer.weight.data = ori_weight
                    else:
                        raise NotImplementedError("Only support tensor_wise and block_wise for now!")
                    
                    tmp = (err < best)
                    tmp = tmp.view(target_shape)

                    scale = torch.where(tmp, scale1, scale)
                    zero_point = torch.where(tmp, zero1, zero_point)
                    code_map = torch.where(tmp, code_map1, code_map)
                    level_map = torch.where(tmp, level_map1, level_map)
                    asym_map = torch.where(tmp, asym_map1, asym_map)
                    best = torch.minimum(err, best)

                    del scale1, code_map1, level_map1, zero1, asym_map1
                    torch.cuda.empty_cache()
                
                torch.cuda.empty_cache()

    return scale, zero_point, level_map, code_map, asym_map








import torch
import linklink as link
from quant.quant_layer import QuantModule, StraightThrough, lp_loss, log_quantize
from quant.quant_model import QuantModel
from quant.quant_block import BaseQuantBlock
from quant.adaptive_rounding import AdaRoundQuantizer
from quant.data_utils import save_grad_data, save_inp_oup_data, get_similarity


def block_search(model: QuantModel, block: BaseQuantBlock, search_data: torch.Tensor,
                 search_method: str = 'tensor_wise', scale_method: str = 'dynamic',
                 batch_size: int = 32, opt_mode: str = 'mse', asym: bool = False,
                 include_act_func: bool = True, act_quant: bool = False, multi_gpu: bool = False):
    """
    Block reconstruction to optimize the output from each block.

    :param model: QuantModel
    :param block: BaseQuantBlock that needs to be optimized
    :param search_data: data for hyperparameter searching, typically 128 training images
    :param search_method: tensor-wise or layer-wise or block-wise
    :param scale_method: only dynamic is supported
    :param batch_size: mini-batch size for hyperparameter searching
    :param opt_mode: optimization mode
    :param asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
    :param include_act_func: optimize the output after activation function
    :param act_quant: use activation quantization or not
    :param multi_gpu: use multi-GPU or not
    """
    if scale_method != 'log_dynamic':
        raise NotImplementedError('scale_method {} not supported for hyperparameter searching'
                                  .format(scale_method))

    if not include_act_func:
        org_act_func = block.activation_function
        block.activation_function = StraightThrough()

    if act_quant:
        raise NotImplementedError(
            'Activation quantization is not yet supported for hyperparameter searching')

    # Save block-wise data before hyperparameter searching
    if search_method == 'block_wise':
        torch.cuda.empty_cache()
        block_inps, block_outs = save_inp_oup_data(model, block, search_data, asym, act_quant, batch_size)
    
    # Start hyperparameter searching
    for name, module in block.named_modules():
        if isinstance(module, QuantModule):
            # Save layer-wise data before hyperparameter searching
            if search_method == 'layer_wise':
                torch.cuda.empty_cache()
                layer_inps = {}
                layer_outs = {}
                inps, outs = save_inp_oup_data(model, module, search_data, asym, act_quant, batch_size)
                layer_inps[name] = inps
                layer_outs[name] = outs

            ori_weight = module.org_weight.data.clone().detach()
            new_weight_quantizer = module.weight_quantizer
            channel_wise = new_weight_quantizer.channel_wise

            temp_weights = ori_weight.clone().detach()
            
            # get the channel-wise max of the weight
            weight_reshape = temp_weights.view(temp_weights.shape[0], -1)
            tmp = torch.zeros(temp_weights.shape[0], device=temp_weights.device)
            xmin = torch.minimum(weight_reshape.min(1)[0], tmp)
            xmax = torch.maximum(weight_reshape.max(1)[0], tmp)

            if new_weight_quantizer.sym:
                xmax = torch.maximum(torch.abs(xmin), xmax)
                xmin = -xmax
            
            tmp = (xmin == 0) & (xmax == 0)
            # assert that xmin and xmax are not both 0
            assert not torch.any(tmp), "xmin and xmax are both 0"

            # Initialize hyperparameters
            dim_diff = temp_weights.dim() - (xmin.unsqueeze(1)).dim()
            target_shape = (xmin.unsqueeze(1)).shape + (1,) * dim_diff

            best = torch.full_like(xmin.unsqueeze(1), 1e10) # best shape: [num_channels, 1]
            scale = torch.zeros_like((xmin.unsqueeze(1)).view(target_shape)) # scale shape: [num_channels, 1, 1, 1]
            zero = torch.zeros_like(temp_weights) # zero shape: [num_channels, 1/in_channels, kernel_size/1, kernel_size/1]
            code_map = torch.zeros_like(temp_weights) # code_map shape: [num_channels, 1/in_channels, kernel_size/1, kernel_size/1]
            level_map = torch.zeros_like(temp_weights) # level_map shape: [num_channels, 1/in_channels, kernel_size/1, kernel_size/1]
            asym_map = torch.zeros_like((xmin.unsqueeze(1)).view(target_shape)) # asym_map shape: [num_channels, 1, 1, 1]

            if not channel_wise:
                assert False, 'Per-layer quantization is currently not supported!'
            else:
                search_threshold = 0.0 # search_threshold = 0.999 if disable scaling factor search
                for clip_ratio in torch.arange(1.0, search_threshold, -0.01):
                    for code_no_2 in range(int(2 ** (new_weight_quantizer.n_bits - 1)), -1, -2): # from log2 to DLog to Log_sqrt2
                        code_map1, level_map1, zero1, asym_map1 = new_weight_quantizer.get_code_map(temp_weights, 
                                                                                                    xmax.unsqueeze(1), xmin.unsqueeze(1), 
                                                                                                    code_no_2, new_weight_quantizer.n_bits, 
                                                                                                    new_weight_quantizer.sym)
                        
                        scale1 = torch.ones([temp_weights.shape[0], 1], device=temp_weights.device) * clip_ratio
                        scale1 = scale1.view(target_shape) # scale1 shape: [num_channels, 1, 1, 1]

                        q = log_quantize(temp_weights, scale=scale1, zero=zero1, method=scale_method, code_map=code_map1, level_map=level_map1, asym_map=asym_map1, hardware_approx=new_weight_quantizer.hardware_approx)
                        
                        if search_method == 'tensor_wise':
                            err = q - ori_weight
                            err.abs_()
                            err.pow_(2)
                            err = torch.sum(err.view(err.shape[0], -1), dim=-1, keepdim=True)

                            tmp = (err < best)
                            tmp = tmp.view(target_shape)

                            scale = torch.where(tmp, scale1, scale)
                            zero = torch.where(tmp, zero1, zero)
                            code_map = torch.where(tmp, code_map1, code_map)
                            level_map = torch.where(tmp, level_map1, level_map)
                            asym_map = torch.where(tmp, asym_map1, asym_map)
                            best = torch.minimum(err, best)
                        
                        elif search_method == 'layer_wise':    
                            module.org_weight.data = q
                            quantized_out = torch.zeros_like(layer_outs[name])

                            w_quant = module.use_weight_quant
                            a_quant = module.use_act_quant
                            module.set_quant_state(False, False)

                            for i in range(int(search_data.size(0) / batch_size)):                        
                                with torch.no_grad():
                                    quantized_output = module(layer_inps[name][i * batch_size:(i + 1) * batch_size])
                                    torch.cuda.empty_cache()
                                quantized_out[i * batch_size:(i + 1) * batch_size] = quantized_output
                            
                            module.set_quant_state(w_quant, a_quant) # Restore quantization state
                            module.org_weight.data = ori_weight # Restore original weights

                            err = quantized_out - layer_outs[name]
                            err.abs_()
                            err.pow_(2)
                            err = torch.sum(err, dim=[0, 2, 3]).unsqueeze(1) # err shape: [num_channels, 1]

                            tmp = (err < best)
                            tmp = tmp.view(target_shape) # tmp shape: [num_channels, 1, 1, 1]

                            scale = torch.where(tmp, scale1, scale)
                            zero = torch.where(tmp, zero1, zero)
                            code_map = torch.where(tmp, code_map1, code_map)
                            level_map = torch.where(tmp, level_map1, level_map)
                            asym_map = torch.where(tmp, asym_map1, asym_map)
                            best = torch.minimum(err, best)

                        elif search_method == 'block_wise':
                            assert False, 'Block-wise search is not yet supported!'
                        
                        else:
                            raise NotImplementedError('search_method {} not supported for hyperparameter searching'
                                                      .format(search_method))                    

            module.weight_quantizer.delta = scale
            module.weight_quantizer.zero_point = zero
            module.weight_quantizer.code_map = code_map
            module.weight_quantizer.level_map = level_map
            module.weight_quantizer.asym_map = asym_map 
            module.weight_quantizer.inited = True
            module.set_quant_state(True, act_quant)





    # if opt_mode != 'mse':
    #     cached_grads = save_grad_data(model, block, cali_data, act_quant, batch_size=batch_size)
    # else:
    #     cached_grads = None
    # device = 'cuda'
    # for i in range(iters):
    #     idx = torch.randperm(cached_inps.size(0))[:batch_size]
    #     cur_inp = cached_inps[idx].to(device)
    #     cur_out = cached_outs[idx].to(device)
    #     cur_grad = cached_grads[idx].to(device) if opt_mode != 'mse' else None

    #     optimizer.zero_grad()
    #     out_quant = block(cur_inp)

    #     err = loss_func(out_quant, cur_out, cur_grad)
    #     err.backward(retain_graph=True)
    #     if multi_gpu:
    #         for p in opt_params:
    #             link.allreduce(p.grad)
    #     optimizer.step()
    #     if scheduler:
    #         scheduler.step()

    # torch.cuda.empty_cache()

    # # Finish optimization, use hard rounding.
    # for name, module in block.named_modules():
    #     if isinstance(module, QuantModule):
    #         module.weight_quantizer.soft_targets = False

    # # Reset original activation function
    # if not include_act_func:
    #     block.activation_function = org_act_func



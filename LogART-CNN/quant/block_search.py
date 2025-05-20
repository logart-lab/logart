import torch
import linklink as link
from quant.quant_layer import QuantModule, StraightThrough, lp_loss
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
    model.set_quant_state(False, False)
    block.set_quant_state(True, act_quant)
    if scale_method != 'dynamic':
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
            delta = torch.zeros_like(ori_weight)
            zero_point = torch.zeros_like(ori_weight)
            code_map = torch.zeros_like(ori_weight)
            level_map = torch.zeros_like(ori_weight)

            if channel_wise:
                n_channels = ori_weight.shape[0]
                # determine the 2/sqrt(2) ratio channel-by-channel
                for c in range(n_channels):
                    score_list = []
                    for code2 in range(0, 2 ** (new_weight_quantizer.n_bits - 1) + 1):
                        temp_weights = ori_weight.clone().detach()
                        qw = new_weight_quantizer.log_quantize(temp_weights[c], code_no_2=code2)

                        if search_method == 'tensor_wise':
                            score = get_similarity(ori_weight[c], qw, opt_mode)
                            score_list.append(score)

                        elif search_method == 'layer_wise':                
                            temp_weights[c] = qw
                            module.org_weight.data = temp_weights
                            score_mean = torch.zeros(int(search_data.size(0) / batch_size))

                            w_quant = module.use_weight_quant
                            a_quant = module.use_act_quant
                            module.set_quant_state(False, False)

                            for i in range(int(search_data.size(0) / batch_size)):
                                with torch.no_grad():
                                    quantized_output = module(layer_inps[name][i * batch_size:(i + 1) * batch_size])
                                    torch.cuda.empty_cache()
                                score_mean[i] = get_similarity(layer_outs[name][i * batch_size:(i + 1) * batch_size], 
                                                               quantized_output, opt_mode)
                            
                            module.set_quant_state(w_quant, a_quant) # Restore quantization state
                            module.org_weight.data = ori_weight # Restore original weights

                            score = score_mean.mean()
                            score_list.append(score)

                        elif search_method == 'block_wise':
                            temp_weights[c] = qw
                            module.org_weight.data = temp_weights
                            score_mean = torch.zeros(int(search_data.size(0) / batch_size))

                            w_quant = {}
                            a_quant = {}
                            for name_sqs, module_sqs in block.named_modules():
                                if isinstance(module_sqs, QuantModule):
                                    w_quant[name_sqs] = module_sqs.use_weight_quant
                                    a_quant[name_sqs] = module_sqs.use_act_quant
                                    module_sqs.set_quant_state(False, False)

                            for i in range(int(search_data.size(0) / batch_size)):
                                with torch.no_grad():
                                    quantized_output = block(block_inps[i * batch_size:(i + 1) * batch_size])
                                    torch.cuda.empty_cache()
                                score_mean[i] = get_similarity(block_outs[i * batch_size:(i + 1) * batch_size], 
                                                               quantized_output, opt_mode)
                            
                            # Restore quantization states
                            for name_sqs, module_sqs in block.named_modules():
                                if isinstance(module_sqs, QuantModule):
                                    module_sqs.set_quant_state(w_quant[name_sqs], a_quant[name_sqs])

                            module.org_weight.data = ori_weight # Restore original weights

                            score = score_mean.mean()
                            score_list.append(score)
                        
                        else:
                            raise NotImplementedError('search_method {} not supported for hyperparameter searching'
                                                      .format(search_method))
                    
                    best_score = max(score_list)
                    
                    code_no_2 = score_list.index(best_score)
                    code_map[c], level_map[c], zero_point[c] = new_weight_quantizer.get_code_map(ori_weight[c], 
                                                                                                 code_no_2=code_no_2)
                    
            else:
                score_list = []
                for code2 in range(0, 2 ** (new_weight_quantizer.n_bits - 1) + 1):
                    temp_weights = ori_weight.clone().detach()
                    qw = new_weight_quantizer.log_quantize(temp_weights, code_no_2=code2)

                    if search_method == 'tensor_wise':
                        score = get_similarity(ori_weight, qw, opt_mode)
                        score_list.append(score)
                    
                    elif search_method == 'layer_wise':
                        module.org_weight.data = qw
                        score_mean = torch.zeros(int(search_data.size(0) / batch_size))

                        w_quant = module.use_weight_quant
                        a_quant = module.use_act_quant
                        module.set_quant_state(False, False)

                        for i in range(int(search_data.size(0) / batch_size)):
                            with torch.no_grad():
                                quantized_output = module(layer_inps[name][i * batch_size:(i + 1) * batch_size])
                                torch.cuda.empty_cache()
                            score_mean[i] = get_similarity(layer_outs[name][i * batch_size:(i + 1) * batch_size], 
                                                           quantized_output, opt_mode)                  
                        
                        module.set_quant_state(w_quant, a_quant) # Restore quantization states
                        module.org_weight.data = ori_weight # Restore original weights

                        score = score_mean.mean()
                        score_list.append(score)
                    
                    elif search_method == 'block_wise':
                        module.org_weight.data = qw
                        score_mean = torch.zeros(int(search_data.size(0) / batch_size))
                        
                        w_quant = {}
                        a_quant = {}
                        for name_sqs, module_sqs in block.named_modules():
                            if isinstance(module_sqs, QuantModule):
                                w_quant[name_sqs] = module_sqs.use_weight_quant
                                a_quant[name_sqs] = module_sqs.use_act_quant
                                module_sqs.set_quant_state(False, False)

                        for i in range(int(search_data.size(0) / batch_size)):
                            with torch.no_grad():
                                quantized_output = block(block_inps[i * batch_size:(i + 1) * batch_size])
                                torch.cuda.empty_cache()
                            score_mean[i] = get_similarity(block_outs[i * batch_size:(i + 1) * batch_size], 
                                                           quantized_output, opt_mode)  
                        
                        # Restore quantization state
                        for name_sqs, module_sqs in block.named_modules():
                            if isinstance(module_sqs, QuantModule):
                                module_sqs.set_quant_state(w_quant[name_sqs], a_quant[name_sqs])

                        module.org_weight.data = ori_weight # Restore original weights

                        score = score_mean.mean()
                        score_list.append(score)

                    else:
                        raise NotImplementedError('search_method {} not supported for hyperparameter searching'
                                                    .format(search_method))

                best_score = max(score_list)
                code_no_2 = score_list.index(best_score)
                code_map, level_map, zero_point = new_weight_quantizer.get_code_map(ori_weight, 
                                                                                    code_no_2=code_no_2)

            module.weight_quantizer.delta = delta
            module.weight_quantizer.zero_point = zero_point
            module.weight_quantizer.code_map = code_map
            module.weight_quantizer.level_map = level_map
            module.weight_quantizer.inited = True





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



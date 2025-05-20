import torch
import linklink as link
from quant.quant_layer import QuantModule, StraightThrough, lp_loss, UniformAffineQuantizer
from quant.quant_model import QuantModel
from quant.block_recon import LinearTempDecay
from quant.adaptive_rounding import AdaRoundQuantizer
from quant.data_utils import save_grad_data, save_inp_oup_data, get_similarity


def layer_search(model: QuantModel, layer: QuantModule, search_data: torch.Tensor, 
                 search_method: str = 'tensor_wise', scale_method: str = 'dynamic',
                 batch_size: int = 32, opt_mode: str = 'mse', asym: bool = False,
                 include_act_func: bool = True, act_quant: bool = False, multi_gpu: bool = False):
    """
    Hyperparameter search to optimize the output from each layer.

    :param model: QuantModel
    :param layer: QuantModule that needs to be optimized
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
    layer.set_quant_state(True, act_quant)
    if scale_method != 'dynamic':
        raise NotImplementedError('scale_method {} not supported for hyperparameter searching'
                                  .format(scale_method))

    if not include_act_func:
        org_act_func = layer.activation_function
        layer.activation_function = StraightThrough()

    if act_quant:
        raise NotImplementedError(
            'Activation quantization is not yet supported for hyperparameter searching')

    # Save data before hyperparameter searching
    if search_method in {'layer_wise', 'block_wise'}:
        torch.cuda.empty_cache()
        cached_inps, cached_outs = save_inp_oup_data(model, layer, search_data, asym, act_quant, batch_size)
    
    # Start hyperparameter searching
    ori_weight = layer.org_weight.data.clone().detach()
    new_weight_quantizer = layer.weight_quantizer
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

                elif search_method in {'layer_wise', 'block_wise'}:
                    temp_weights[c] = qw
                    layer.org_weight.data = temp_weights
                    quantized_out = torch.zeros_like(cached_outs)

                    w_quant = layer.use_weight_quant
                    a_quant = layer.use_act_quant
                    layer.set_quant_state(False, False)

                    for i in range(int(search_data.size(0) / batch_size)):                        
                        with torch.no_grad():
                            quantized_output = layer(cached_inps[i * batch_size:(i + 1) * batch_size])
                            torch.cuda.empty_cache()
                        quantized_out[i * batch_size:(i + 1) * batch_size] = quantized_output
                    
                    layer.set_quant_state(w_quant, a_quant) # Restore quantization state
                    layer.org_weight.data = ori_weight # Restore original weights

                    score = get_similarity(cached_outs, quantized_out, opt_mode)
                    score_list.append(score)

                else:
                    raise NotImplementedError('search_method {} not supported for hyperparameter searching'
                                              .format(search_method))
            
            best_score = max(score_list)
            
            code_no_2 = score_list.index(best_score)
            code_map[c], level_map[c], zero_point[c] = new_weight_quantizer.get_code_map(
                ori_weight[c], code_no_2=code_no_2)
    else:
        score_list = []
        for code2 in range(0, 2 ** (new_weight_quantizer.n_bits - 1) + 1):
            temp_weights = ori_weight.clone().detach()
            qw = new_weight_quantizer.log_quantize(temp_weights, code_no_2=code2)

            if search_method == 'tensor_wise':
                score = get_similarity(ori_weight, qw, opt_mode)
                score_list.append(score)

            elif search_method in {'layer_wise', 'block_wise'}:
                layer.org_weight.data = qw
                quantized_out = torch.zeros_like(cached_outs)

                w_quant = layer.use_weight_quant
                a_quant = layer.use_act_quant
                layer.set_quant_state(False, False)

                for i in range(int(search_data.size(0) / batch_size)):
                    with torch.no_grad():
                        quantized_output = layer(cached_inps[i * batch_size:(i + 1) * batch_size])
                        torch.cuda.empty_cache()
                    quantized_out[i * batch_size:(i + 1) * batch_size] = quantized_output
                
                layer.set_quant_state(w_quant, a_quant) # Restore quantization state
                layer.org_weight.data = ori_weight # Restore original weights

                score = get_similarity(cached_outs, quantized_out, opt_mode)
                score_list.append(score)
            
            else:
                raise NotImplementedError('search_method {} not supported for hyperparameter searching'
                                              .format(search_method))

        best_score = max(score_list)
        code_no_2 = score_list.index(best_score)
        code_map, level_map, zero_point = new_weight_quantizer.get_code_map(ori_weight, 
                                                                            code_no_2=code_no_2)

    layer.weight_quantizer.delta = delta
    layer.weight_quantizer.zero_point = zero_point
    layer.weight_quantizer.code_map = code_map
    layer.weight_quantizer.level_map = level_map
    layer.weight_quantizer.inited = True

    




import errno
import torch
import linklink as link
from quant.quant_layer import QuantModule, StraightThrough, lp_loss, UniformAffineQuantizer, log_quantize
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
    if scale_method != 'log_dynamic':
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
            for code_no_2 in range(int(2 ** (new_weight_quantizer.n_bits - 1)), -1, -2):
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
                
                elif search_method in {'layer_wise', 'block_wise'}:
                    layer.org_weight.data = q
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

                    err = quantized_out - cached_outs
                    err.abs_()
                    err.pow_(2)
                    if len(err.shape) == 4: # err shape: [batch_size, num_out_channels, out_w?, out_h?]
                        err = torch.sum(err, dim=[0, 2, 3]).unsqueeze(1) # err shape: [num_out_channels, 1]
                    elif len(err.shape) == 2: # err shape: [batch_size, num_out_channels]
                        err = torch.sum(err, dim=[0]).unsqueeze(1) # err shape: [num_out_channels, 1]
                    else:
                        assert False, 'Wrong shape of err'

                    tmp = (err < best)
                    tmp = tmp.view(target_shape)

                    scale = torch.where(tmp, scale1, scale)
                    zero = torch.where(tmp, zero1, zero)
                    code_map = torch.where(tmp, code_map1, code_map)
                    level_map = torch.where(tmp, level_map1, level_map)
                    asym_map = torch.where(tmp, asym_map1, asym_map)
                    best = torch.minimum(err, best)
                
                else:
                    raise NotImplementedError('search_method {} not supported for hyperparameter searching'
                                              .format(search_method))
                        
    layer.weight_quantizer.delta = scale
    layer.weight_quantizer.zero_point = zero
    layer.weight_quantizer.code_map = code_map
    layer.weight_quantizer.level_map = level_map
    layer.weight_quantizer.asym_map = asym_map 
    layer.weight_quantizer.inited = True
    layer.set_quant_state(True, act_quant)

    




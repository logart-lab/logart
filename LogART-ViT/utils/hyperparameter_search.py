import torch
from torch import nn
import torch.nn.functional as F
from utils.search_utils import *
from quantizers.adaround import AdaRoundQuantizer
from quant_layers import *
from utils.search_wrap_net import WrapNet
import timm

def get_vit_blocks(model, return_model_type=True):
    from timm.layers.patch_embed import PatchEmbed
    from timm.models.vision_transformer import Block as ViTBlock
    from timm.models.swin_transformer import SwinTransformerBlock, PatchMerging

    types_of_block = (
        PatchEmbed,
        ViTBlock,
        SwinTransformerBlock,
        PatchMerging,
    )
    blocks = []
    names = []
    for name, module in model.named_modules():
        if isinstance(module, types_of_block) or name.split('.')[-1] == 'head':
            blocks.append(module)
            names.append(name)
    return blocks

def input_caching(model, calib_data, dev):
    #  transformer blocks（list）
    transformer_blocks = get_vit_blocks(model, return_model_type=False)

    # first block
    first_block = transformer_blocks[0]

    # class InputCatcher
    class InputCatcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.n_data = 0
            self.inps = []

        def forward(self, *inp, **kwargs):
            self.inps.append([i.detach().to(torch.float32) for i in inp])
            self.n_data += 1
            raise ValueError  # 中断 forward

    model = model.to(dev)

    # Replace the first block with InputCatcher
    # find the parent module of the first block
    for name, module in model.named_modules():
        if module is first_block:
            submodules = name.split('.')
            parent_module = model
            for sub_name in submodules[:-1]:
                parent_module = getattr(parent_module, sub_name)
            catcher = InputCatcher(first_block.to(dev))
            setattr(parent_module, submodules[-1], catcher)
            break

    # catch the input of the first block
    try:
        inputs = calib_data.to(dev)
        model(inputs)
    except ValueError:
        pass

    # restore the first block
    setattr(parent_module, submodules[-1], first_block.to(dev))

    torch.cuda.empty_cache()

    return catcher.inps[0][0]


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def hyperparameter_search(model, backup_model, search_data, cfg, device="cpu"):
    transformer_blocks = get_vit_blocks(model)
    transformer_blocks_backup = get_vit_blocks(backup_model)
    org_dtype = next(iter(model.parameters())).dtype

    if cfg.search_method == 'tensor_wise': # tensor-wise does not require input
        inps = None
    elif cfg.search_method == 'block_wise':
        ori_inps = search_data.to(device)
        inps = ori_inps
    else:
        raise NotImplementedError("Only support tensor_wise and block_wise for now!")
    
    quantizers = {}
    for idx_block in range(len(transformer_blocks)):
        print('-' * 50)
        print(f'>>>> Quantize {idx_block+1}-th Transformer Block.... ({idx_block+1}/{len(transformer_blocks)})')
        transformer_block = transformer_blocks[idx_block].to(device=device, dtype=torch.float32)
        fp_layers = find_layers(transformer_block, layers=[nn.Conv2d, nn.Linear]) 
        fp_layers_backup = find_layers(transformer_blocks_backup[idx_block], layers=[AsymmetricallyBatchingQuantConv2d, AsymmetricallyBatchingQuantLinear])
        wrappers = {}
        for name, fp_layer in fp_layers.items():
            wrappers[name] = WrapNet(fp_layer)
            wrappers[name].i, wrappers[name].name = idx_block, name
            wrappers[name].quantizer = UniformQuantizer(n_bits=cfg.w_bit, 
                                                        symmetric=cfg.w_sym, 
                                                        channel_wise=True, 
                                                        scale_method=cfg.scale_method, 
                                                        hardware_approx=cfg.hardware_approx)
            wrappers[name].quantizer.init_quantization_scale(wrappers[name].layer.weight.data, channel_wise=True)

        if cfg.search_method == 'tensor_wise':
            pass
        elif cfg.search_method == 'block_wise':
            def record_inp_outp(n,wrappers=wrappers):
                def hook(module, inp, outp, wrappers=wrappers, n=n):
                    # inp[0] tensor
                    wrappers[n].inps = inp[0].detach()
                    wrappers[n].outs = outp.detach()
                return hook

            hooks = []
            for n, m in transformer_block.named_modules():
                if type(m) in [nn.Conv2d, nn.Linear]: 
                    hooks.append(m.register_forward_hook(record_inp_outp(n, wrappers=wrappers)))

            
            with torch.no_grad():
                    _ = transformer_block(inps.to(device=device))

            for h in hooks:
                h.remove()
        
        else:
            raise NotImplementedError("Only support tensor_wise and block_wise for now!")

        last_block = False
        print(f">>> Begin Quantization Parameters")
        if idx_block == len(transformer_blocks) - 1:
            last_block = True

        # Perform hyperparameter search
        search_qparams(block=transformer_block, 
                       wrappers=wrappers, 
                      inps=inps,
                       last_block=last_block, 
                       search_method=cfg.search_method, 
                       batch_size=cfg.calib_batch_size)
        
        # Quantize the block
        for name, wrapper in wrappers.items():
            fp_layers_backup[name].w_quantizer.zero_point = wrapper.quantizer.zero_point
            fp_layers_backup[name].w_quantizer.code_map = wrapper.quantizer.code_map
            fp_layers_backup[name].w_quantizer.level_map = wrapper.quantizer.level_map
            fp_layers_backup[name].w_quantizer.scale = wrapper.quantizer.scale
            fp_layers_backup[name].w_quantizer.asym_map = wrapper.quantizer.asym_map
            fp_layers_backup[name].w_quantizer.hardware_approx = wrapper.quantizer.hardware_approx
            fp_layers_backup[name].w_quantizer.inited = True

            wrapper.quant()
            quantizers['%d.%s' % (idx_block, name)] = wrapper.quantizer
            wrapper.free()
        transformer_blocks[idx_block] = transformer_block.to(device=device, dtype=org_dtype)########

        # Update the input for the next block
        if cfg.search_method == 'tensor_wise':
            pass
        elif cfg.search_method == 'block_wise':
            # inps = transformer_block(inps.to(device=device))
            if idx_block + 1 < len(transformer_blocks):
                next_block = transformer_blocks[idx_block + 1]

                captured = {}

                def hook(module, inp, outp):
                    captured['inps'] = inp[0].detach()

                hook_handle = next_block.register_forward_hook(hook)

                with torch.no_grad():
                    _ = model(ori_inps.to(device=device))

                hook_handle.remove()

                inps = captured['inps']

        else:
            raise NotImplementedError("Only support tensor_wise and block_wise for now!")

        del transformer_block
        del wrappers 
        torch.cuda.empty_cache()

    return quantizers

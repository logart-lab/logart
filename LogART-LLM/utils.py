from sympy.logic import false
from sympy.printing.latex import true
from torch import nn

def find_layers(module, layers=[nn.Conv2d, nn.Linear, nn.Embedding], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def get_ptq_arguments(**parser_kwargs):
    import argparse
    parser = argparse.ArgumentParser(**parser_kwargs)
    
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    
    ## Model
    parser.add_argument("--model_path", type=str, default='facebook/opt-125m', help='path of the model to be quantized')
    parser.add_argument("--save_model", action='store_true', help='Whether to save the fake-quantized model')
    
    ## Calib. Data
    parser.add_argument('--calib_data', type=str, default="wikitext2", choices=["c4", "wikitext2"])
    parser.add_argument('--nsamples', type=int, default=32, help='Number of calibration data samples.')
    parser.add_argument('--seqlen', type=int, default=2048, help='maximum sequence length')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')

    ## Quantization Configs
    parser.add_argument('--w_bits', type=int, default=3)
    parser.add_argument('--w_sym', action='store_true', help='Whether to perform symmetric weight quantization')
    parser.add_argument('--scale_search', action='store_true', help='Whether to perform scaling factor search')
    parser.add_argument('--w_method', type=str, default='log_dynamic', choices=['log_2', 'log_sqrt2', 'log_dynamic'], help='Quantization Method')
    parser.add_argument('--group_size', type=int, default=-1)
    parser.add_argument('--hardware_approx', action='store_true', help='Whether to use hardware approximation for quantization')
    parser.add_argument('--test_with_hardware_approx', action='store_true', help='Whether to use hardware approximation for final test of quantization')
    
    ## aespa Options
    parser.add_argument('--block_v', action="store_true", help="Whether to apply block-wise objective for the value projection")

    # Quantization Parameters Computation (scale and zero)
    parser.add_argument('--use_zfold', action='store_true', help="Whether to apply Z-Fold")

    # Integer Weight Optimization
    parser.add_argument('--optq_init', action="store_true", help="Whether to update weights based on OPTQ before learning weight-rounding policy")
    parser.add_argument('--act_order', action='store_true', help='Whether to apply Hessian-based re-ordering (heuristic in OPTQ)')
    parser.add_argument('--learn_rounding', action='store_true', help='Whether to perform pre-computation-based weight-rounding policy learning')

    # Hyperparams.
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate for adaround training')
    parser.add_argument('--round_weight', type=float, default=1.0, help=' weight of rounding loss in adaround')
    parser.add_argument('--round_weight_qkv', type=float, default=1.5, help='rounding loss weight for QKV')
    parser.add_argument('--num_iters', type=int, default=500, help='number of iterations for adaround training')
    
    parser.add_argument('--replace', type=float, default=1/2048, choices=[1.0, 1/2048], help='Value to be replaced for the Hessian diagonal elements corresponding to dead neurons')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')

    return parser.parse_args() 


def get_aespa_weight_quant_infos(args):
    qconfigs = {
        "w_bits": args.w_bits, "w_sym": args.w_sym, "scale_search": args.scale_search, "w_method": args.w_method, "hardware_approx": args.hardware_approx, "group_size": args.group_size, "test_with_hardware_approx": args.test_with_hardware_approx
    }
    aespa_opts = {
        "block_v": args.block_v,
        'use_zfold': args.use_zfold, 
        "round_optim": {"optq_init": args.optq_init, "learn_rounding": args.learn_rounding, "test_with_hardware_approx": args.test_with_hardware_approx}
    }
    if args.optq_init:
        aespa_opts['round_optim']['act_order'] = args.act_order
    if args.learn_rounding:
        aespa_opts['round_optim']['lr'] = args.lr
        aespa_opts['round_optim']['round_weight'] = args.round_weight
        aespa_opts['round_optim']['round_weight_qkv'] = args.round_weight_qkv
        aespa_opts['round_optim']['num_iters'] = args.num_iters
        
    hyperparams = {"replace": args.replace, "percdamp": args.percdamp}
    
    return qconfigs, aespa_opts, hyperparams


def save_ppl_results(ppl_results, process_time, args):
    import os
    from pathlib import Path
    import csv

    data = [{**ppl_results, **{"time": process_time}}]
    output_dir = "results"
    if not os.path.exists(Path(output_dir)):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"aespa-{args.model_name}-w{args.w_bits}-{'sym' if args.w_sym else 'asym'}-{args.calib_data}_{args.nsamples}_{args.seqlen}_{args.seed}"
    if args.block_v:
        filename += "-block_v"
    if args.use_zfold:
        filename += "-zfold"
    if args.optq_init:
        filename += "-optq_init"
        if args.act_order:
            filename += "_act_order"
    if args.learn_rounding:
        filename += f"-learn_rounding-lr_{args.lr}_rw_{args.round_weight}_rwqkv_{args.round_weight_qkv}_niters_{args.num_iters}"
    filename += ".csv"
    with open(os.path.join(output_dir, filename), "w", newline='') as file:
        header = data[0].keys()
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def set_qmodel_dir(args):
    qmodel_dir = f"{args.model_name}-w{args.w_bits}-{'sym' if args.w_sym else 'asym'}-{args.calib_data}_{args.nsamples}_{args.seqlen}_{args.seed}"
    if args.block_v:
        qmodel_dir += "-block_v"
    if args.use_zfold:
        qmodel_dir += "-zfold"
    if args.optq_init:
        qmodel_dir += "-optq_init"
        if args.act_order:
            qmodel_dir += "_act_order"
    if args.learn_rounding:
        qmodel_dir += f"-learn_rounding-lr_{args.lr}_rw_{args.round_weight}_rwqkv_{args.round_weight_qkv}_niters_{args.num_iters}"

    return qmodel_dir
import os
import sys
import torch
from torch import nn
import numpy as np
import argparse
import importlib
import timm
import copy
import time

import utils.datasets as mydatasets
from utils.calibrator import QuantCalibrator
from utils.block_recon import BlockReconstructor
from utils.mlp_recon import MLPReconstructor
from utils.wrap_net import wrap_modules_in_net, wrap_reparamed_modules_in_net
from utils.test_utils import *
from utils.csv_log import *
from datetime import datetime
import logging

while True:
    try:
        timestamp = datetime.now()
        formatted_timestamp = timestamp.strftime("%Y%m%d_%H%M")
        root_path = './checkpoint/quant_result/{}'.format(formatted_timestamp)
        os.makedirs(root_path)
        break
    except FileExistsError:
        time.sleep(10)
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[
                        logging.FileHandler('{}/output.log'.format(root_path)),
                        logging.StreamHandler()
                    ])

import builtins
original_print = builtins.print
def custom_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    original_print(*args, **kwargs)
builtins.print = custom_print

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default="vit_base",
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large',
                                 'deit_tiny', 'deit_small', 'deit_base', 
                                 'swin_tiny', 'swin_small', 'swin_base', 'swin_base_384'],
                        help="model")
    parser.add_argument('--config', type=str, default="./configs/4bit/best.py",
                        help="File path to import Config class from")
    parser.add_argument('--dataset', default="./dataset",
                        help='path to dataset')
    parser.add_argument("--calib-size", default=argparse.SUPPRESS,
                        type=int, help="size of calibration set")
    parser.add_argument("--calib-batch-size", default=argparse.SUPPRESS,
                        type=int, help="batchsize of calibration set")
    parser.add_argument("--val-batch-size", default=500,
                        type=int, help="batchsize of validation set")
    parser.add_argument("--num-workers", default=8, type=int,
                        help="number of data loading workers (default: 8)")
    parser.add_argument("--device", default="cuda", type=str, help="device")

    parser.add_argument('--reconstruct-mlp', action='store_true', help='reconstruct mlp with ReLU function.')
    parser.add_argument('--load-reconstruct-checkpoint', type=str, default=None, 
                        help='Path to the reconstructed checkpoint.')
    parser.add_argument('--test-reconstruct-checkpoint', action='store_true', 
                        help='validate the reconstructed checkpoint.')
    
    calibrate_mode_group = parser.add_mutually_exclusive_group()
    calibrate_mode_group.add_argument('--calibrate', action='store_true', help="Calibrate the model")
    calibrate_mode_group.add_argument('--load-calibrate-checkpoint', type=str, default=None, 
                                      help="Path to the calibrated checkpoint.")
    parser.add_argument('--test-calibrate-checkpoint', action='store_true', 
                        help='validate the calibrated checkpoint.')

    optimize_mode_group = parser.add_mutually_exclusive_group()
    optimize_mode_group.add_argument('--optimize', action='store_true', default=True, 
                                     help="Optimize the model")
    optimize_mode_group.add_argument('--load-optimize-checkpoint', type=str, default=None, 
                                     help="Path to the optimized checkpoint.")
    parser.add_argument('--test-optimize-checkpoint', action='store_true', 
                        help='validate the optimized checkpoint.')

    parser.add_argument("--print-freq", default=10,
                        type=int, help="print frequency")
    parser.add_argument("--seed", default=3407, type=int, help="seed")
    parser.add_argument('--w_bit', type=int, default=argparse.SUPPRESS, help='bit-precision of weights')
    parser.add_argument('--a_bit', type=int, default=argparse.SUPPRESS, help='bit-precision of activation')
    parser.add_argument("--recon-metric", type=str, default=argparse.SUPPRESS, 
                        choices=['hessian_perturb', 'mse', 'mae'], help='mlp reconstruction metric')
    parser.add_argument("--iters_w", default=20000, type=int, help="number of iterations in AdaRound")
    parser.add_argument('--rrweight', type=float, default=0.01, 
                        help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument("--scale-method", type=str, default='linear_minmax', 
                        choices=['linear_mse', 'linear_minmax', 'log_2', 'log_sqrt2', 'dynamic'],
                        help='scale method')
    parser.add_argument("--hardware_approx", action='store_true', help='1+1/2 hardware approximation method')
    parser.add_argument("--calib-metric", type=str, default=argparse.SUPPRESS, choices=['mse', 'mae'], 
                        help='calibration metric')
    parser.add_argument("--optim-metric", type=str, default=argparse.SUPPRESS, 
                        choices=['hessian', 'hessian_perturb', 'mse', 'mae'], help='optimization metric')
    parser.add_argument('--optim-mode', type=str, default=argparse.SUPPRESS, choices=['qinp', 'rinp', 'qdrop'], 
                        help='`qinp`: use quanted input; `rinp`: use raw input; `qdrop`: use qdrop input.')
    parser.add_argument('--drop-prob', type=float, default=argparse.SUPPRESS, 
                        help='dropping rate in qdrop. set `drop-prob = 1.0` if do not use qdrop.')
    parser.add_argument('--pct', type=float, default=argparse.SUPPRESS, help='clamp percentile of mlp.fc2.')
    return parser


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cur_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_model(model, args, cfg, mode='calibrate'):
    assert mode in ['calibrate', 'optimize']
    if mode == 'calibrate':
        auto_name = '{}_w{}_a{}_calibsize_{}_{}.pth'.format(
            args.model, cfg.w_bit, cfg.a_bit, cfg.calib_size, cfg.calib_metric)
    else:
        auto_name = '{}_w{}_a{}_optimsize_{}_{}_{}{}.pth'.format(
            args.model, cfg.w_bit, cfg.a_bit, cfg.optim_size, cfg.optim_metric, cfg.optim_mode, '_recon' if False else '')
    save_path = os.path.join(root_path, auto_name)

    logging.info(f"Saving checkpoint to {save_path}")
    torch.save(model.state_dict(), save_path)


def load_model(model, args, device, mode='calibrate'):
    assert mode in ['calibrate', 'optimize']
    ckpt_path = args.load_calibrate_checkpoint if mode == 'calibrate' else args.load_optimize_checkpoint
    ckpt = torch.load(ckpt_path)
    for name, module in model.named_modules():
        if hasattr(module, 'mode'):
            module.calibrated = True
            module.mode = 'quant_forward'
        if isinstance(module, nn.Linear) and 'reduction' in name:
            module.bias = nn.Parameter(torch.zeros(module.out_features))
        quantizer_attrs = ['a_quantizer', 'w_quantizer', 'A_quantizer', 'B_quantizer']
        for attr in quantizer_attrs:
            if hasattr(module, attr):
                getattr(module, attr).inited = True
                ckpt_name = name + '.' + attr + '.scale'
                getattr(module, attr).scale.data = ckpt[ckpt_name].clone()
 
    result = model.load_state_dict(ckpt, strict=False)
    logging.info(str(result))
    model.to(device)
    model.eval()
    return model

    
def main(args):
    logging.info("{} - start the process.".format(get_cur_time()))
    logging.info(str(args))
    dir_path = os.path.dirname(os.path.abspath(args.config))
    if dir_path not in sys.path:
        sys.path.append(dir_path)
    module_name = os.path.splitext(os.path.basename(args.config))[0]
    imported_module = importlib.import_module(module_name)
    Config = getattr(imported_module, 'Config')
    logging.info("Successfully imported Config class!")
        
    cfg = Config()
    cfg.calib_size = args.calib_size if hasattr(args, 'calib_size') else cfg.calib_size
    cfg.calib_batch_size = args.calib_batch_size if hasattr(args, 'calib_batch_size') else cfg.calib_batch_size
    cfg.recon_metric = args.recon_metric if hasattr(args, 'recon_metric') else cfg.recon_metric
    cfg.scale_method = args.scale_method if hasattr(args, 'scale_method') else cfg.scale_method
    cfg.hardware_approx = args.hardware_approx if hasattr(args, 'hardware_approx') else cfg.hardware_approx 
    cfg.calib_metric = args.calib_metric if hasattr(args, 'calib_metric') else cfg.calib_metric
    cfg.optim_metric = args.optim_metric if hasattr(args, 'optim_metric') else cfg.optim_metric
    cfg.optim_mode = args.optim_mode if hasattr(args, 'optim_mode') else cfg.optim_mode
    cfg.drop_prob = args.drop_prob if hasattr(args, 'drop_prob') else cfg.drop_prob
    cfg.reconstruct_mlp = False
    cfg.pct = args.pct if hasattr(args, 'pct') else cfg.pct
    cfg.w_bit = args.w_bit if hasattr(args, 'w_bit') else cfg.w_bit
    cfg.a_bit = args.a_bit if hasattr(args, 'a_bit') else cfg.a_bit
    for name, value in vars(cfg).items():
        logging.info(f"{name}: {value}")
        
    device = torch.device(args.device)
    
    model_zoo = {
        'vit_tiny'  : 'vit_tiny_patch16_224',
        'vit_small' : 'vit_small_patch16_224',
        'vit_base'  : 'vit_base_patch16_224',
        'vit_large' : 'vit_large_patch16_224',

        'deit_tiny' : 'deit_tiny_patch16_224',
        'deit_small': 'deit_small_patch16_224',
        'deit_base' : 'deit_base_patch16_224',

        'swin_tiny' : 'swin_tiny_patch4_window7_224',
        'swin_small': 'swin_small_patch4_window7_224',
        'swin_base' : 'swin_base_patch4_window7_224',
        'swin_base_384': 'swin_base_patch4_window12_384',
    }

    seed_all(args.seed)
    
    logging.info('Building model ...')
    try:
        model = timm.create_model(model_zoo[args.model], 
                                  checkpoint_path='./checkpoint/vit_raw/{}.bin'.format(model_zoo[args.model]))
    except:
        model = timm.create_model(model_zoo[args.model], pretrained=True)
    full_model = copy.deepcopy(model)
    full_model.to(device)
    full_model.eval()
    model.to(device)
    model.eval()
    data_path = args.dataset
    g = mydatasets.ViTImageNetLoaderGenerator(data_path, args.val_batch_size, args.num_workers, kwargs={"model":model})
    
    logging.info('Building validation dataloader ...')
    val_loader = g.val_loader()
    criterion = nn.CrossEntropyLoss().to(device)

    reparam = args.load_calibrate_checkpoint is None and args.load_optimize_checkpoint is None
    logging.info('Wraping quantiztion modules (reparam: {}, recon: {}) ...'.format(reparam, False)) 
    model = wrap_modules_in_net(model, cfg, reparam=reparam, recon=False)
    model.to(device)
    model.eval()

    # Weight Quantization initialization
    search_loader = g.calib_loader(num=1, batch_size=1, seed=args.seed)
    for name, module in model.named_modules():
        if hasattr(module, 'mode'):
            module.mode = "debug_only_quant_weight"
    start_time = time.time()
    validate(search_loader, model, criterion, print_freq=args.print_freq, device=device)

    if args.optimize:
        logging.info('Building calibrator ...')
        calib_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size, seed=args.seed)
        logging.info("{} - start {} guided block reconstruction".format(get_cur_time(), cfg.optim_metric))
        block_reconstructor = BlockReconstructor(model, full_model, calib_loader, metric=cfg.optim_metric, 
                                                 temp=cfg.temp, use_mean_hessian=cfg.use_mean_hessian, 
                                                 scale_method=args.scale_method, hardware_approx=args.hardware_approx, 
                                                 iters_w=args.iters_w, rrweight=args.rrweight)
        block_reconstructor.reconstruct_model(quant_act=False, mode=cfg.optim_mode, drop_prob=cfg.drop_prob, 
                                              keep_gpu=cfg.keep_gpu)
        
        end_time = time.time()

        logging.info("{} - {} guided block reconstruction finished.".format(get_cur_time(), cfg.optim_metric))
        logging.info('Validating on test set after block reconstruction ...')
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, 
                                                  device=device)
    
    
    print(f"Runtime：{end_time - start_time:.4f} s")

    logging.info("{} - finished the process.".format(get_cur_time()))

    experiment_set = {
        'arch': args.model,
        'n_bits_w': cfg.w_bit,
        'channel_wise': True,
        'search_method': 'tensor_wise',
        'search_samples': 0,
        'scale_method' : cfg.scale_method,
        'hardware_approx': cfg.hardware_approx,
        'calib_samples': cfg.optim_size,
        'iters_w': args.iters_w,
        'rounding_vs_reconstruction': args.rrweight,
        'config': args.config,
    }
    results = {
        'Acc@1': format(val_prec1, ".2f"),
        'Acc@5': format(val_prec5, ".2f"),
        'GPU_hours': (end_time - start_time) / 3600,
        }
    log_to_csv('results.csv', experiment_set, results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)


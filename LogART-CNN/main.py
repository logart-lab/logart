import torch
import torch.nn as nn
import argparse
import os
import random
import numpy as np
import time
import hubconf
from quant import *
from data.imagenet import build_imagenet_data
import torchvision.models as models
from linklink.log_helper import log_to_csv


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.no_grad()
def validate_model(val_loader, model, device=None, print_freq=100):
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def get_train_samples(train_loader, num_samples):
    train_data = []
    for batch in train_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='running parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general parameters for data and model
    parser.add_argument('--seed', default=0, type=int, help='random seed for results reproduction')
    parser.add_argument('--arch', default='mobilenetv2', type=str, help='dataset name',
                        choices=['resnet18', 'resnet50', 'squeezenet1_1', 'mobilenetv2', 'efficientnet-b0', 
                                 'regnetx_3200m', 'mnasnet'])
    parser.add_argument('--batch_size', default=32, type=int, help='mini-batch size for data loader')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loader')
    parser.add_argument('--data_path', default='/data', type=str, help='path to ImageNet data')
    parser.add_argument('--record_results', action='store_true', 
                        help='whether to record the results')

    # quantization parameters
    parser.add_argument('--n_bits_w', default=4, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--channel_wise', action='store_true',
                        help='apply channel_wise quantization for weights')
    parser.add_argument('--scale_method', default='log_dynamic', type=str, 
                        help='Linear scale method or logarithmic base',
                        choices=['linear_mse', 'linear_minmax', 'log_2', 'log_sqrt2', 'log_dynamic'])
    parser.add_argument('--w_sym', action='store_true', help='symmetric weight quantization')
    parser.add_argument('--test_before_calibration', action='store_true')
    parser.add_argument('--hardware_approx', action='store_true',
                        help='apply hardware-level log sqrt2 approximation to 1+1/2')
    parser.add_argument('--test_with_hardware_approx', action='store_true')

    # hyperparameter search parameters, only dynamic scale method is supported
    parser.add_argument('--search_samples', default=32, type=int, 
                        help='size of the hyperparameter searching dataset') 
    parser.add_argument('--search_method', default='layer_wise', type=str, 
                        help='Search method for hyperparameters',
                        choices=['tensor_wise', 'layer_wise', 'block_wise'])
    
    # weight calibration parameters
    parser.add_argument('--calib_samples', default=2048, type=int, help='size of the calibration dataset') 
    parser.add_argument('--iters_w', default=2000, type=int, help='number of iteration for adaround') 
    parser.add_argument('--calib_lr', default=0.05, type=float, help='learning rate for AdaRound')
    parser.add_argument('--weight', default=1, type=float, 
                        help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--sym', action='store_true', help='symmetric reconstruction, not recommended')
    parser.add_argument('--b_start', default=20, type=int, help='temperature at the start of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, 
                        help='in the warmup period no regularization is applied')    

    args = parser.parse_args()

    seed_all(args.seed)
    # build imagenet data loader
    train_loader, test_loader = build_imagenet_data(train_batch_size=args.batch_size, 
                                                    test_batch_size=64, workers=args.workers,
                                                    data_path=args.data_path)

    # load model
    if args.arch.startswith('squeezenet'):
        cnn = models.__dict__[args.arch](pretrained=True)
    else:
        cnn = eval('hubconf.{}(pretrained=True)'.format(args.arch))
    cnn.cuda()
    cnn.eval()

    # Start counting GPU hours
    start_time = time.time()

    # build quantization parameters
    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 
                 'symmetric': args.w_sym, 'scale_method': args.scale_method, 
                 'hardware_apprx': args.hardware_approx}
    qnn = QuantModel(model=cnn, weight_quant_params=wq_params)
    qnn.cuda()
    qnn.eval()

    # Hyperparameter searching dataset preparation
    search_data = get_train_samples(train_loader, num_samples=args.search_samples)
    device = next(qnn.parameters()).device

    # Initialize weight quantization parameters
    qnn.set_quant_state(True, False)    
    # Kwargs for weight rounding calibration
    kwargs = dict(search_data=search_data, search_method=args.search_method, asym=True, 
                  scale_method=args.scale_method, opt_mode='mse')
    
    # Start calibration hyperparameter search
    def calib_search_model(model: nn.Module):
        """
        Hyperparameter searching for calibration. 
        For the first and last layers, we can only apply layer-wise search.
        """
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Hyperparameter searching for layer {}'.format(name))
                    layer_search(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    print('Hyperparameter searching for block {}'.format(name))
                    block_search(qnn, module, **kwargs)
            else:
                calib_search_model(module)
    
    # Start hyperparameter searching
    calib_search_model(qnn)
    search_end_time = time.time()
    print(f"Search Runtime：{search_end_time - start_time:.4f} s")

    # Test the model before calibration
    if args.test_before_calibration:
        qnn.set_quant_state(True, False)
        print('Quantized accuracy before reconstruction: {}'.format(validate_model(test_loader, qnn)))

    # Calibration dataset preparation
    cali_data = get_train_samples(train_loader, num_samples=args.calib_samples)
    device = next(qnn.parameters()).device

    # Initialize weight quantization parameters
    qnn.set_quant_state(True, False)
    # Kwargs for weight rounding calibration
    kwargs = dict(cali_data=cali_data, iters=args.iters_w, lr=args.calib_lr, weight=args.weight, asym=True,
                  b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, opt_mode='mse')

    def recon_model(model: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(name))
                    block_reconstruction(qnn, module, **kwargs)
            else:
                recon_model(module)

    # Start reconstruction
    recon_model(qnn)

    # Test the model with hardware-level approximation
    if args.test_with_hardware_approx:
        for module in qnn.modules():
            if isinstance(module, QuantModule):
                if hasattr(module.weight_quantizer, 'hardware_approx'):
                    module.weight_quantizer.hardware_approx = True

    qnn.set_quant_state(weight_quant=True, act_quant=False)

    end_time = time.time()
    print(f"Runtime：{end_time - start_time:.4f} s")

    # Test the model
    acc1, acc5 = validate_model(test_loader, qnn)

    # Record Results
    if args.record_results:
        experiment_set = {
            'arch': args.arch,
            'seed': args.seed,
            'n_bits_w': args.n_bits_w,
            'channel_wise': args.channel_wise,
            'w_sym': args.w_sym,
            'search_method': args.search_method,
            'search_samples': args.search_samples,
            'scale_method' : args.scale_method,
            'hardware_approx': args.hardware_approx,
            'calib_samples': args.calib_samples,
            'iters_w': args.iters_w,
            'calib_lr': args.calib_lr,
            'rounding_vs_reconstruction': args.weight,
            'test_with_hardware_approx': args.test_with_hardware_approx
        }
        results = {
            'Acc@1': format(acc1.item(), ".2f"),
            'Acc@5': format(acc5.item(), ".2f"),
            'GPU_hours': (end_time - start_time) / 60,
            'Search_hours': (search_end_time - start_time) / 60,
            'Reconstruction_hours': (end_time - search_end_time) / 60
            }
        log_to_csv('results.csv', experiment_set, results)

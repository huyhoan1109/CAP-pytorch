import os
import torch
import argparse
from MA.EMA import EMA
from datasets import SemiData
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from models.BaseNet import BaseNet
from losses import compute_batch_loss
from backbone.convnext2 import convnextv2_base
from augmentation.transforms import get_pre_transform, get_multi_transform
from utils import save_checkpoints, load_checkpoints, str2bool, neg_log
from config import CHECKPOINT_PATH, DATASET_INFO, WARMUP_EPOCH, LAMBDA_U, TOTAL_EPOCH, T, SCHEDULER, OPTIMIZER

def parse_args():
    parser = argparse.ArgumentParser(
        description='Semi supervised model',
        epilog='Example: python run.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', type=str, default='voc2012', help='Using dataset with model')
    parser.add_argument('--train', type=str2bool, default=True, help='Training semi model')
    parser.add_argument('--eval', type=str2bool, default=False, help='Evaluating semi model')
    parser.add_argument('--use-ema', type=str2bool, default=True, help='Using exponential moving average model')
    parser.add_argument('--ema-decay', type=int, default=0.999, help='Moving average decay')
    parser.add_argument('--resume', type=str2bool, default=True, help='Resume training from checkpoint')
    parser.add_argument('--e-stop', type=int, default=5, help='Use early stoping for model training')
    parser.add_argument('--cp-path', type=str, default=CHECKPOINT_PATH, help='Path to a directory that store checkpoints')
    parser.add_argument('--eval-it', type=int, default=1000, help='Path to a directory that store checkpoints')
    parser.add_argument('--sch', type=int, default=1, choices=SCHEDULER.keys(), help='Choose scheduler type')
    parser.add_argument('--opt', type=int, default=1, choices=OPTIMIZER.keys(), help='Choose optimizer type')    
    args = parser.parse_args()
    return args

def get_loaders(args):
    dataset = args.dataset
    labeled_dataset = SemiData(
        DATASET_INFO[dataset]['images'], 
        DATASET_INFO[dataset]['meta'], 
        pre_transform = get_pre_transform()
    )
    
    labeled_loader = DataLoader(
        labeled_dataset, 
        64, 
        True, 
        drop_last=True
    )
    
    unlabeled_dataset = SemiData(
        DATASET_INFO[dataset]['images'], 
        DATASET_INFO[dataset]['meta'], 
        pre_transform = get_pre_transform(),
        multi_transform = get_multi_transform(),
        mode='unlabeled'
    )
    
    unlabeled_loader = DataLoader(
        unlabeled_dataset, 
        64, 
        True,
        drop_last=True
    )
    
    valid_dataset = SemiData(
        DATASET_INFO[dataset]['images'], 
        DATASET_INFO[dataset]['meta'], 
        pre_transform = get_pre_transform(), 
        mode='valid'
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        64, 
        True,
        drop_last=True
    )

    test_dataset = SemiData(
        DATASET_INFO[dataset]['images'], 
        DATASET_INFO['voc2012']['meta'], 
        pre_transform = get_pre_transform(), 
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset, 
        64, 
        True,
        drop_last=True
    )

    loaders = {
        'labeled': labeled_loader, 
        'unlabeled': unlabeled_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    return loaders

def get_lr_scheduler(args, optimizer):
    sch = SCHEDULER.get(args.sch, None)
    if sch['name'] == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, sch['step_size'])
    elif sch['name'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, sch['T_max'], sch['eta_min'])
    elif sch['name'] == 'OneCycleLR':
        scheduler = lr_scheduler.OneCycleLR(optimizer, sch['max_lr'])
    else:
        raise 'Scheduler not available!'
    return scheduler

def get_optimizer(args, model):
    opt = OPTIMIZER.get(args.opt, None)
    params = model.parameters()
    if opt['name'] == 'SGD':
        optimizer = optim.SGD(
            params=params, 
            lr=opt['lr'], 
            momentum=opt['momentum'], 
            weight_decay=opt['w_decay'],
            nesterov=opt['nesterov']
        )
    elif opt['name'] == 'Adam':
        optimizer = optim.Adam(
            params=params, 
            lr=opt['lr'], 
            betas=opt['betas'],
            eps=opt['eps'],
            weight_decay=opt['w_decay'],
            amsgrad=opt['amsgrad']
        )
    elif opt['name'] == 'AdamW':
        optimizer = optim.AdamW(
            params=params,
            lr=opt['lr'], 
            betas=opt['betas'],
            eps=opt['eps'],
            weight_decay=opt['w_decay'],
            amsgrad=opt['amsgrad']
        )
    else:
        raise 'Optimizer not available!'
    return optimizer

def train_model(args, model, loaders=None, ema=None, optimizer=None, scheduler=None):
    
    labeled_loader = loaders.get('labeled', None)
    unlabeled_loader = loaders.get('unlabeled', None)
    valid_loader = loaders.get('valid', None)
    
    assert labeled_loader != None
    assert unlabeled_loader != None
    
    last_path = os.path.join(args.cp_path, 'last.pth')
    last_iter, total_iter, last_epoch, total_epoch = load_checkpoints(last_path, model, ema, optimizer, scheduler)        
    
    if last_iter is None:
        last_iter = 0
    if total_iter is None:
        total_iter = len(labeled_loader)
    if last_epoch is None:
        last_epoch = 0
    if total_epoch is None:
        total_epoch = TOTAL_EPOCH
    
    gamma_bank = None
    warpup_epoch = WARMUP_EPOCH
    run_times = 0
    
    for epoch in range(last_epoch, total_epoch):
        current_iter = 0
        for labeled_info, unlabeled_info in zip(labeled_loader, unlabeled_loader):       
            model.train()
            if ema is not None:
                model.train()
            
            current_iter += 1
            run_times += 1
            X_lb = labeled_info['X']
            y_lb = labeled_info['y']
            batch_lb = X_lb.shape[0]
            
            if gamma_bank == None:
                num_classes = y_lb.shape[1]
                gamma_bank = torch.zeros(num_classes)
            
            lb_true = torch.sum((y_lb == 1), dim=0)
            gamma_bank += lb_true
            run_times += 1
            gamma = gamma_bank / (batch_lb * run_times)
            
            if epoch >= warpup_epoch:
                
                thresh_low = gamma.clone()
                thresh_high = 1 - thresh_low
                alpha = neg_log(thresh_high)
                beta = neg_log(thresh_low)

                X_ulb = unlabeled_info['X']
                output_net = model(X_lb, X_ulb, thresh_low, thresh_high)

                loss = compute_batch_loss(output_net, y_lb, unlabeled=True, lambda_u=LAMBDA_U, alpha=alpha, beta=beta)
            else:

                output_net = model(X_lb)
                
                loss = compute_batch_loss(output_net, y_lb)
            
            if ema is not None:
                ema.update()

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if run_times % args.eval_it == 0 and valid_loader != None: 
                # TODO
                # Early stoping and evaluating process
                is_best = False
                save_checkpoints(epoch, current_iter, total_iter, total_epoch, model, ema, optimizer, scheduler, is_best, args.cp_path)

def eval_model(args, model, ema=None):
    model.eval()
    # TODO
    model.train()
    


if __name__ == '__main__':
    # TODO
    # DATASET_INFO[args.dataset]['num_classes']
    args = parse_args()
    backbone = convnextv2_base(20)
    model = BaseNet(backbone)
    ema = EMA(model, beta=0.999)
    loaders = get_loaders(args)
    optimizer = get_optimizer(args, model)
    scheduler = get_lr_scheduler(args, optimizer)
    if args.train:
        train_model(args, model, loaders, ema, optimizer, scheduler)
    
    pass


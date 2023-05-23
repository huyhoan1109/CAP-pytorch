import os
import argparse

from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from MA.EMA import EMA
from datasets import SemiData
from models.CapNet import CapNet
from losses import compute_batch_loss
from backbone.convnext2 import convnextv2_base
from augmentation.transforms import get_pre_transform, get_multi_transform
from utils import save_checkpoints, load_checkpoints, str2bool
from config import CHECKPOINT_PATH, DATASET_INFO, WARMUP_EPOCH, LAMBDA_U, TOTAL_EPOCH, T, SCHEDULER, OPTIMIZER, LAST_MODEL

def parse_args():
    parser = argparse.ArgumentParser(
        description='Semi supervised model',
        epilog='Example: python run.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', type=str, default='voc2012', choices=DATASET_INFO.keys(), help='Using dataset with model')
    parser.add_argument('--train', type=str2bool, default=True, help='Training semi model')
    parser.add_argument('--eval', type=str2bool, default=False, help='Evaluating semi model')
    parser.add_argument('--use-ema', type=str2bool, default=True, help='Using exponential moving average model')
    parser.add_argument('--ema-decay', type=int, default=0.9997, help='Moving average decay')
    parser.add_argument('--resume', type=str2bool, default=True, help='Resume training from checkpoint')
    parser.add_argument('--e-stop', type=int, default=5, help='Use early stoping for model training')
    parser.add_argument('--cp-path', type=str, default=CHECKPOINT_PATH, help='Path to a directory that store checkpoints')
    parser.add_argument('--eval-it', type=int, default=1000, help='Path to a directory that store checkpoints')
    parser.add_argument('--sch', type=int, default=1, choices=SCHEDULER.keys(), help='Choose scheduler type')
    parser.add_argument('--opt', type=int, default=1, choices=OPTIMIZER.keys(), help='Choose optimizer type')    
    parser.add_argument('--use-asl', type=str2bool, default=True, help='Whether to use ASL loss')
    args = parser.parse_args()
    return args

def get_loaders(args):
    dataset = args.dataset
    labeled_dataset = SemiData(
        DATASET_INFO[dataset]['images'], 
        DATASET_INFO[dataset]['meta'], 
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

def train_model(args, loaders, model, ema=None, optimizer=None, scheduler=None):
    
    labeled_loader = loaders.get('labeled', None)
    unlabeled_loader = loaders.get('unlabeled', None)
    valid_loader = loaders.get('valid', None)
    
    assert labeled_loader != None
    assert unlabeled_loader != None
    
    last_path = os.path.join(args.cp_path, LAST_MODEL)
    last_iter, total_iter, last_epoch, total_epoch = load_checkpoints(last_path, model, ema, optimizer, scheduler)        
    
    if last_iter is None:
        last_iter = 0
    if total_iter is None:
        total_iter = len(labeled_loader)
    if last_epoch is None:
        last_epoch = 0
    if total_epoch is None:
        total_epoch = TOTAL_EPOCH
    
    warpup_epoch = WARMUP_EPOCH
    for epoch in range(last_epoch, total_epoch):
        curr_iter = 0
        for lb_batch, ulb_batch in zip(labeled_loader, unlabeled_loader):
            curr_iter += 1     
            # model.train()
            # if ema is not None:
            #     model.train()            
            # if epoch >= warpup_epoch:
            #     model.semi_mode = True
            # loss = compute_batch_loss(args, model, lb_batch, ulb_batch, lambda_u=LAMBDA_U)        
            # if ema is not None:
            #     ema.update()      
            # loss.backward()
            # optimizer.step()
            # scheduler.step()
            # if ((epoch * total_iter + curr_iter) % args.eval_it == 0) and valid_loader != None: 
            #     # TODO
            #     # Early stoping and evaluating process
            #     is_best = False
            #     save_checkpoints(curr_iter, total_iter, epoch, total_epoch, model, ema, optimizer, scheduler, is_best, args.cp_path)

def eval_model(args, model, ema=None):
    model.eval()
    # TODO    


if __name__ == '__main__':
    # TODO
    args = parse_args()
    backbone = convnextv2_base(20)
    num_classes = DATASET_INFO[args.dataset]['num_classes']
    model = CapNet(backbone, num_classes)
    ema = EMA(model, beta=args.ema_decay)
    loaders = get_loaders(args)
    optimizer = get_optimizer(args, model)
    scheduler = get_lr_scheduler(args, optimizer)
    if args.train:
        train_model(args, loaders, model, ema, optimizer, scheduler)    
    pass


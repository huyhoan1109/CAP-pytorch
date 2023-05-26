import os
import argparse
from tqdm import tqdm

from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from metrics.MAP import MAP
from MA.EMA import EMA
from datasets import SemiData
from models.CapNet import CapNet
from losses import compute_batch_loss
from backbone.convnext2 import convnextv2_base
from utils import save_checkpoints, load_checkpoints, str2bool, WandbLogger, AverageMeter
from config import CHECKPOINT_PATH, DATASET_INFO, WARMUP_EPOCH, LAMBDA_U, TOTAL_EPOCH, T, SCHEDULER, OPTIMIZER, LAST_MODEL, MAX_ESTOP

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
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Training device')
    args = parser.parse_args()
    return args

def get_loaders(args):
    dataset = args.dataset
    labeled_dataset = SemiData(
        DATASET_INFO[dataset]['images'], 
        DATASET_INFO[dataset]['meta'], 
        device=args.device
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
        mode='unlabeled',
        device=args.device
    )
    
    unlabeled_loader = DataLoader(
        unlabeled_dataset, 
        64, 
        True,
        drop_last=True,
    )
    
    valid_dataset = SemiData(
        DATASET_INFO[dataset]['images'], 
        DATASET_INFO[dataset]['meta'], 
        mode='valid',
        device=args.device
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
        mode='test',
        device=args.device
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

def train_model(args, logger, trackers, performances, loaders, model, ema=None, optimizer=None, scheduler=None):
    
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
    batch = {}
    best_accuracy = 0
    stop_count = 0
    logger.set_steps()
    for epoch in range(last_epoch, total_epoch):
        curr_iter = 0
        for lb_batch, ulb_batch in enumerate((labeled_loader, unlabeled_loader)):
            curr_iter += 1
            if last_iter >= curr_iter and epoch == last_epoch:
                continue
            
            model.semi_mode = True if epoch >= warpup_epoch else False
            
            batch['lb'] = lb_batch
            batch['ulb'] = ulb_batch
            logger.log({'trainer/global_step': curr_iter + epoch * total_iter})
            loss, _ = compute_batch_loss(args, logger, trackers, performances, batch, model, lambda_u=LAMBDA_U, mode='train')        
            
            ema.update() if ema is not None else None           
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if ((epoch * total_iter + curr_iter) % args.eval_it == 0) and valid_loader != None: 
                is_best = False
                for valid_batch in zip(valid_loader):
                    batch['valid'] = valid_batch
                    loss, accuracy = compute_batch_loss(args, logger, trackers, performances, batch, model, lambda_u=LAMBDA_U, mode='valid')        
                    if accuracy >= best_accuracy:
                        best_accuracy = accuracy
                        is_best = True
                        stop_count = 0
                    else:
                        stop_count += 1
                        if stop_count >= MAX_ESTOP:
                            break
                    save_checkpoints(args.cp_path, curr_iter, total_iter, epoch, total_epoch, model, ema, optimizer, scheduler, is_best)    


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
    
    trackers = {
        'train': {
            'lb_loss': AverageMeter(),
            'ulb_loss': AverageMeter(),
            'cap_loss': AverageMeter(),
            # 'items': {} 
        },
        'val': {
            'loss': AverageMeter(),
            # 'items': {}
        }
    }
    performances = {
        'MAP': MAP()
    }

    logger = WandbLogger(args, '4a4edf57140d746df80b213d934913111fdc8143')
    if args.train:
        train_model(args, logger, trackers, performances, loaders, model, ema, optimizer, scheduler)    
    pass


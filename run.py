import os
import argparse
from tqdm import tqdm

import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from metrics.MAP import MAP
from MA.EMA import EMA
from datasets import SemiData
from models.CapNet import CapNet
from losses import compute_loss_accuracy
from backbone.convnext import convnext_base
from backbone.resnet import ResNet50
from utils import save_checkpoints, load_checkpoints, str2bool, WandbLogger, AverageMeter, get_lr
from config import N_WORKERS, BATCH_SIZE, CHECKPOINT_PATH, DATASET_INFO, WARMUP_EPOCH, LAMBDA_U, TOTAL_EPOCH, TOTAL_ITERS, T, SCHEDULER, OPTIMIZER, LAST_MODEL, MAX_ESTOP, BEST_MODEL

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
    parser.add_argument('--sch', type=int, default=1, choices=SCHEDULER.keys(), help='Choose scheduler type')
    parser.add_argument('--opt', type=int, default=1, choices=OPTIMIZER.keys(), help='Choose optimizer type')    
    parser.add_argument('--device', type=str, default='cuda:0', help='Training device')
    parser.add_argument('--workers', type=int, default=N_WORKERS, help='Number of workers')
    parser.add_argument('--eval-it', type=int, default=250, help='Evaluation iteration')
    args = parser.parse_args()
    return args

def get_loaders(args):
    dataset = args.dataset
    img_path = DATASET_INFO[dataset]['images']
    labeled_dataset = SemiData(
        img_path, 
        DATASET_INFO[dataset]['meta'], 
        device=args.device
    )
    
    labeled_loader = DataLoader(
        labeled_dataset, 
        BATCH_SIZE, 
        True, 
        drop_last=True,
        num_workers=N_WORKERS
    )
    
    unlabeled_dataset = SemiData(
        img_path, 
        DATASET_INFO[dataset]['meta'], 
        mode='unlabeled',
        device=args.device
    )
    
    unlabeled_loader = DataLoader(
        unlabeled_dataset, 
        BATCH_SIZE, 
        True,
        drop_last=True,
        num_workers=N_WORKERS
    )
    
    valid_dataset = SemiData(
        img_path, 
        DATASET_INFO[dataset]['meta'], 
        mode='valid',
        device=args.device
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        BATCH_SIZE, 
        True,
        drop_last=True,
        num_workers=N_WORKERS
    )

    test_dataset = SemiData(
        img_path, 
        DATASET_INFO['voc2012']['meta'], 
        mode='test',
        device=args.device
    )
    
    test_loader = DataLoader(
        test_dataset, 
        BATCH_SIZE, 
        True,
        drop_last=True,
        num_workers=N_WORKERS
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
        scheduler = lr_scheduler.OneCycleLR(optimizer, sch['max_lr'], total_steps=TOTAL_EPOCH * TOTAL_ITERS)
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
            nesterov=opt['nesterov'],
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
    assert valid_loader != None
    
    best_accuracy, _, _, _, _ = load_checkpoints(args.cp_path, model, ema, optimizer, scheduler, load_best=True)
    
    _, last_iter, total_iter, last_epoch, total_epoch = load_checkpoints(args.cp_path, model, ema, optimizer, scheduler, load_best=False)
    

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
    logger.set_steps()
    valid_epoch = 0
    for epoch in range(last_epoch, total_epoch):
        with tqdm(total=total_iter, desc=f"Epoch [{epoch+1}/{total_epoch}]") as t:
            for idx, (lb_batch, ulb_batch) in enumerate(zip(labeled_loader, unlabeled_loader)):
                logger.log({'trainer/global_step': idx + (epoch-last_epoch) * total_iter})
                batch['lb'] = lb_batch
                batch['ulb'] = ulb_batch
                model.train()
                model.semi_mode = True if epoch >= warpup_epoch else False
                train_loss, _ = compute_loss_accuracy(args, logger, trackers, performances, batch, model, lambda_u=LAMBDA_U, mode='train')        
                train_loss.backward()
                optimizer.step()
                scheduler.step()
                ema.update() if ema is not None else None
                logger.log({'train/lr': get_lr(optimizer)})
                if ((epoch * total_iter + idx) % args.eval_it == (args.eval_it - 1)):
                    avg_accuracy = AverageMeter()
                    for valid_idx, valid_batch in enumerate(valid_loader):
                        batch['valid'] = valid_batch
                        model.eval()
                        logger.log({'valid_step': valid_idx + len(valid_loader) * valid_epoch})
                        _, accuracy = compute_loss_accuracy(args, logger, trackers, performances, batch, model, lambda_u=LAMBDA_U, mode='valid')
                        avg_accuracy.update(accuracy)
                    valid_epoch += 1
                    is_best = False
                    save_checkpoints(args.cp_path, avg_accuracy.show(), idx, total_iter, epoch, total_epoch, model, ema, optimizer, scheduler, is_best)    
                    if avg_accuracy.show() < best_accuracy:
                        pass
                    else:
                        is_best = True
                        best_accuracy = avg_accuracy.show()
                        save_checkpoints(args.cp_path, accuracy, idx, total_iter, epoch, total_epoch, model, ema, optimizer, scheduler, is_best)  
                ordered_dict = {
                    'lb_loss': trackers['train']['lb_loss'].show(),
                    'ulb_loss': trackers['train']['ulb_loss'].show(),
                    'cap_loss': trackers['train']['cap_loss'].show(),
                    'main_loss': trackers['train']['main_loss'].show()
                }
                t.set_postfix(ordered_dict=ordered_dict)
                t.update()

if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device(args.device)
    num_classes = DATASET_INFO[args.dataset]['num_classes']
    backbone = ResNet50(num_classes).to(args.device)
    model = CapNet(backbone, num_classes, args.device)
    ema = EMA(model, beta=args.ema_decay).to(args.device)
    loaders = get_loaders(args)
    optimizer = get_optimizer(args, model)
    scheduler = get_lr_scheduler(args, optimizer)
    
    trackers = {
        'train': {
            'lb_loss': AverageMeter(),
            'ulb_loss': AverageMeter(),
            'cap_loss': AverageMeter(),
            'main_loss': AverageMeter()
        },
        'valid': {
            'loss': AverageMeter(),
        }
    }
    performances = {
        'MAP': MAP()
    }

    logger = WandbLogger(args, '4a4edf57140d746df80b213d934913111fdc8143')
    if args.train:
        train_model(args, logger, trackers, performances, loaders, model, ema, optimizer, scheduler)
        # logger.save_checkpoints()
    pass
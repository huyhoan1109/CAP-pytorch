import os
import torch
import wandb
import shutil
import numpy as np
from config import LAST_MODEL, BEST_MODEL, NEG_EPSILON

class WandbLogger():
    def __init__(self, args) -> None:
        self.args = args
        self.logger = wandb
        self.init()

    def init(self):
        proj = self.args.proj
        log_dir = self.args.log_dir
        config = self.args.config
        if self.logger.run is None:
            self.logger.login(anonymous='must')
            os.makedirs(log_dir, exist_ok=True)   
            self.logger.init(
                project = proj,
                dir = log_dir,
                config = config
            )

    def log(self, state):
        self.logger.log(state)

    def save_checkpoints(self):
        print("Uploading checkpoints to wandb ...")
        cp_dir = self.args.cp_dir
        model = self.logger.Artifact(
            'model' + self.logger.run.id, type='model'
        )
        model.add_dir(cp_dir)
        self.logger.log_artifact(model, aliases=["latest"])
    
    def finish(self):
        self.logger.finish()        

def get_lr(optimizer):
    lrs = [param_group['lr'] for param_group in optimizer.param_groups]
    return sum(lrs) / len(lrs)

def save_checkpoints(
    current_epoch,
    current_iter, 
    total_iter, 
    total_epoch, 
    model,
    ema,
    optimizer,
    scheduler,
    is_best,
    checkpoint_path
    ):

    state = {
        'iter': current_iter,
        'total_iter': total_iter,
        'epoch': current_epoch,
        'total_epoch': total_epoch,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }

    print(f"Saving checkpoint...")

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} does not exist!. Create new checkpoint ...")
        os.makedirs(checkpoint_path)
    elif os.path.exists(checkpoint_path) and not is_best:
        print(f"Checkpoint {file_path} does exist!. Override the checkpoint ...")
    
    file_path = os.path.join(checkpoint_path, LAST_MODEL)
    torch.save(state, file_path)
    
    if is_best:
        best_path = os.path.join(checkpoint_path, BEST_MODEL)
        if os.path.isfile(best_path):
            print(f"Checkpoint {best_path} does exist!. Override the checkpoint ...")
        shutil.copyfile(file_path, best_path)

def load_checkpoints(checkpoint, model, ema, optimizer=None, scheduler=None, load_best=False):
    if not os.path.exists(checkpoint):
        print(f"Checkpoint {checkpoint} isn't existed")
        return None, None, None, None
    
    print(f"Load checkpoint from {checkpoint} ...")
    cp = torch.load(checkpoint)
    
    # Load model
    model.load_state_dict(cp['model_state_dict'])  # maybe epoch as well

    if ema is not None and cp['ema_state_dict'] is not None:
        ema.load_state_dict(cp['ema_state_dict'])

    # Load optimizer and scheduler
    if optimizer and 'optimizer' in cp:
        optimizer.load_state_dict(cp['optimizer'])

    if scheduler and 'scheduler' in cp:
        scheduler.load_state_dict(cp['scheduler'])

    last_iter = cp['iter'] if 'iter' in cp else None
    total_iter = cp['total_iter'] if 'total_iter' in cp else None
    last_epoch = cp['epoch'] if 'epoch' in cp else None
    total_epoch = cp['total_epoch'] if 'total_epoch' in cp else None

    return last_iter, total_iter, last_epoch, total_epoch

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
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
    
    def __call__(self):
        return self.avg
    
def str2bool(v):
    """
    Converts string to bool type; enables command line 
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise 'Boolean value expected.'

def load_npy_pkl(file):
    load = np.load(file, allow_pickle=True)
    return load

def load_meta(path, mode):
    if mode not in ('unlabeled', 'labeled', 'valid', 'test'):
        raise 'Load metadata mode not available!'
    if mode == 'unlabeled':
        meta = {
            'images': np.load(os.path.join(path,f'formatted_{mode}_images.npy')),
            'id2cat': load_npy_pkl(os.path.join(path,f'id2cat.npy')),
            'cat2id': load_npy_pkl(os.path.join(path,f'cat2id.npy'))
        }
    else:
        meta = {
            'images': np.load(os.path.join(path,f'formatted_{mode}_images.npy')),
            'labels': np.load(os.path.join(path,f'formatted_{mode}_labels.npy')),
            'id2cat': load_npy_pkl(os.path.join(path,f'id2cat.npy')),
            'cat2id': load_npy_pkl(os.path.join(path,f'id2cat.npy'))
        }
    return meta  

def neg_log(x):
    return - torch.log(x + NEG_EPSILON)
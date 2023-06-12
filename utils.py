import os
import torch
import shutil
import numpy as np
from config import LAST_MODEL, BEST_MODEL, NEG_EPSILON

class WandbLogger():
    def __init__(self, args, key, anonymous='must') -> None:
        self.args = args
        import wandb
        self.logger = wandb
        self.key = key
        self.project = 'CAP'
        self.log_dir = 'logs'
        self.name = 'class-distribution-aware-pseudo-labels-runs'
        self.secret = anonymous
        self.init()

    def init(self):
        if self.logger.run is None:
            self.logger.login(key=self.key, anonymous=self.secret)
            os.makedirs(self.log_dir, exist_ok=True)   
            self.logger.init(
                project = self.project,
                dir = self.log_dir,
                name=self.name
            )

    def log(self, state, commit=True):
        self.logger.log(state, commit=commit)

    def save_checkpoints(self):
        print("Uploading checkpoints to wandb ...")
        cp_dir = self.args.cp_path
        model = self.logger.Artifact(
            'model' + self.logger.run.id, type='model'
        )
        model.add_dir(cp_dir)
        self.logger.log_artifact(model, aliases=["latest"])

    def set_steps(self):
        self.logger.define_metric('trainer/global_step')
        self.logger.define_metric('valid_step')
        self.logger.define_metric('train/*', step_metric='trainer/global_step')
        self.logger.define_metric('val/*', step_metric='valid_step')

    def finish(self):
        self.logger.finish()        

def get_lr(optimizer):
    lrs = [param_group['lr'] for param_group in optimizer.param_groups]
    return sum(lrs) / len(lrs)

def save_checkpoints(
    checkpoint_path,
    accuracy,
    current_iter,
    total_iter,
    current_epoch,
    total_epoch, 
    model,
    ema,
    optimizer,
    scheduler,
    is_best,
    ):

    state = {
        'accuracy': accuracy,
        'iter': current_iter,
        'total_iter': total_iter,
        'epoch': current_epoch,
        'total_epoch': total_epoch,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }

    if not is_best:
        print(f"Saving checkpoint...")

        file_path = os.path.join(checkpoint_path, LAST_MODEL)

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint folder {checkpoint_path} does not exist!. Create new checkpoint ...")
            os.makedirs(checkpoint_path)
        elif os.path.exists(checkpoint_path) and not is_best:
            print(f"Checkpoint {file_path} does exist!. Override the checkpoint ...")
        
        torch.save(state, file_path)
    
    else:
    
        best_path = os.path.join(checkpoint_path, BEST_MODEL)
        if os.path.isfile(best_path):
            print(f"Checkpoint {best_path} does exist!. Override the checkpoint ...")
        torch.save(state, best_path)

def load_checkpoints(checkpoint_path, model, ema, optimizer=None, scheduler=None, load_best=False):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} isn't existed")
        return 0, None, None, None, None
    
    print(f"Load checkpoint from {checkpoint_path} ...")
    if load_best:
        load_path = os.path.join(checkpoint_path, BEST_MODEL)
    else:
        load_path = os.path.join(checkpoint_path, LAST_MODEL)
    try:    
        cp = torch.load(load_path)
        
        # Load model
        model.load_state_dict(cp['model_state_dict'])  # maybe epoch as well

        if ema is not None and cp['ema_state_dict'] is not None:
            ema.load_state_dict(cp['ema_state_dict'])

        # Load optimizer and scheduler
        if optimizer and 'optimizer' in cp:
            optimizer.load_state_dict(cp['optimizer'])

        if scheduler and 'scheduler' in cp:
            scheduler.load_state_dict(cp['scheduler'])


        accuracy = cp['accuracy'] if 'accuracy' in cp else 0
        last_iter = cp['iter'] if 'iter' in cp else None
        total_iter = cp['total_iter'] if 'total_iter' in cp else None
        last_epoch = cp['epoch'] if 'epoch' in cp else None
        total_epoch = cp['total_epoch'] if 'total_epoch' in cp else None

        return accuracy, last_iter, total_iter, last_epoch, total_epoch
    except:
        return 0, None, None, None, None

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0
    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def show(self):
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
    return -torch.log(x+NEG_EPSILON)
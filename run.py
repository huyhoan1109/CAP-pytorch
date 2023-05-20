import os
import argparse
from utils import save_checkpoints, load_checkpoints, str2bool
from config import CHECKPOINT_PATH, DATASET_INFO

def parse_args():
    parser = argparse.ArgumentParser(
        description='Semi supervised model',
        epilog='Example: python run.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # parser.add_argument('--dataset', type=str, default='voc', help='Training semi model')
    parser.add_argument('--train', type=str2bool, default=True, help='Training semi model')
    parser.add_argument('--eval', type=str2bool, default=False, help='Evaluating semi model')
    parser.add_argument('--resume', type=str2bool, default=True, help='Resume training from checkpoint')
    parser.add_argument('--e-stop', type=str2bool, default=True, help='Use early stoping for model training')
    parser.add_argument('--cp-path', type=str, default=CHECKPOINT_PATH, help='Path to a directory that store checkpoints')
    args = parser.parse_args()
    return args

# TODO
def train_model(model, ema, optimizer, scheduler, args):
    model.train()
    last_path = os.path.join(args.cp_path, 'last.pth')
    last_iter, total_iter, last_epoch, total_epoch = load_checkpoints(last_path, model, ema, optimizer, scheduler)

    save_checkpoints()

# TODO
def eval_model(model, args):
    model.eval()

if __name__ == '__main__':
    # TODO
    pass

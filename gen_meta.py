import argparse
from utils import str2bool
from config import LABEL2ID, DATASET_INFO, DATA_PATH
from preproc import format_voc

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generating metadata.',
        epilog='Example: python gen_meta.py --dataset voc2012',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--labeled', type=float, default=0.2, help='Labeled dataset size')
    parser.add_argument('--valid', type=float, default=0.1, help='Valid dataset size')
    parser.add_argument('--test', type=float, default=0.1, help='Test dataset size')
    parser.add_argument('--random-state', type=int, default=43, help='Generating random state')
    parser.add_argument('--dataset', type=str, default='voc2012', help='Dataset to generate metadata')
    args = parser.parse_args()
    return args

# TODO
def generate_meta(args):
    if args.dataset == 'voc2012':
        meta = DATASET_INFO['voc2012']['meta']
        label2id = LABEL2ID['voc2012']
        format_voc.generate_npy(DATA_PATH, meta, label2id, args.labeled, args.valid, args.test, args.random_state)

if __name__ == '__main__':
    args = parse_args()
    generate_meta(args)
    
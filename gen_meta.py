import os
import argparse
import numpy as np
from utils import str2bool, load_npy_pkl
from config import CAT2ID, ID2CAT, DATASET_INFO
from preproc import format_voc, format_cub

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generating metadata.',
        epilog='Example: python gen_meta.py --dataset voc2012',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', type=str, default='voc2012', choices=DATASET_INFO.keys(), help='Dataset to generate metadata')
    parser.add_argument('--labeled', type=float, default=0.2, help='Labeled dataset size')
    parser.add_argument('--valid', type=float, default=0.1, help='Valid dataset size')
    parser.add_argument('--test', type=float, default=0.1, help='Test dataset size')
    parser.add_argument('--random-state', type=int, default=43, help='Generating random state')
    args = parser.parse_args()
    return args

# TODO
def generate_meta(args):
    dataset = args.dataset
    if dataset == 'voc2012':
        meta = DATASET_INFO['voc2012']['meta']
        cat2id = CAT2ID['voc2012']
        id2cat = ID2CAT['voc2012']
        data_path = DATASET_INFO['voc2012']['labels']
        format_voc.generate_npy(data_path, meta, cat2id, id2cat, args)
    elif dataset == 'cub':
        meta = DATASET_INFO['cub']['meta']
        data_path = os.path.join(DATASET_INFO['cub']['root'], 'CUB_200_2011')
        cat2id = {}
        id2cat = {}
        class_path = os.path.join(data_path, 'classes.txt')
        with open(class_path, 'r') as f:
            for line in f:
                cur_line = line.rstrip().split(' ')
                '''
                    In CUB dataset, id starts from 1. 
                    So in order to create label matrix, we must minus id by 1 
                '''
                Id = int(cur_line[0]) - 1
                label = cur_line[-1]
                cat2id[label] = Id
                id2cat[Id] = label
        format_cub.generate_npy(data_path, meta, cat2id, id2cat, args)
    else:
        raise NotImplementedError('Dataset not available!')

if __name__ == '__main__':
    args = parse_args()
    generate_meta(args)
    
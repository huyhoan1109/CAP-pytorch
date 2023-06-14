import os
import argparse
import numpy as np
import pandas as pd
from preproc import format_voc2, format_voc
from utils import str2bool, load_npy_pkl
from config import CAT2ID, ID2CAT, DATASET_INFO

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generating metadata.',
        epilog='Example: python gen_meta.py --dataset voc2012',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', type=str, default='voc2012', choices=DATASET_INFO.keys(), help='Dataset to generate metadata')
    parser.add_argument('--labeled', type=float, default=0.4, help='Labeled dataset size')
    parser.add_argument('--valid', type=float, default=0.1, help='Valid dataset size')
    parser.add_argument('--test', type=float, default=0.1, help='Test dataset size')
    parser.add_argument('--random-state', type=int, default=43, help='Generating random state')
    args = parser.parse_args()
    return args

def generate_meta(args):
    dataset = args.dataset
    meta = DATASET_INFO[dataset]['meta']
    print(f'Generating metadata from dataset {dataset} ...')
    if dataset == 'voc2012':
        cat2id = CAT2ID[dataset]
        split_path = DATASET_INFO[dataset]['split']
        format_voc2.generate_npy(split_path, meta, cat2id, args)
    # elif dataset == 'cub':        
    #     cat2id = {}
    #     data_path = DATASET_INFO[dataset]['root'] + '/CUB_200_2011'
    #     class_path = data_path + '/classes.txt'
    #     with open(class_path, 'r') as f:
    #         for line in f:
    #             curr_line = line.rsplit()
    #             Id = int(curr_line[0]) - 1
    #             label = curr_line[-1]
    #             cat2id[label] = Id
    #     format_cub.generate_npy(data_path, meta, cat2id, args)
    else:
        raise NotImplementedError('Dataset not available!')

if __name__ == '__main__':
    args = parse_args()
    generate_meta(args)
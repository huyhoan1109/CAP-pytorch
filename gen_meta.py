import os
import argparse
import numpy as np
import pandas as pd
from utils import str2bool, load_npy_pkl
from config import CAT2ID, ID2CAT, DATASET_INFO
from preproc import format_voc, format_cub, format_mlrs

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
    meta = DATASET_INFO[dataset]['meta']
    print(f'Generating metadata from dataset {dataset} ...')
    if dataset == 'voc2012':
        cat2id = CAT2ID[dataset]
        split_path = DATASET_INFO[dataset]['split']
        format_voc.generate_npy(split_path, meta, cat2id, args)
    elif dataset == 'cub':        
        cat2id = {}
        data_path = DATASET_INFO[dataset]['root'] + '/CUB_200_2011'
        class_path = data_path + '/classes.txt'
        with open(class_path, 'r') as f:
            for line in f:
                curr_line = line.rsplit()
                Id = int(curr_line[0]) - 1
                label = curr_line[-1]
                cat2id[label] = Id
        format_cub.generate_npy(data_path, meta, cat2id, args)
    elif dataset == 'mlrs':
        split_path = DATASET_INFO[dataset]['root']
        cat_p = os.path.join(split_path, 'Categories_names.xlsx')
        xls = pd.read_excel(cat_p)
        col = xls.columns[0]
        cat2id = {xls[col][k]: k for k in xls[col].keys()}
        format_mlrs.generate_npy(split_path, meta, cat2id, args)  
    else:
        raise NotImplementedError('Dataset not available!')

if __name__ == '__main__':
    args = parse_args()
    generate_meta(args)
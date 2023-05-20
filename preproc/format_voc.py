import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from constants import DATA_PATH, METADATA_PATH, LABEL2ID

def save_voc(path, image_dict, X, mode='labeled'):
    if mode not in ('labeled', 'unlabeled', 'valid', 'test'):
        raise 'Save mode not available!'
    num_images = len(X)
    X.sort()
    label_matrix = np.zeros((num_images, len(LABEL2ID['voc'])))
    for i in range(num_images):
        cur_image = X[i]
        label_indices = np.array(image_dict[cur_image])
        label_matrix[i, label_indices] = 1.0

    if mode == 'unlabeled':
        np.save(os.path.join(path, 'formatted_unlabeled_images.npy'), X)
    else:
        np.save(os.path.join(path, f'formatted_{mode}_images.npy'), X)
        np.save(os.path.join(path, f'formatted_{mode}_labels.npy'), label_matrix)

def checksize(size, mode='labeled'):
    if size < 0 or size > 1:
        raise f'Size of {mode} dataset must be between 0 and 1!'

def generate_voc_npy(args):
    image_dict = {}
    data = {
        'labeled': [],
        'unlabeled': [],
        'val': [], 
        'test': []
    }
    checksize(args.labeled)
    checksize(args.valid, 'valid')
    checksize(args.test, 'test')
    for cat in LABEL2ID['voc']:
        image_list = []
        labels = []
        num_images = 0
        with open(os.path.join(args.load_dir, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main', f"{cat}_trainval.txt"), 'r') as f:
            for line in f:
                num_images += 1
                cur_line = line.rstrip().split(' ')
                img_name, label = cur_line[0], int(cur_line[-1])
                f_img = img_name + '.jpg'
                image_list.append(f_img)
                labels.append(label)
                if label == 1:
                    if f_img not in image_dict:
                        image_dict[f_img] = []
                    image_dict[f_img].append(LABEL2ID['voc'][cat])
        X_t, X_valid, y_t, _ = train_test_split(image_list, labels, test_size=args.valid, random_state=args.random_state)
        X_train, X_test, y_train, _ = train_test_split(X_t, y_t,test_size=args.test, random_state=args.random_state)
        X_unlabeled, X_labeled, _ , _ = train_test_split(X_train, y_train, test_size=args.labeled, random_state=args.random_state)
        data['unlabeled'].extend(X_unlabeled)
        data['labeled'].extend(X_labeled)
        data['val'].extend(X_valid)
        data['test'].extend(X_test)

    X_ulb, X_lb_tr, X_v, X_t = data['unlabeled'], data['labeled'], data['val'], data['test']
    save_voc(args.save_dir, image_dict, X_lb_tr)
    save_voc(args.save_dir, image_dict, X_ulb, mode='unlabeled')
    save_voc(args.save_dir, image_dict, X_v, mode='valid')
    save_voc(args.save_dir, image_dict, X_t, mode='test')
        
def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize PASCAL VOC dataset.',
        epilog='Example: python format_voc.py --save-dir ../data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--random-state', type=float, default=42, help='random state to split dataset')
    parser.add_argument('--load-dir', type=str, default=DATA_PATH, help='path to a directory containing dataset')
    parser.add_argument('--save-dir', type=str, default=METADATA_PATH+'/voc2012', help='path to a directory to save metadata')
    parser.add_argument('--labeled', type=float, default=0.2, help='size of labeled dataset')
    parser.add_argument('--valid', type=float, default=0.1, help='size of valid dataset')
    parser.add_argument('--test', type=float, default=0.1, help='size of test dataset')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    generate_voc_npy(args)

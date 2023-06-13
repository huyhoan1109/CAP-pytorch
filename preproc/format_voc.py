import os
import numpy as np
from sklearn.model_selection import train_test_split

def save_voc(X, cat2id, image_dict, path, mode='labeled'):
    if mode not in ('labeled', 'unlabeled', 'valid', 'test'):
        raise 'Save mode not available!'

    X.sort()
    num_images = len(X)
    label_matrix = np.zeros((num_images, len(cat2id)))

    for i in range(num_images):
        cur_image = X[i]
        label_indices = np.array(image_dict[cur_image])
        label_matrix[i, label_indices] = 1.0

    img_path = os.path.join(path, f'formatted_{mode}_images.npy')
    np.save(img_path, X)
    print(f'Generated {img_path}')

    if mode != 'unlabeled':
        label_path = os.path.join(path, f'formatted_{mode}_labels.npy')
        np.save(label_path, label_matrix)
        print(f'Generated {label_path}')

def checksize(size, mode='labeled'):
    if size < 0 or size > 1:
        raise f'Size of {mode} dataset must be between 0 and 1!'

def generate_npy(split_path, meta_path, cat2id, args):
    """
    Parameters:
    """

    os.makedirs(meta_path, exist_ok=True)
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

    #NEED TO CHANGE

    for cat in cat2id:
        image_list = []
        labels = []
        num_images = 0
        with open(os.path.join(split_path, f"{cat}_trainval.txt"), 'r') as f:
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
                    image_dict[f_img].append(cat2id[cat])
        valid_t = args.valid
        X_t, X_valid, y_t, _ = train_test_split(image_list, labels, test_size=valid_t, random_state=args.random_state)
        test_t = args.test / (1 - args.valid) 
        X_train, X_test, y_train, _ = train_test_split(X_t, y_t,test_size=test_t, random_state=args.random_state)
        label_t = args.labeled / (1 - test_t)
        X_unlabeled, X_labeled, _ , _ = train_test_split(X_train, y_train, test_size=label_t, random_state=args.random_state)
        data['unlabeled'].extend(X_unlabeled)
        data['labeled'].extend(X_labeled)
        data['val'].extend(X_valid)
        data['test'].extend(X_test)

    X_ulb, X_lb_tr, X_v, X_t = data['unlabeled'], data['labeled'], data['val'], data['test']

    save_voc(X_lb_tr, cat2id, image_dict, meta_path)
    save_voc(X_ulb, cat2id, image_dict, meta_path, mode='unlabeled')
    save_voc(X_v, cat2id, image_dict, meta_path, mode='valid')
    save_voc(X_t, cat2id, image_dict, meta_path, mode='test')
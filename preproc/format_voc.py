import os
import numpy as np
import pickle as pk
from sklearn.model_selection import train_test_split

def save_voc(X, label2id, image_dict, path, mode='labeled'):
    if mode not in ('labeled', 'unlabeled', 'valid', 'test'):
        raise 'Save mode not available!'
    
    X.sort()
    num_images = len(X)
    label_matrix = np.zeros((num_images, len(label2id)))
    
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

def generate_npy(data_path, meta_path, label2id, id2label, args):
    
    """
    Parameters:
    
        data_path : str
            Path to a directory store data
        meta_path : str
            Path to a directory store metadata
        label2id : dict
            Dictionary that help mapping label to id
        id2label : dict
            Dictionary that help mapping id to label
        args: 
            Contain belows parameters
            
            labeled_size : float
                Size of labeled dataset (in the interval [0, 1))
            valid_size : float
                Size of valid dataset (in the interval [0, 1))
            test_size : float
                Size of test dataset (in the interval [0, 1))
            random_state : int | None
                Random state Instance
        
    """

    print('Generating metadata from dataset ...')
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
    for cat in label2id:
        image_list = []
        labels = []
        num_images = 0
        with open(os.path.join(data_path, f"{cat}_trainval.txt"), 'r') as f:
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
                    image_dict[f_img].append(label2id[cat])
        X_t, X_valid, y_t, _ = train_test_split(image_list, labels, test_size=args.valid, random_state=args.random_state)
        X_train, X_test, y_train, _ = train_test_split(X_t, y_t,test_size=args.test, random_state=args.random_state)
        X_unlabeled, X_labeled, _ , _ = train_test_split(X_train, y_train, test_size=args.labeled, random_state=args.random_state)
        data['unlabeled'].extend(X_unlabeled)
        data['labeled'].extend(X_labeled)
        data['val'].extend(X_valid)
        data['test'].extend(X_test)

    X_ulb, X_lb_tr, X_v, X_t = data['unlabeled'], data['labeled'], data['val'], data['test']
    save_voc(X_lb_tr, label2id, image_dict, meta_path)
    save_voc(X_ulb, label2id, image_dict, meta_path, mode='unlabeled')
    save_voc(X_v, label2id, image_dict, meta_path, mode='valid')
    save_voc(X_t, label2id, image_dict, meta_path, mode='test')
    np.save(meta_path + '/label2id.npy', label2id)
    np.save(meta_path + '/id2label.npy', id2label)
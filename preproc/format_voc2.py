import os
import numpy as np
from sklearn.model_selection import train_test_split

def generate_npy(split_path, meta_path, cat2id, args):
    """
    Parameters:
    
        split_path : str
            Path to a directory store data
        meta_path : str
            Path to a directory store metadata
        cat2id : dict
            Dictionary that help mapping label to id
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

    os.makedirs(meta_path, exist_ok=True)

    data = {
        'labeled': [],
        'unlabeled': [],
        'val': [], 
        'test': []
    }

    #NEED TO CHANGE

    ann_dict = {}
    image_list = {'train': [], 'val': []}

    for phase in ['train', 'val']:
        for cat in cat2id:
            with open(os.path.join(split_path, f"{cat}_{phase}.txt"), 'r') as f:
                for line in f:
                    cur_line = line.rstrip().split(' ')
                    image_id = cur_line[0]
                    label = cur_line[-1]
                    image_fname = image_id + '.jpg'
                    if int(label) == 1:
                        if image_fname not in ann_dict:
                            ann_dict[image_fname] = []
                            image_list[phase].append(image_fname)
                        ann_dict[image_fname].append(cat2id[cat])

        image_list[phase].sort()
        num_images = len(image_list[phase])
        label_matrix = np.zeros((num_images, len(cat2id)))
        for i in range(num_images):
            cur_image = image_list[phase][i]
            label_indices = np.array(ann_dict[cur_image])
            label_matrix[i, label_indices] = 1.0
        
        if phase == 'train':
            X_labeled, X_valid, y_labeled, y_valid = train_test_split(np.array(image_list[phase]), np.array(label_matrix), train_size=args.labeled/0.5)
            np.save(os.path.join(meta_path, 'formatted_labeled_images.npy'), X_labeled)
            np.save(os.path.join(meta_path, 'formatted_valid_images.npy'), X_valid)
            np.save(os.path.join(meta_path, 'formatted_labeled_labels.npy'), y_labeled)
            np.save(os.path.join(meta_path, 'formatted_valid_labels.npy'), y_valid)
        else:
            X_unlabeled, X_test, _, y_test = train_test_split(np.array(image_list[phase]), np.array(label_matrix), train_size=args.labeled/0.5)
            np.save(os.path.join(meta_path, 'formatted_unlabeled_images.npy'), X_unlabeled)
            np.save(os.path.join(meta_path, 'formatted_test_images.npy'), X_test)
            np.save(os.path.join(meta_path, 'formatted_test_labels.npy'), y_test)
    
    cat2id_f = meta_path + '/cat2id.npy'
    id2cat_f = meta_path + '/id2cat.npy'
    np.save(cat2id_f, cat2id)
    print(f'Generated {cat2id_f}')
    id2cat = {cat2id[k] for k in cat2id.keys()}
    np.save(id2cat_f, id2cat)
    print(f'Generated {id2cat_f}')
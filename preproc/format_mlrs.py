import os
import numpy as np

def save_mlrs():
    pass

def generate_npy(split_path, meta_path, cat2id, args):
    """
    Parameters:
    
        split_path : str
            Path or directory store train and test split
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
    
    cat2id_f = meta_path + '/cat2id.npy'
    id2cat_f = meta_path + '/id2cat.npy'
    
    np.save(cat2id_f, cat2id)
    print(f'Generated {cat2id_f}')
    id2cat = {cat2id[k] for k in cat2id.keys()}
    np.save(id2cat_f, id2cat)
    print(f'Generated {id2cat_f}')
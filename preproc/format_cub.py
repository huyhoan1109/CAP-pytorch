import os

#TODO
def generate_npy(data_path, meta_path, cat2id, id2cat, args):
    # create one hot label matrix for dataset
    # args.labeled, args.valid, args.test, args.random_state 

    """
    Parameters:
    
        data_path : str
            Path to a directory store data
        meta_path : str
            Path to a directory store metadata
        cat2id : dict
            Dictionary that help mapping label to id
        id2cat : dict
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
    print(data_path)
    dataset_file = os.path.join(data_path, 'train_test_split.txt')
    pass

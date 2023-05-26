# import os
# import numpy as np
# from sklearn.model_selection import train_test_split

# def save_cub(X, y, path, mode='labeled'):
#     img_path = os.path.join(path, f'formatted_{mode}_images.npy')
#     np.save(img_path, X)
#     print(f'Generated {img_path}')
    
#     if mode != 'unlabeled':
#         label_path = os.path.join(path, f'formatted_{mode}_labels.npy')
#         np.save(label_path, y)
#         print(f'Generated {label_path}')

# #TODO
# def generate_npy(data_path, meta_path, cat2id, args):
#     # create one hot label matrix for dataset
#     # args.labeled, args.valid, args.test, args.random_state 

#     """
#     Parameters:
    
#         data_path : str
#             Path or directory store data to process
#         meta_path : str
#             Path to a directory store metadata
#         cat2id : dict
#             Dictionary that help mapping label to id
#         args: 
#             Contain belows parameters
            
#             labeled_size : float
#                 Size of labeled dataset (in the interval [0, 1))
#             valid_size : float
#                 Size of valid dataset (in the interval [0, 1))
#             test_size : float
#                 Size of test dataset (in the interval [0, 1))
#             random_state : int | None
#                 Random state Instance
        
#     """

#     os.makedirs(meta_path, exist_ok=True)
    
#     store = {
#         'X': [],
#         'y': []
#     }

#     img_id2name = {}
#     cub_dict = {}

#     with open(data_path+'/images.txt') as f:
#         for line in f:
#             curr_line = line.rsplit()
#             img_id = int(curr_line[0]) - 1
#             img_name = curr_line[-1]
#             img_id2name[img_id] = img_name

#     with open(data_path+'/image_class_labels.txt') as f:
#         for line in f:
#             curr_line = line.rsplit()
#             img_id = int(curr_line[0]) - 1
#             label_id = int(curr_line[-1]) - 1
#             if img_id not in cub_dict:
#                 cub_dict[img_id] = [label_id]
#             else:
#                 cub_dict[img_id].append(label_id)

#     for img_id in cub_dict:
#         label_mtx = np.zeros(len(cat2id))
#         label_indices = cub_dict[img_id]
#         label_mtx[label_indices] = 1.0
#         store['X'].append(img_id2name[img_id])
#         store['y'].append(label_mtx)

    
#     valid_t = args.valid
#     X_t, X_valid, y_t, y_valid = train_test_split(store['X'], store['y'], test_size=valid_t, random_state=args.random_state)
#     test_t = args.test / (1 - args.valid)
#     X_train, X_test, y_train, y_test = train_test_split(X_t, y_t,test_size=test_t, random_state=args.random_state)
#     label_t = args.labeled / (1 - test_t)
#     X_ulb, X_lb, _ , y_lb = train_test_split(X_train, y_train, test_size=label_t, random_state=args.random_state)

#     save_cub(X_valid, y_valid, meta_path, 'valid')
#     save_cub(X_test, y_test, meta_path, 'test')
#     save_cub(X_lb, y_lb, meta_path, 'labeled')
#     save_cub(X_ulb, None, meta_path, 'unlabeled')

#     cat2id_f = meta_path + '/cat2id.npy'
#     id2cat_f = meta_path + '/id2cat.npy'
    
#     np.save(cat2id_f, cat2id)
#     print(f'Generated {cat2id_f}')
#     id2cat = {cat2id[k] for k in cat2id.keys()}
#     np.save(id2cat_f, id2cat)
#     print(f'Generated {id2cat_f}')

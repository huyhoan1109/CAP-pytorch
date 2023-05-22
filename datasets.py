import os
import torch
import numpy as np
from PIL import Image
from utils import load_meta
from torch.utils.data import Dataset

class SemiData(Dataset):
    def __init__(self, img_path, meta_path, mode='labeled', pre_transform=None, multi_transform=None, device='cpu'):
        '''
        Args:
        img_path: str
            Path to image dicrectory
        meta_path: str
            Path to generated metadata directory
        transforms: transform.Compose
            Image transforming method
        mode: str
            Data loading method ('unlabeled', 'labeled', 'valid', 'test')
        device: str
            Device to load tensor ('cpu', 'cuda')

        SemiData will load metadata which contains:

        images: list(str)
            path to images
        labels: np.array 
            one hot vectors (only load in 'labeled', 'val', 'test' mode) 
        cat2id: dict
            dictionary mapping category to id 
        id2cat: dict
            dictionary mapping id to category 
        '''
        super().__init__()
        self.img_path = img_path
        self.meta_path = meta_path
        self.pre_transform = pre_transform
        self.multi_transform = multi_transform
        self.mode = mode
        self.metadata = self._load_meta()
        self.device = device
                    
    def _load_meta(self):
        return load_meta(self.meta_path, self.mode)
    
    def __len__(self):
        return len(self.metadata['images'])

    def __getitem__(self, idx):
        img_p = os.path.join(self.img_path, self.metadata['images'][idx])
        with Image.open(img_p) as img:
            img = self.pre_transform(img.convert('RGB'))
        
        if self.mode == 'unlabeled':
            # dim_y = (len(img), len(self.metadata['cat2id']))
            w_aug = torch.FloatTensor(np.array(self.multi_transform['weak'](img)))
            s_aug = torch.FloatTensor(np.array(self.multi_transform['strong'](img)))
            out = {
                'idx': idx,
                'X': (w_aug, s_aug)
            } 
        else:
            # label is a one hot vector
            label = np.copy(self.metadata['labels'][idx, :])
            out = {
                'idx': idx,
                'X': torch.FloatTensor(np.array(img)).to(self.device),
                'y': torch.FloatTensor(label).to(self.device)
            }   
        
        return out
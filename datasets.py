import os
import torch
import numpy as np
from PIL import Image
from utils import load_meta
from torch.utils.data import Dataset

class SemiData(Dataset):
    def __init__(self, img_path, meta_path, transform=None, mode='labeled', device='cpu'):
        '''
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
        self.transform = transform
        self.metadata = self._load_meta()
        self.mode = mode
        self.device = device
                    
    def _load_meta(self):
        return load_meta(self.meta_path, self.mode)
    
    def __len__(self):
        return len(self.metadata['images'])

    def __getitem__(self, idx):
        img_p = os.path.join(self.img_path, self.metadata['images'][idx])
        
        with Image.open(img_p) as img:
            img = np.copy(img.convert('RGB'))
        
        if self.mode == 'unlabeled':
            dim_y = (len(img), len(self.metadata['cat2id']))
            out = {
                'idx': idx,
                'X': torch.FloatTensor(img).to(self.device),
                'y': torch.zeros(dim_y).float().to(self.device),
            } 
        else:
            # label is a one hot vector
            label = np.copy(self.metadata['labels'][idx, :])
            out = {
                'idx': idx,
                'X': torch.FloatTensor(img).to(self.device),
                'y': torch.FloatTensor(label).to(self.device)
            }   
        
        return out
import os
import torch
import numpy as np
from PIL import Image
from utils import load_meta
from torch.utils.data import Dataset

class SemiData(Dataset):
    def __init__(self, root, meta_path, transform=None, mode='labeled', device='cpu'):
        super().__init__()
        self.root = root
        self.mode = mode
        self.transform = transform
        self.meta_path = meta_path
        self.metadata = self._load_meta()
        self.device = device
                    
    def _load_meta(self):
        return load_meta(self.meta_path, self.mode)
    
    def __len__(self):
        return len(self.metadata['images'])

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.metadata['images'][idx])
        
        with Image.open(img_path) as img:
            img = np.copy(img.convert('RGB'))
        if self.mode == 'unlabeled':
            out = {
                'idx': idx,
                'X': torch.FloatTensor(img).to(self.device),
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
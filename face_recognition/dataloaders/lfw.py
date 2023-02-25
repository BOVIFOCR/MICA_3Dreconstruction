import os, sys
import numpy as np
from glob import glob

import torch
import torch.utils.data

# source: https://www.learnpytorch.io/04_pytorch_custom_datasets/#52-create-a-custom-dataset-to-replicate-imagefolder
class LFW_3DMM(torch.utils.data.Dataset):

    def __init__(self, root_dir='', file_ext='identity.npy', device='cuda:0', transform=None):
        self.root_dir = root_dir
        self.file_ext = file_ext
        self.device = device
        self.transform = transform
        self.paths = self.load_paths(self.root_dir, self.file_ext)
        self.num_samples = len(self.paths)
        self.classes = sorted(list(set([self._class_name_from_path_sample(path) for path in self.paths])))
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
    
    def _class_name_from_path_sample(self, path=''):
        class_name = path.split('/')[-3]
        return class_name

    def load_paths(self, root_dir='', file_ext=''):
        paths = sorted(glob(root_dir + '/*/*/*'+file_ext))     # LFW is organized as: /path/to/lfw/Aaron_Eckhart/Aaron_Eckhart_0001/identity.npy
        # print('LFW_3DMM - load_paths - paths:', paths)
        return paths

    def load_sample(self, index: int):
        sample_path = self.paths[index]
        sample_data = np.load(sample_path, allow_pickle=True)
        return sample_data

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index: int):
        sample_data = torch.from_numpy(self.load_sample(index)).to(self.device)
        # class_name  = self.paths[index].parent.name    # expects path in data_folder/class_name/image.jpeg
        class_name  = self._class_name_from_path_sample(self.paths[index])    # expects path in data_folder/class_name/image.jpeg
        # print('class_name:', class_name)
        class_idx = torch.tensor(self.class_to_idx[class_name]).to(self.device)
        # print('class_name:', class_name)
        # print('class_idx:', class_idx)
        # print('---------------')
        return sample_data, class_idx # return data, label (X, y)
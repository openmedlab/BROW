import pandas as pd
import openslide
import numpy as np


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



def build_transform():    
    mean,std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    return transforms.Compose([
            # RandomResizedCrop(args.input_size, interpolation=3),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)],
        )
    
# 定义子数据集类
class SubDataset(Dataset):
    def __init__(self, slidedir):
        ext = slidedir.split('.')[-1]
        len_ext = len(ext)+1
        coords_path = slidedir[:-len_ext] + '.npy'
        self.coords_arr = np.load(coords_path)
        self.len_coords = self.coords_arr.shape[0]
        self.slidedir = slidedir
        self.trans = build_transform()
    def __len__(self):
        return self.len_coords

    def __getitem__(self, index):
        coors = self.coords_arr[index]
        patch = openslide.open_slide(self.slidedir).read_region(coors, 0, (256,256)).convert('RGB')
        patch = self.trans(patch)
        return patch

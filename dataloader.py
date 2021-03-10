#!/usr/bin/python3
# coding=utf-8

import os
import sys
import openslide
from PIL import Image
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import pandas as pd
import openslide

import config

class Dataset_Online_WSI(data.Dataset):
    def __init__(self, stage, test_split=0, splits=4):
        super(Dataset_Online_WSI, self).__init__()
        self.folder = config.slide_all
        self.mask_folder = config.mask_folder_all
        
        df_all = pd.read_csv(config.csv_all)
        split_num_all = int(len(df_all)/splits)

        if stage in ['test', 'Test']:
            self.df = df_all[test_split*split_num_all : (test_split+1)*split_num_all]
            print('Test set in {:d}/{:d}, with {:d} samples'.format(test_split, splits, len(self.df)))
        elif stage in ['train', 'Train']:
            self.df = df_all.drop(labels=list(range(test_split*split_num_all, (test_split+1)*split_num_all)))
            print('Train set in {:d}/{:d}, with {:d} samples'.format(test_split, splits, len(self.df)))
        else:
            print('Error stage!')
        print('load df successfully')

        self.wsi_names = self.df['img_name'] # 
        self.targets = self.df['score']
    
    def __getitem__(self, index):
        foremask = np.expand_dims(
                np.load(os.path.join(self.mask_folder, self.wsi_names[index][:-5]+'.npy')), 
            axis=2)
        foremask = np.transpose(foremask, (2, 0, 1))

        target = int(self.targets[index])
        if target > 0.5:
            target = target-1    

        wsi_filepath = os.path.join(self.folder, self.wsi_names[index])
        wsi_handle = openslide.OpenSlide(wsi_filepath)
        npy_slide = np.array(
            wsi_handle.read_region(
                location=(0, 0), 
                level=3, 
                size=wsi_handle.level_dimensions[3])
        )[:, :, :3]
        # cast to float
        npy_slide = npy_slide.astype(np.float32) / 255.0
        npy_slide = np.transpose(npy_slide, (2, 0, 1))

        return wsi_filepath, target, npy_slide, self.wsi_names[index][:-5], foremask

    def __len__(self):
        return len(self.targets)


def get_dataloader(stage, test_split, wsi_shuffle, num_workers=1):
    """
    for train: 
        get_dataloader(stage='train', wsi_shuffle=True, augmentations=True)
    for test :
        get_dataloader(stage='test', wsi_shuffle=False, augmentations=False)
    """
    if stage in ['train', 'Train']:
        print('build WSI train dataset with {:d} num_workers, batch size 1.'.format(num_workers))
        dataset = Dataset_Online_WSI(
            stage='train', test_split=test_split
            )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=wsi_shuffle, num_workers=num_workers, pin_memory=False)
        
    elif stage in ['test', 'Test']:
        print('build WSI test dataset with {:d} num_workers, batch size 1.'.format(num_workers))
        dataset = Dataset_Online_WSI(
            stage='test', test_split=test_split
            )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=wsi_shuffle, num_workers=num_workers, pin_memory=False)
        
    return dataloader

if __name__ == '__main__':
    test_loader = get_dataloader(stage='test', test_split=0, wsi_shuffle=True)
    train_loader = get_dataloader(stage='train', test_split=0, wsi_shuffle=True)
    test_loader = get_dataloader(stage='test', test_split=1, wsi_shuffle=True)
    train_loader = get_dataloader(stage='train', test_split=1, wsi_shuffle=True)
    test_loader = get_dataloader(stage='test', test_split=2, wsi_shuffle=True)
    train_loader = get_dataloader(stage='train', test_split=2, wsi_shuffle=True)
    test_loader = get_dataloader(stage='test', test_split=3, wsi_shuffle=True)
    train_loader = get_dataloader(stage='train', test_split=3, wsi_shuffle=True)

    for i, (wsi_filepath, target, npy_slide, wsi_name) in enumerate(train_loader):
        if i > 10:
            break
        print(i, wsi_filepath, target, wsi_name)
    
    print(len(test_loader))
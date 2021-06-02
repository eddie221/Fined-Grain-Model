#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 17:24:29 2021

@author: mmplab603
"""

import torchvision.transforms as transforms
from PIL import Image
from config_seg import IMAGE_SIZE, BATCH_SIZE, KFOLD
if KFOLD == 1:
    from config_seg import VAL_SPLIT
import torch
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from voc12_loader import voc12Dataset
from data_transform import *

data_transforms = {
        'train': Compose([
            Resize((300, 300)),
            RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
            RandomFlip(),
        ]),
        'val': Compose([
            Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ]),
}

def load_voc12(path):
    dataloader = []
    dataset_sizes = []
    trainset = voc12Dataset(path, "train", data_transforms['train'])
    #trainset[torch.randint(0, 1000, (1,))]
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size = BATCH_SIZE,
                                              shuffle = True,
                                              num_workers = 4)
    
    valset = voc12Dataset(path, "val", data_transforms['val'])
    valloader = torch.utils.data.DataLoader(valset,
                                            batch_size = BATCH_SIZE,
                                            shuffle = True,
                                            num_workers = 4)
    dataloader.append({'train' : trainloader, 'val' : valloader})
    dataset_sizes.append({'train' : len(trainloader), 'val' : len(valloader)})
    return dataloader, None
    
if __name__ == "__main__":
    load_voc12("/home/mmplab603/program/datasets/VOCdevkit/VOC2012/SegmentationClassAug/")
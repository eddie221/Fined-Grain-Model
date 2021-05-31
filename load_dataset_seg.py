#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 17:24:29 2021

@author: mmplab603
"""

import torchvision.transforms as transforms
import torchvision
from PIL import Image
from config_seg import IMAGE_SIZE, BATCH_SIZE, KFOLD
if KFOLD == 1:
    from config_seg import VAL_SPLIT
import torch
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from voc12_loader import voc12Dataset

data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((300, 300), Image.BILINEAR),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
}

def load_voc12(path):
    dataloader = []
    dataset_sizes = []
    trainset = voc12Dataset(path, "train", data_transforms['train'])
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
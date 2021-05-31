#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 11:54:24 2021

@author: mmplab603
"""

from torch.utils.data.dataset import Dataset
import glob
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import os

class voc12Dataset(Dataset):
    def __init__(self, dataset_path, phase, image_transform):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.dataset_path = dataset_path
        self.root = dataset_path
        self.image_transform = image_transform
        self.gt_transform = transforms.Compose(self.image_transform.transforms[:-2])
        self.image_pool = glob.glob("{}/{}/*".format(dataset_path, phase))  
        assert len(self.image_pool) != 0, "Can't find image in {}".format("{}/{}/*".format(dataset_path, phase))
        
    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        image_path = self.image_pool[index]
        image_name = image_path.split('/')[-1][:-4]
        image = Image.open(image_path)
        ground_truth_path = os.path.join(self.root, "ground_truth/{}.png".format(image_name))
        gt = np.array(Image.open(ground_truth_path))
        gt[gt > 25] = 0
        image = self.image_transform(image)
        gt = self.gt_transform(Image.fromarray(gt))
        
        gt = torch.as_tensor(np.array(gt), dtype=torch.int64)
        
        return image, gt    
        
    
    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.image_pool)
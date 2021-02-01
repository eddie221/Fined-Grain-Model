#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:54:34 2020

@author: mmplab603
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
from neural_network.classify import resnet50
from PIL import Image
import glob
import csv
import matplotlib.pyplot as plt
import torchvision
import numpy as np

class ISIC_dataset(Dataset):
    def __init__(self, root, transform):
        self.trans = transform
        self.image_path = glob.glob(os.path.join(root, '*.jpg'))
        
    def __getitem__(self, idx):
        img = Image.open(self.image_path[idx])
        img = self.trans(img)
        img_path = self.image_path[idx]
        img_name = img_path.split('/')[-1][:-4]
        return img, img_name
        
    def __len__(self):
        return len(self.image_path)

IMAGE_SIZE = 448
NUM_CLASS = 9

use_gpu = torch.cuda.is_available()

#image_path = glob.glob('./ISIC_2019_Test_Input/*.jpg')  
#image_path = glob.glob('./test/*.jpg')  

trans = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)), 
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

all_image_datasets = torchvision.datasets.ImageFolder('../datasets/ISIC 2019/', trans)
dataset = torch.utils.data.DataLoader(all_image_datasets, batch_size = 2, shuffle = True, num_workers = 2)    
# =============================================================================
# image_folder = ISIC_dataset('./ISIC_2019_Test_Input/', trans)
# 
# dataset = torch.utils.data.DataLoader(image_folder, batch_size = 1, shuffle=False)
# =============================================================================
#print(len(dataset))

TITLE = ['image', 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
# 4 5 1 0 2 3 8 7 6 

CSV_DATA = [TITLE]
model_predict = []
def train_step(model, dataset):
    correct = 0
    total = 0
    for step, (data, path) in enumerate(dataset):
        if use_gpu:
            b_data = data.cuda()
        else:
            b_data = data
        output = model(b_data)
        output = torch.nn.functional.softmax(output.data, dim = 1)
        output_data = output.data
        csv_d = [path, output_data[0][4].item(), output_data[0][5].item(), output_data[0][1].item(), output_data[0][0].item(), output_data[0][2].item(), output_data[0][3].item(), output_data[0][8].item(), output_data[0][7].item(), output_data[0][6].item()]
        CSV_DATA.append(csv_d)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted.cpu() == path).sum().item()
        total += data.size(0)
        print(correct / total)
        #print("\r Step : {} prediction : {}".format(step, predicted), end='')
        #print(predicted)
    #return predicted.data    


model = resnet50(num_classes = NUM_CLASS).cuda()
model.train(False)
for param_path in glob.glob('./pkl/fold_0_epoch_3_20210103-*.pkl'):
    params = torch.load(param_path)
    load = []
    not_load = []
    for name, param in params.items():
        if name in model.state_dict():
            try:
                model.state_dict()[name].copy_(param)
                load.append(name)
            except:
                not_load.append(name)
        else:
            not_load.append(name)
    print("Load {} layers".format(len(load)))
    print("Not load {} layers".format(len(not_load)))
    print(not_load)
    train_step(model, dataset)


# =============================================================================
# with open('output.csv', 'w', newline='') as csvfile:
#     for row_data in CSV_DATA:
#         writer = csv.writer(csvfile)
#         writer.writerow(row_data)
# =============================================================================


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 09:54:05 2020

@author: eddie
"""

import neural_network.model as model_net
import torchvision.transforms as transforms
import torchvision
import torch
#import args
import os
import random
from PIL import Image
import numpy as np
import time
from config import BATCH_SIZE, IMAGE_SIZE, LR, NUM_CLASS, INDEX, EPOCH, REMAEK, CON_MATRIX
#from torch.utils.tensorboard import SummaryWriter

#print environment information
print(torch.cuda.is_available())
DEVICE = 'cuda:0'

#writer = SummaryWriter('../tensorflow/logs/cub_{}'.format(INDEX), comment = "224_64")

use_gpu = torch.cuda.is_available()

optimizer_select = ''
loss_function_select = ''
model_name = ''
data_dir = '../COFENet/skin/ISIC 2019/'

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Random2DTranslation(object):
    
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
    - height (int): target image height.
    - width (int): target image width.
    - p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
        - img (PIL Image): Image to be cropped.
        """
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)
        
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        #new_width, new_height = 512, 512
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img
    
data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((600, 600), Image.BILINEAR),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            
        ]),
        'val': transforms.Compose([
            transforms.Resize((600, 600), Image.BILINEAR),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
}
    


def load_data():
    image_datasets = {x : torchvision.datasets.ImageFolder(os.path.join(data_dir, x),
                                                           data_transforms[x]) 
                        for x in ['train', 'val']}
    
    image_dataloader = {x : torch.utils.data.DataLoader(image_datasets[x],
                                                        batch_size=BATCH_SIZE,
                                                        #sampler = data_sampler[x],
                                                        shuffle=True,
                                                        num_workers=16)
                        for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(dataset_sizes)
    
    count = torch.zeros([NUM_CLASS])
    
    for data in image_datasets['train'].imgs:
        count[data[1]] += 1
        
    count = dataset_sizes['train'] / count
    
    count = count / count.sum()
    print(count)
    return image_dataloader, dataset_sizes, count

def create_nn_model():
    global model_name
    model_name = 'cofe_resnet'
    model = model_net.Model_Net(num_classes = NUM_CLASS).to(DEVICE)
    #model = Resnet.resnet50(NUM_CLASS).to(DEVICE)
    #model = model.to(DEVICE)
    return model

def create_opt_loss(model, bal_var):
    global optimizer_select
    global loss_function_select
    bal_var = bal_var.to(DEVICE)
    optimizer = [#torch.optim.SGD(model.backbone.parameters(), lr = LR, momentum = 0.9, weight_decay = 1e-4),
                 torch.optim.Adam(model.parameters(), lr = LR, weight_decay = 1e-4)
                ]
    set_lr_secheduler = [torch.optim.lr_scheduler.MultiStepLR(optimizer[0], milestones=[60, 100, 150], gamma=0.1),
                        ]
    
    loss_func = [torch.nn.CrossEntropyLoss()]
    optimizer_select = 'Adam'
    loss_function_select = 'crossentropy'
    return optimizer, set_lr_secheduler, loss_func

def load_param(model):
    # load resnet
    params = torch.load("../COFENet/pkl/resnet50.pth")
    for name, param in params.items():
        if name in model.backbone1.state_dict():
            try:
                model.backbone1.state_dict()[name].copy_(param)
                print(name)
            except:
                print("{} can not load.".format(name))
                
        if name in model.backbone2.state_dict():
            try:
                model.backbone2.state_dict()[name].copy_(param)
                print(name)
            except:
                print("{} can not load.".format(name))
# =============================================================================
#         if name in model.backbone2.state_dict():
#             try:
#                 model.backbone2.state_dict()[name].copy_(param)
#                 print(name)
#             except:
#                 print("{} can not load.".format(name))
# =============================================================================
            
    return model

def train_step(model, data, label, loss_func, optimizers, phase):
    if use_gpu:
        b_data = data.to(DEVICE)
        b_label = label.to(DEVICE)
    else:
        b_data = data
        b_label = label
    
    for optimizer in optimizers:
        optimizer.zero_grad() 
        
    output_1, output_2, cam_1, cam_rf_1, cam_2, cam_rf_2 = model(b_data)
    _, predicted = torch.max(output_2.data, 1)
    
    #loss function
    cls_loss = loss_func[0](output_1, b_label) + loss_func[0](output_2, b_label)
# =============================================================================
#     for i in range(1, len(output_1)):
#         cls_loss += loss_func[0](output_1[i], b_label) + loss_func[0](output_2[i], b_label)
# =============================================================================

    er_loss = torch.mean(torch.abs(cam_1.view(cam_1.shape[0], -1) - cam_rf_1.view(cam_rf_1.shape[0], -1))) +\
        torch.mean(torch.abs(cam_2.view(cam_2.shape[0], -1) - cam_rf_2.view(cam_rf_2.shape[0], -1)))

    loss = cls_loss + er_loss / 2
    
    if phase == 'train':
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
    
    return loss.data, cls_loss.data, er_loss.data, predicted.data    

#training
def training(model, job):
    global t_min_loss
    global optimizer_select
    global loss_function_select
    global model_name
    #with torch.autograd.set_detect_anomaly(True):
    max_acc = {'train' : 0.0, 'val' : 0.0}
    min_loss = {'train' : 10000.0, 'val' : 10000.0}
    last_acc = {'train': 0.0, 'val':0.0}
    image_data, dataset_sizes, bal_var = load_data()
    best_epoch = 0
    min_loss_epoch = 0
    optimizers, lr_schedulers, loss_func = create_opt_loss(model, bal_var)
    for epoch in range(EPOCH):
        start = time.time()
        print('Epoch {}/{}'.format(epoch, EPOCH - 1))
        print('-' * 10)
        if CON_MATRIX:
            confusion_matrix = {'train' : np.zeros([NUM_CLASS, NUM_CLASS]), 'val' : np.zeros([NUM_CLASS, NUM_CLASS])}
        for phase in job:
            loss_rate = 0.0
            cls_rate_1 = 0.0
            cls_rate_2 = 0.0
            er_rate = 0.0
            correct = 0.0
            
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)
                
            for step, (data, label) in enumerate(image_data[phase]):
                loss, cls_loss, er_loss, predicted = train_step(model, data, label, loss_func, optimizers, phase)
                if use_gpu:
                    b_data = data.to(DEVICE)
                    b_label = label.to(DEVICE)
                else:
                    b_data = data
                    b_label = label
                    
                loss_rate += loss * b_data.size(0)
                cls_rate_1 += cls_loss * b_data.size(0)
                er_rate += er_loss * b_data.size(0)
                
                correct += (predicted == b_label).sum().item()
                if CON_MATRIX:
                    np.add.at(confusion_matrix[phase], tuple([predicted.cpu().detach().numpy(), b_label.cpu().detach().numpy()]), 1)
            
            loss_rate = loss_rate / dataset_sizes[phase]
            cls_rate_1 = cls_rate_1 / dataset_sizes[phase]
            cls_rate_2 = cls_rate_2 / dataset_sizes[phase]
            er_rate = er_rate / dataset_sizes[phase]
            correct = correct / dataset_sizes[phase]

            if max_acc[phase] < correct:
                last_acc[phase] = max_acc[phase]
                max_acc[phase] = correct
                if phase == 'val':
                    best_epoch = epoch
                    save_data = {'Model_name' : model_name,
                             'Optimizer' : optimizer_select,
                             'Loss_function' : loss_function_select,
                             'Epoch' : epoch + 1,
                             'Best_acc' : correct,
                             'model_param' : model.state_dict()}
                    print('save')
                    torch.save(save_data, './pkl/{}_{}.pkl'.format(model_name, INDEX))
                    
# =============================================================================
#             if phase == "train":
#                 writer.add_scalar('Loss/train', loss, epoch)
#                 writer.add_scalar('Accuracy/train', correct, epoch)
#             else:
#                 writer.add_scalar('Loss/test', loss, epoch)
#                 writer.add_scalar('Accuracy/test', correct, epoch)
# =============================================================================
            
            if min_loss[phase] > loss_rate:
                min_loss[phase] = loss_rate
                if phase == 'train':
                    min_loss_epoch = epoch
            
            print('Index : {}'.format(INDEX))
            print("dataset : {}".format(data_dir))
            print("Model name : {}".format(model_name))
            print("{} set loss : {:.6f}".format(phase, loss_rate))
            print("{} set cls_loss_1 : {:.6f}".format(phase, cls_rate_1))
            print("{} set cls_loss_2 : {:.6f}".format(phase, cls_rate_2))
            print("{} set er_loss : {:.6f}".format(phase, er_rate))
            print("{} set min loss : {:.6f}".format(phase, min_loss[phase]))
            print("{} set acc : {:.6f}%".format(phase, correct * 100.0))
            print("{} last update : {:.6f}%".format(phase, max_acc[phase] * 100 - last_acc[phase] * 100))
            print("{} set max acc : {:.6f}%".format(phase, max_acc[phase] * 100.0))
            if phase == 'train':
                print('min loss epoch : {}'.format(min_loss_epoch))
            if phase == 'val' :
                print('best acc epoch : {}'.format(best_epoch))
            if CON_MATRIX:
                print("{} confusion matrix :".format(phase))
                print(confusion_matrix[phase])
            print()   
        
        for lr_scheduler in lr_schedulers:
            lr_scheduler.step()
            
        print(time.time() - start)
        
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

if __name__ == '__main__':
    model = create_nn_model()
    model = load_param(model)
    training = training(model, ['train', 'val'])

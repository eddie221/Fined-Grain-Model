#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 09:54:05 2020

@author: eddie
"""

import neural_network.vgg_liftpool as vgg_liftpool
import torchvision.transforms as transforms
import torchvision
import torch
#import args
import random
from PIL import Image
import numpy as np
import time
from config import *
if KFOLD == 1:
    from config import VAL_SPLIT
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
import tqdm
import os
import logging

if not os.path.exists('./pkl/{}/'.format(INDEX)):
    os.mkdir('./pkl/{}/'.format(INDEX))
#from torch.utils.tensorboard import SummaryWriter

#print environment information
print(torch.cuda.is_available())
DEVICE = 'cuda:0'

#writer = SummaryWriter('../tensorflow/logs/cub_{}'.format(INDEX), comment = "224_64")

use_gpu = torch.cuda.is_available()

optimizer_select = ''
loss_function_select = ''
model_name = ''
data_name = 'Cifar100'
data_dir = '../datasets/ISIC 2019/'

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

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

def load_data_cifar():
    dataloader = []
    dataset_sizes = []
    trainset = torchvision.datasets.CIFAR100(root='./data',
                                            train = True,
                                            download = True,
                                            transform = data_transforms['train'])
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size = BATCH_SIZE,
                                              shuffle = True,
                                              num_workers = 2)
    
    testset = torchvision.datasets.CIFAR100(root='./data',
                                           train = False,
                                           download = True,
                                           transform = data_transforms['val'])
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size = BATCH_SIZE,
                                             shuffle = False,
                                             num_workers = 2)
    
    dataloader.append({'train' : trainloader, 'val' : testloader})
    dataset_sizes.append({'train' : len(trainloader), 'val' : len(testloader)})
    
    return dataloader, None


def load_data():
    all_image_datasets = torchvision.datasets.ImageFolder(data_dir, data_transforms['train'])
    
    dataloader = []
    dataset_sizes = []
    if KFOLD != 1:
        kf = KFold(KFOLD, shuffle = True)
        for train_idx, val_idx in kf.split(all_image_datasets):
            train_dataset = torch.utils.data.Subset(all_image_datasets, train_idx)
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)    
            val_dataset = torch.utils.data.Subset(all_image_datasets, val_idx)
            valloader = torch.utils.data.DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
            dataloader.append({'train' : trainloader, 'val' : valloader})
            dataset_sizes.append({'train' : len(trainloader), 'val' : len(valloader)})
    else:
        indices = list(range(len(all_image_datasets)))
        dataset_size = len(all_image_datasets)
        split = int(np.floor(VAL_SPLIT * dataset_size))
        np.random.seed(0)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        trainloader = torch.utils.data.DataLoader(all_image_datasets,
                                                       batch_size = BATCH_SIZE,
                                                       sampler = train_sampler,
                                                       num_workers = 16)
        valloader = torch.utils.data.DataLoader(all_image_datasets,
                                                       batch_size = BATCH_SIZE,
                                                       sampler = valid_sampler,
                                                       num_workers = 16)
        dataloader.append({'train' : trainloader, 'val' : valloader})
        dataset_sizes.append({'train' : len(trainloader), 'val' : len(valloader)})
        
        
    return dataloader, dataset_sizes, all_image_datasets
# =============================================================================
#     dataset_sizes = len(all_image_datasets)ting(256, kernel_size=
#     print(dataset_sizes)
#     
#     count = torch.zeros([NUM_CLASS])
#     
#     for data in all_image_datasets.imgs:
#         count[data[1]] += 1
#         
#     count = dataset_sizes['train'] / count
#     
#     count = count / count.sum()
#     print(count)
#     return all_image_dataloader, dataset_sizes, count
# =============================================================================

def create_nn_model():
    global model_name
    model_name = 'vgg_liftpool'
    model = vgg_liftpool.vgg13_bn(num_classes = NUM_CLASS).to(DEVICE)
    #model = resnet.resnet50(num_classes = NUM_CLASS).to(DEVICE)
    assert model_name == model.name, "Wrong model loading. Expect {} but get {}.".format(model_name, model.name)

    print(model)
    if 'liftpool' in model_name:
        print("lift pooling : {}".format(len(model.lifting_pool)))
    return model

def create_opt_loss(model):
    global optimizer_select
    global loss_function_select
    optimizer = [#torch.optim.SGD(model.backbone.parameters(), lr = LR, momentum = 0.9, weight_decay = 1e-4),
                 torch.optim.Adam(model.parameters(), lr = LR, weight_decay = 1e-4)
# =============================================================================
#                  torch.optim.Adam([{'params' : [param for name, param in model.named_parameters() if name != 'Lifting_down']},
#                                    {'params' : [param for name, param in model.named_parameters() if name == 'Lifting_down'], 'lr' : 1e-3}],
#                                   lr = LR, weight_decay = 1e-4)
# =============================================================================
                ]
    set_lr_secheduler = [torch.optim.lr_scheduler.MultiStepLR(optimizer[0], milestones=[100, 200, 300], gamma=0.1),
                        ]
    
    loss_func = [torch.nn.CrossEntropyLoss(),
                 torch.nn.MSELoss()]
    optimizer_select = 'Adam'
    loss_function_select = 'crossentropy'
    return optimizer, set_lr_secheduler, loss_func

def load_param(model):
    # load resnet
    params = torch.load("../pretrain/resnet50.pth")
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
# =============================================================================
#     load = []
#     not_load = []
#     for name, param in params.items():
#         if name in model.backbone1.state_dict():
#             try:
#                 model.backbone1.state_dict()[name].copy_(param)
#                 load.append(name)
#             except:
#                 not_load.append(name)
#         else:
#             print(name)
#                 
#     print("Load {} layers".format(len(load)))
#     print("Not load {} layers".format(len(not_load)))
#     load = []
#     not_load = []
#     params = torch.load("../pretrain/resnet50.pth")
#     for name, param in params.items():
#         if name in model.backbone2.state_dict():
#             try:
#                 model.backbone2.state_dict()[name].copy_(param)
#                 load.append(name)
#             except:
#                 not_load.append(name)
#         else:
#             print(name)
#             
#     print("Load {} layers".format(len(load)))
#     print("Not load {} layers".format(len(not_load)))
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
    output_1 = model(b_data)
    _, predicted = torch.max(output_1.data, 1)
    
    #loss function
    cls_loss = loss_func[0](output_1, b_label)# + loss_func[0](output_1[1], b_label) + loss_func[0](output_1[2], b_label) + loss_func[0](output_1[3], b_label)
    loss = cls_loss
    for j in range(len(model.lifting_pool)):
        loss += 1e-4 * model.lifting_pool[j].regular_term_loss()
    
    if phase == 'train':
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
            
    return loss.item(), predicted.detach().cpu()

#training
def training(job):
    global t_min_loss
    global optimizer_select
    global loss_function_select
    global model_name
    #with torch.autograd.set_detect_anomaly(True):
    #kfold_image_data, dataset_sizes, all_image_datasets = load_data()
    kfold_image_data, all_image_datasets = load_data_cifar()
    ACCMeters = []
    LOSSMeters = []
    for i in range(KFOLD):
        ACCMeters.append(AverageMeter(True))
        LOSSMeters.append(AverageMeter(False))
        
    for index, image_data in enumerate(kfold_image_data):
        model = create_nn_model()
        if PRETRAIN:
            print("Load pretrained")
            model = load_param(model)
        else:
            print("Not load pretrained")
        optimizers, lr_schedulers, loss_func = create_opt_loss(model)
        max_acc = {'train' : AverageMeter(True), 'val' : AverageMeter(True)}
        min_loss = {'train' : AverageMeter(False), 'val' : AverageMeter(False)}
        last_acc = {'train' : AverageMeter(True), 'val' : AverageMeter(True)}
        
        for epoch in range(1, EPOCH + 1):
            start = time.time()
            print('Fold {}/{} Epoch {}/{}'.format(index + 1, KFOLD, epoch, EPOCH))
            logging.info("-" * 15)
            logging.info('Fold {}/{} Epoch {}/{}'.format(index + 1, KFOLD, epoch, EPOCH))
            print('-' * 10)
            if CON_MATRIX:
                confusion_matrix = {'train' : np.zeros([NUM_CLASS, NUM_CLASS]), 'val' : np.zeros([NUM_CLASS, NUM_CLASS])}
            for phase in job:
                loss_t = AverageMeter(False)
                correct_t = AverageMeter(True)
                cls_rate_1 = AverageMeter(False)
                
                if phase == 'train':
                    model.train(True)
                    if all_image_datasets is not None:
                        all_image_datasets.transform = data_transforms['train']
                else:
                    model.train(False)
                    if all_image_datasets is not None:
                        all_image_datasets.transform = data_transforms['val']
                step = 0
                for data, label in tqdm.tqdm(image_data[phase]):
                    loss, predicted = train_step(model, data, label, loss_func, optimizers, phase)
                    
                    loss_t.update(loss, data.size(0))
                    correct_t.update((predicted.cpu() == label).sum().item(), label.shape[0])
                    
                    step += 1
                    if CON_MATRIX:
                        np.add.at(confusion_matrix[phase], tuple([predicted.cpu().numpy(), label.detach().numpy()]), 1)
                if max_acc[phase].avg < correct_t.avg:
                    last_acc[phase] = max_acc[phase]
                    max_acc[phase] = correct_t
                    
                    if phase == 'val':
                        ACCMeters[index] = correct_t
                        LOSSMeters[index] = loss_t
                        save_data = model.state_dict()
                        print('save')
                        torch.save(save_data, './pkl/{}/fold_{}_best_{}.pkl'.format(INDEX, index, INDEX))
                        
                logging.info("{} set loss : {:.6f}".format(phase, loss_t.avg))        
                logging.info("{} set acc : {:.6f}%".format(phase, correct_t.avg * 100.))        
                print('Index : {}'.format(INDEX))
                print("dataset : {}".format(data_name))
                print("Model name : {}".format(model_name))
                print("{} set loss : {:.6f}".format(phase, loss_t.avg))
                print("{} set cls_loss_1 : {:.6f}".format(phase, cls_rate_1.avg))
                print("{} set min loss : {:.6f}".format(phase, min_loss[phase].avg))
                print("{} set acc : {:.6f}%".format(phase, correct_t.avg * 100.))
                print("{} last update : {:.6f}%".format(phase, (max_acc[phase].avg - last_acc[phase].avg) * 100.))
                print("{} set max acc : {:.6f}%".format(phase, max_acc[phase].avg * 100.))
                if CON_MATRIX:
                    print("{} confusion matrix :".format(phase))
                    print(confusion_matrix[phase])
                print()   
            
            for lr_scheduler in lr_schedulers:
                lr_scheduler.step()
                
            print(time.time() - start)
        del model
        del optimizers
        del lr_schedulers
        del loss_func
    acc = 0
    loss = 0
    for idx in range(1, len(ACCMeters) + 1):
        print("Fold {} best acc : {:.6f} loss : {:.6f}".format(idx, ACCMeters[idx - 1].avg, LOSSMeters[idx - 1].avg))
        acc += ACCMeters[idx - 1].avg
        loss += LOSSMeters[idx - 1].avg
    print("Avg. ACC : {:.6f} Avg. Loss : {:.6f}".format(acc / KFOLD, loss / KFOLD))
    
class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self, acc):
        self.reset()
        self.acc = acc
    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, batch):
        self.value = value
        if self.acc:
            self.sum += value
        else:       
            self.sum += value * batch
        self.count += batch
        self.avg = self.sum / self.count

        
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
    logging.basicConfig(filename = './pkl/{}/logging.txt'.format(INDEX), level=logging.DEBUG)
    logging.info('Index : {}'.format(INDEX))
    logging.info("dataset : {}".format(data_dir))
    training = training(['train', 'val'])

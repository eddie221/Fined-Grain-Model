#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 13:30:26 2021

@author: mmplab603
"""

import neural_network.UNet as UNet
import torch
#import args
import numpy as np
import time
from config_seg import *
import tqdm
import os
import logging
from load_dataset_seg import load_voc12
import neural_network.resnet_seg as resnet_seg
import neural_network.resnet_seg_LDW as resnet_seg_LDW

if not os.path.exists('./pkl/{}/'.format(INDEX)):
    os.mkdir('./pkl/{}/'.format(INDEX))

#print environment information
print(torch.cuda.is_available())
DEVICE = 'cuda:0'

use_gpu = torch.cuda.is_available()

optimizer_select = ''
loss_function_select = ''
model_name = ''
data_name = 'voc12'

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

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
    model_name = 'Resnet_seg_LDW'
    #model_name = 'UNet'
    model = resnet_seg_LDW.resnet50(num_classes = NUM_CLASS).to(DEVICE)
    #model = UNet.UNet(3, NUM_CLASS).to(DEVICE)
    print(NUM_CLASS)
    #model = resnet.resnet50(num_classes = NUM_CLASS).to(DEVICE)
    assert model_name == model.name, "Wrong model loading. Expect {} but get {}.".format(model_name, model.name)
    
    print(model)
    if 'LDW' in model_name:
        print("LDW-pooling : {}".format(len(model.lifting_pool)))
    return model

def create_opt_loss(model):
    global optimizer_select
    global loss_function_select
    optimizer = [torch.optim.SGD(model.parameters(), lr = LR, momentum = 0.9, weight_decay = 5e-4),
                 #torch.optim.Adam(model.parameters(), lr = LR, weight_decay = 1e-4)
# =============================================================================
#                  torch.optim.Adam([{'params' : [param for name, param in model.named_parameters() if name != 'Lifting_down']},
#                                    {'params' : [param for name, param in model.named_parameters() if name == 'Lifting_down'], 'lr' : 1e-3}],
#                                   lr = LR, weight_decay = 1e-4)
# =============================================================================
                ]
    set_lr_secheduler = [torch.optim.lr_scheduler.MultiStepLR(optimizer[0], milestones=[50, 100, 150, 200, 250, 300, 350, 400, 450], gamma=0.1),
                        ]
    
    loss_func = [torch.nn.CrossEntropyLoss(),]
    optimizer_select = 'SGD'
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
# =============================================================================
# 
# def mIoU(pred_mask, mask, n_classes=23, smooth=1e-10):
#     with torch.no_grad():
#         pred_mask = torch.nn.functional.softmax(pred_mask, dim=1)
#         pred_mask = torch.argmax(pred_mask, dim=1)
#         pred_mask = pred_mask.contiguous().view(-1)
#         mask = mask.contiguous().view(-1)
# 
#         iou_per_class = []
#         for clas in range(0, n_classes): #loop per pixel class
#             true_class = pred_mask == clas
#             true_label = mask == clas
# 
#             if true_label.long().sum().item() == 0: #no exist label in this loop
#                 iou_per_class.append(np.nan)
#             else:
#                 intersect = torch.logical_and(true_class, true_label).sum().float().item()
#                 union = torch.logical_or(true_class, true_label).sum().float().item()
# 
#                 iou = (intersect + smooth) / (union +smooth)
#                 iou_per_class.append(iou)
#         return np.nanmean(iou_per_class)
# =============================================================================

def mIoU(pred, target, n_classes = 21):
    with torch.no_grad():
        ious = []
        pred = torch.argmax(pred, dim = 1)
# =============================================================================
#         for i in range(4):
#             plt.subplot(4, 2, i * 2 + 1)
#             plt.imshow(pred[i])
#             plt.subplot(4, 2, i * 2 + 2)
#             plt.imshow(target[i])
#         plt.show()
# =============================================================================
        pred = pred.view(-1)
        target = target.view(-1)
        test = 0
        # Ignore IoU for background class ("0")
        for cls in range(n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
            pred_inds = pred == cls
            target_inds = target == cls
            intersection = (pred_inds[target_inds]).long().sum()  # Cast to long to prevent overflows
            union = pred_inds.long().sum() + target_inds.long().sum() - intersection
            if union == 0:
                ious.append(float("nan"))  # If there is no ground truth, do not include in evaluation
            else:
                ious.append(float(intersection) / float(union))
        ious = np.array(ious)
        miou = np.nanmean(ious)
    return miou

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
    #if phase == 'val':
    #miou = mIoU(output_1.cpu().data, label, NUM_CLASS)
    #loss function
    cls_loss = loss_func[0](output_1, b_label)# + loss_func[0](output_1[1], b_label) + loss_func[0](output_1[2], b_label) + loss_func[0](output_1[3], b_label)
    filter_constraint = 0
    for j in range(len(model.lifting_pool)):
        filter_constraint += model.lifting_pool[j].regular_term_loss()
    
    loss = filter_constraint / len(model.lifting_pool) + cls_loss
    
    if phase == 'train':
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
            
    return loss.item(), output_1.detach().cpu().data

#training
def training(job):
    global t_min_loss
    global optimizer_select
    global loss_function_select
    global model_name
    #with torch.autograd.set_detect_anomaly(True):
    #kfold_image_data, dataset_sizes, all_image_datasets = load_data()
    #kfold_image_data, all_image_datasets = load_data_cifar("./data")
    kfold_image_data, all_image_datasets = load_voc12("./SegmentationClassAug")
    #kfold_image_data, all_image_datasets = load_voc12("/home/mmplab603/program/datasets/VOCdevkit/VOC2012/SegmentationClassAug/")
    miouMeters = []
    LOSSMeters = []
    miou_class = MIOU(num_classes = NUM_CLASS)
    for i in range(KFOLD):
        miouMeters.append(0.0)
        LOSSMeters.append(AverageMeter(False))
        
    for index, image_data in enumerate(kfold_image_data):
        model = create_nn_model()
        if PRETRAIN:
            print("Load pretrained")
            model = load_param(model)
        else:
            print("Not load pretrained")
        optimizers, lr_schedulers, loss_func = create_opt_loss(model)
        max_miou = {'train' : 0.0, 'val' : 0.0}
        min_loss = {'train' : AverageMeter(False), 'val' : AverageMeter(False)}
        last_miou = {'train' : 0.0, 'val' : 0.0}
        
        for epoch in range(1, EPOCH + 1):
            start = time.time()
            print('Fold {}/{} Epoch {}/{}'.format(index + 1, KFOLD, epoch, EPOCH))
            logging.info("-" * 15)
            logging.info('Fold {}/{} Epoch {}/{}'.format(index + 1, KFOLD, epoch, EPOCH))
            print('-' * 10)
            if CON_MATRIX:
                confusion_matrix = {'train' : np.zeros([NUM_CLASS, NUM_CLASS]), 'val' : np.zeros([NUM_CLASS, NUM_CLASS])}
            for phase in job:
                print("phase", phase)
                loss_t = AverageMeter(False)
                cls_rate = AverageMeter(False)
                inter_record = AverageMeter(False)
                union_record = AverageMeter(False)
                #print(inter_record.avg / union_record.avg)
                
                if phase == 'train':
                    model.train(True)
                    if all_image_datasets is not None:
                        all_image_datasets.transform = data_transforms['train']
                else:
                    model.train(False)
                    if all_image_datasets is not None:
                        all_image_datasets.transform = data_transforms['val']
                step = 1
                for data, label in tqdm.tqdm(image_data[phase]):
                    loss, output = train_step(model, data, label, loss_func, optimizers, phase)
                    inter, union = miou_class.get_iou(output, label)
                    inter_record.update(inter)
                    union_record.update(union)
                    loss_t.update(loss, data.size(0))
                    step += 1
                    if CON_MATRIX:
                        np.add.at(confusion_matrix[phase], tuple([predicted.cpu().numpy(), label.detach().numpy()]), 1)
                
                if max_miou[phase] < (inter_record.sum / (union_record.sum + 1e-10)).mean() * 100:
                    last_miou[phase] = max_miou[phase]
                    max_miou[phase] = (inter_record.sum / (union_record.sum + 1e-10)).mean() * 100
                    
                    if phase == 'val':
                        miouMeters[index] = (inter_record.sum / (union_record.sum + 1e-10)).mean() * 100
                        LOSSMeters[index] = loss_t
                        save_data = model.state_dict()
                        print('save')
                        torch.save(save_data, './pkl/{}/fold_{}_best_{}.pkl'.format(INDEX, index, INDEX))
                        
                logging.info("{} set loss : {:.6f}".format(phase, loss_t.avg))        
                logging.info("{} set mIoU : {:.6f}%".format(phase, (inter_record.sum / (union_record.sum + 1e-10)).mean() * 100))        
                print('Index : {}'.format(INDEX))
                print("dataset : {}".format(data_name))
                print("Model name : {}".format(model_name))
                print("{} set loss : {:.6f}".format(phase, loss_t.avg))
                print(inter_record.sum / (union_record.sum + 1e-10) * 100)
                #print("{} set cls : {:.6f}".format(phase, cls_rate_1.avg))
                #print("{} set min loss : {:.6f}".format(phase, min_loss[phase].avg))
                print("{} set mIoU : {:.6f}".format(phase, (inter_record.sum / (union_record.sum + 1e-10)).mean() * 100))
                print("{} last update : {:.6f}".format(phase, (max_miou[phase] - last_miou[phase])))
                print("{} set max mIoU : {:.6f}".format(phase, max_miou[phase]))
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
    miou = 0
    loss = 0
    for idx in range(1, len(miouMeters) + 1):
        print("Fold {} best miou : {:.6f} loss : {:.6f}".format(idx, miouMeters[idx - 1], LOSSMeters[idx - 1].avg))
        miou += miouMeters[idx - 1]
        loss += LOSSMeters[idx - 1].avg
    print("Avg. miou : {:.6f} Avg. Loss : {:.6f}".format(miou / KFOLD, loss / KFOLD))
    
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

    def update(self, value, batch = 1):
        self.value = value
        if self.acc:
            self.sum += value
        else:       
            self.sum += value * batch
        self.count += batch
        self.avg = self.sum / self.count

class MIOU(object):
    def __init__(self, num_classes=21):
        self.num_classes = num_classes
        self.epsilon = 1e-6

    def get_iou(self, output, target):
        if isinstance(output, tuple):
            output = output[0]

        _, pred = torch.max(output, 1)

        # histc in torch is implemented only for cpu tensors, so move your tensors to CPU
        if pred.device == torch.device('cuda'):
            pred = pred.cpu()
        if target.device == torch.device('cuda'):
            target = target.cpu()

# =============================================================================
#         pred = pred.type(torch.ByteTensor)
#         target = target.type(torch.ByteTensor)
# =============================================================================

        # shift by 1 so that 255 is 0
        pred += 1
        target += 1
        pred = pred * (target > 0).long()
        inter = pred * (pred == target).long()
        area_inter = torch.histc(inter.float(), bins=self.num_classes, min=1, max=self.num_classes)
        area_pred = torch.histc(pred.float(), bins=self.num_classes, min=1, max=self.num_classes)
        area_mask = torch.histc(target.float(), bins=self.num_classes, min=1, max=self.num_classes)
        area_union = area_pred + area_mask - area_inter + self.epsilon

        return area_inter.numpy(), area_union.numpy()
        
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
    logging.basicConfig(filename = './pkl/{}/logging.txt'.format(INDEX), level=logging.INFO)
    logging.info('Index : {}'.format(INDEX))
    logging.info("dataset : {}".format(data_name))
    training = training(['train', 'val'])
    #create_nn_model()
    #dataloader, _ = load_ImageNet("../../dataset/imagenet")
    #kfold_image_data, all_image_datasets = load_voc12("../../program/datasets/VOCdevkit/VOC2012/Segmentation/")
    
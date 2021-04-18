#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:34:25 2021

@author: mmplab603
"""

import torch
import torch.nn as nn
from neural_network.lifting_pool import Lifting_down

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, lifting = False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class dev_model(nn.Module):
    def __init__(self, num_classes):
        super(dev_model, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride = 2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride = 2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride = 2)
        
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer([num_classes], 2048)
        
        self.lifting_pool = []
        for m in self.modules():
            if isinstance(m, Lifting_down):
                self.lifting_pool.append(m)
                
        
    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """
        Construct fully connected layer

        - fc_dims (list or tuple): dimensions of fc layers, if None,
                                   no fc layers are constructed
        - input_dim (int): input dimension
        - dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        
        assert isinstance(fc_dims, (list, tuple)), "fc_dims must be either list or tuple, but got {}".format(type(fc_dims))
        
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        
        self.feature_dim = fc_dims[-1]
        
        return nn.Sequential(*layers)
    
    def energy_filter(self, x):
        batch, channel, height, width = x.shape
        x_energe = torch.mean(torch.mean(torch.pow(x, 2), dim = -1), dim = -1)
        x_energe_index = torch.argsort(x_energe, dim = 1)
        x_energe_index = x_energe_index[:, :channel // 2]#:x_energe_index.shape[1] // 2]
        x = x.reshape(-1, height, width)
        x_energe_index = x_energe_index + torch.arange(0, batch).reshape(-1, 1).cuda() * channel
        x = x[x_energe_index.reshape(-1)]
        x = x.reshape(batch, -1, height, width)
        return x
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # layer1
        x = self.layer1(x)
        
        # layer2
        x = self.layer2(x)
        
        # layer3
        x = self.layer3(x)
        
        # layer4
        x = self.layer4(x)
        
        x = self.avg(x).view(x.shape[0], -1)
        x = self.fc(x)
        
        return x
    
    
def dev_mod(num_classes = 1000):
    model = dev_model(num_classes)
    return model
    
if __name__ == "__main__":
    model = dev_model(9).cuda()
    a = torch.randn([8, 3, 224, 224]).cuda()
    optim = torch.optim.Adam(model.parameters(), lr = 1e-4)
# =============================================================================
#     optim = torch.optim.Adam([
#         {'params' : [param for name, param in model.named_parameters() if name != "Lifting_down"]},
#         {'params' : [param for name, param in model.named_parameters() if name == "Lifting_down"], 'lr' : 1e-2},
#         ], lr = 1e-4, weight_decay = 1e-4)
# =============================================================================
# =============================================================================
#     param = torch.load('../pkl/fold_0_best_20210408-3.pkl')
#     model.load_state_dict(param)
# =============================================================================
    
    loss_f = torch.nn.CrossEntropyLoss()
    
    label = torch.tensor([0, 1, 3, 2, 1, 0, 3, 2]).cuda()
    for i in range(10):
        output = model(a)
        optim.zero_grad()
        loss = loss_f(output, label)
        
        loss.backward()
        optim.step()
        for j in range(len(model.lifting_pool)):
            model.lifting_pool[j].filter_constraint()
    #model.lifting_pool[0].filter_constraint()
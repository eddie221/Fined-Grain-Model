#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:34:25 2021

@author: mmplab603
"""

import torch
import torch.nn as nn
import neural_network.lifting_pool as lift_pool
import numpy as np

class feature_block(nn.Module):
    def __init__(self, inplanes, planes, lifting = False):
        super(feature_block, self).__init__()
        self.lifting = lifting
        self.planes = planes
        self.conv1 = nn.Conv2d(inplanes, planes, 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(planes)
        
        if lifting:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes * 4, 1, stride = 2),
                                            nn.BatchNorm2d(planes * 4))
            self.avg = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(nn.Linear(planes, planes // 2),
                                    nn.ReLU(),
                                    nn.Linear(planes // 2, planes),
                                    nn.Sigmoid()
                                    )
            
            self.conv3 = nn.Conv2d(planes, planes * 4, 1)
        else:
            self.conv3 = nn.Conv2d(planes, planes * 4, 1)
            
        self.bn3 = nn.BatchNorm2d(planes * 4)
        
        self.relu = nn.ReLU(inplace=True)  
        
    def attention(self, x):
        x_avg = self.avg(x)
        x_att = self.fc(x_avg.view(x.shape[0], -1))
        x_att = x_att.unsqueeze(2).unsqueeze(2)
        x = x_att * x + x
        return x
        
        
    def forward(self, x):
        batch, channel, h, w = x.shape
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        if self.lifting:
            x = lift_pool.lifting_down(x)
            x = torch.cat(x, dim = 1)
            batch, channel, h, w = x.shape
            x_energe = self.avg(torch.pow(x, 2)).squeeze(-1).squeeze(-1)
            val, idx = torch.topk(x_energe, dim = 1, k = self.planes) 
            idx = np.array(idx) + np.arange(0, self.planes * 4 * batch, self.planes * 4).reshape(-1, 1)
            x = x.view(-1, h, w)
            x = x[idx.reshape(-1, 1).squeeze(1)]
            x = x.view(batch, -1, h, w)
            x = self.attention(x)
            residual = self.downsample(residual)
            
        x = self.conv3(x)
        x = self.bn3(x)
        
        x = x + residual
        x = self.relu(x)
        return x
    
class dev_model(nn.Module):
    def __init__(self, num_classes):
        super(dev_model, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(self.inplanes, 64, 3)
        self.layer2 = self._make_layer(self.inplanes, 128, 4)
        self.layer3 = self._make_layer(self.inplanes, 256, 6)
        self.layer4 = self._make_layer(self.inplanes, 512, 3)
        
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer([num_classes], 2048)
        
    def _make_layer(self, inplanes, planes, blocks):
        layers = []
        layers.append(feature_block(inplanes, planes, True))
        inplanes = planes * 4
        for i in range(1, blocks):
            layers.append(feature_block(inplanes, planes))
        self.inplanes = inplanes
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
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg(x).view(x.shape[0], -1)
        x = self.fc(x)
        return x
    
    
def dev_mod(num_classes = 1000):
    model = dev_model(num_classes)
    return model

if __name__ == "__main__":
    model = dev_model(9)
    print(model)
    a = torch.randn([2, 3, 224, 224])
    model(a)
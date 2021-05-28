
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:34:25 2021
@author: mmplab603
"""

import torch
import torch.nn as nn
from neural_network.lifting_pool import Lifting_down

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class Energy_attention(nn.Module):
    def __init__(self, in_cha):
        super(Energy_attention, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(in_cha)
        self.relu = nn.ReLU()
        self.SE = nn.Sequential(nn.Linear(in_cha, in_cha // 4),
                                nn.ReLU(),
                                nn.Linear(in_cha // 4, in_cha),
                                nn.Sigmoid())
        
    def forward(self, x):
        x_norm = self.instance_norm(x)
        x_norm = self.relu(x_norm)
        x_energy = torch.mean(torch.mean(torch.pow(x_norm, 2), dim = -1), dim = -1)
        
        x_energy = self.SE(x_energy)
        x = x * x_energy.unsqueeze(-1).unsqueeze(-1)
        
        return x
    
class Resnet(nn.Module):
    def __init__(self, num_classes, layers):
        super(Resnet, self).__init__()
        self.name = "resnet_liftpool"
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.lifting1 = nn.Sequential(Lifting_down(64, 2, 2, pad_mode = "pad0"),
                                      nn.Conv2d(256, 64, 1, bias = False),
                                      Energy_attention(64))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.lifting2 = nn.Sequential(Lifting_down(64, 2, 2, pad_mode = "pad0"),
                                      nn.Conv2d(256, 64, 1, bias = False),
                                      Energy_attention(64))
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride = 2)
        
        self.avg = nn.AdaptiveAvgPool2d(1)
        #self.fc = self._construct_fc_layer([num_classes], 2048)
        self.fc = nn.Linear(2048, num_classes)
        
        self.lifting_pool = []
        for m in self.modules():
            if isinstance(m, Lifting_down):
                self.lifting_pool.append(m)
                
        
    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=1, bias=False),
                    Lifting_down(planes, 2, stride = 2, pad_mode = "pad0"),
                    Energy_attention(planes * 4),
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
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.lifting1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.lifting2(x)
        #x = self.maxpool(x)
        # 64, 112, 112
        
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
    
    
def resnet50(num_classes = 1000):
    model = Resnet(num_classes, [3, 4, 6, 3])
    return model


def resnet101(num_classes = 1000):
    model = Resnet(num_classes, [3, 4, 23, 3])
    return model
    
if __name__ == "__main__":
    model = resnet50(9).to("cuda:0")
    a = torch.randn([2, 3, 224, 224]).to("cuda:0")
    optim = torch.optim.Adam(model.parameters(), lr = 1e-4)
    print(len(model.lifting_pool))
    with open('../lifting.txt', 'w') as f:
        print(model, file = f)
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
    label = torch.tensor([0, 1]).cuda()
    for i in range(5):
        output = model(a)
        optim.zero_grad()
        loss = loss_f(output, label)
        for j in range(len(model.lifting_pool)):
            loss += 1e-4 * model.lifting_pool[j].regular_term_loss().squeeze(0)
        loss.backward()
        print(loss)
        optim.step()
        
    #model.lifting_pool[0].filter_constraint()
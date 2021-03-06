#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 17:46:50 2020

@author: eddie
"""
import torch.nn as nn
import torch
import math

class GNN(nn.Module):
    def __init__(self, feature, threshold = 0.005, bias = True, power = 1):
        super(GNN, self).__init__()
        self.A = None
        self.D = None
        self.power = power
        self.laplacian = None
        self.laplacian_sym = None
        self.laplacian_rw = None
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.layer = len(feature) - 1
            
        self.W = nn.ModuleList([])
        for i in range(1, len(feature)):
            self.W.append(nn.Linear(feature[i - 1], feature[i]))
        
# =============================================================================
#     def __init_W_bias__(self):
#         for i in range(self.layer):
#             stdv = 1. / math.sqrt(self.W[i].size(1))
#             self.W[i].data.uniform_(-stdv, stdv)
# =============================================================================
            
    def correlation(self, x):
        if len(x.shape) == 4:
            batch, channel, height, width = x.shape
            x = x.view(batch, channel, -1)

        x_t = x.permute(0, 2, 1)
        x_2 = torch.pow(x, 2)
        x_2 = torch.sqrt(torch.sum(x_2, dim = 2, keepdim = True))
        x_t_2 = x_2.permute(0, 2, 1)
        Denominator_x = torch.bmm(x_2, x_t_2)
        Numerator_x = torch.bmm(x, x_t)
        norm_x = Numerator_x / Denominator_x
        return norm_x
    
    def init_Adjency_Degree_matrix(self, x):
        with torch.no_grad():
            cha_cor = self.correlation(x)
            self.A = cha_cor - self.threshold
            self.A = self.relu(self.A)
            self.A[self.A != 0] = 1
            self.A = self.A - torch.eye(self.A.shape[1]).cuda()
            self.D = torch.diag_embed(torch.sum(self.A, dim = 2))
            self.laplacian = self.D - self.A
            D_sqrt_inv = torch.inverse(torch.sqrt(self.D))
            self.laplacian_sym = torch.bmm(torch.bmm(D_sqrt_inv, self.A), D_sqrt_inv)
            
    def __init_matrix__(self, x, _dir = None):
        batch, channel = x.shape[0:2]
        x_linear = x.view(batch, channel, -1)
            
        self.init_Adjency_Degree_matrix(x_linear)
        return x_linear
            
    def forward(self, x):    
        x_linear = self.__init_matrix__(x)
        x_result = []
        for i in range(self.layer):
            lx = torch.bmm(self.laplacian, x_linear)
            x_linear = self.W[i](lx)
            x_result.append(x_linear)
            x_linear = torch.nn.functional.instance_norm(x_linear)
            x_linear = self.relu(x_linear)
            
        return x_result
    
if __name__ == '__main__':
    torch.manual_seed(0)
    a = torch.randn([1, 5, 25]).cuda()
    #a = torch.arange(25, dtype = torch.float).reshape([1, 1, 5, 5])
    #a = torch.cat([a, a, a], dim = 1)
<<<<<<< Updated upstream
    gnn = GNN([25, 30, 35], power = 1, channel_feature = False).cuda()
=======
    gnn = GNN([25, 10, 5], power = 1).cuda()
>>>>>>> Stashed changes
    
    a = gnn(a)
    print(len(a))
    print(a)

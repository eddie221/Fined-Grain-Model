#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 17:46:50 2020

@author: eddie
"""
import torch.nn as nn
import torch
import math
import numpy as np

class GNN(nn.Module):
    def __init__(self, feature, threshold = 0.005, bias = True, dist = 1, power = 1, grid_adjacency = False):
        super(GNN, self).__init__()
        self.distance = dist
        self.grid_adjacency = grid_adjacency
        self.A = None
        self.D = None
        self.power = power
        self.laplacian = None
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.layer = len(feature) - 1
        self.W = nn.ParameterList([])

        if bias:
            self.bias = nn.ParameterList([])
        else:
            self.bias = None
        for i in range(1, len(feature)):
            self.W.append(nn.Parameter(torch.randn(1, feature[i - 1], feature[i])))
            self.bias.append(nn.Parameter(torch.randn(1)))
            
        self.__init_W_bias__()
        
    def __init_W_bias__(self):
        for i in range(self.layer):
            stdv = 1. / math.sqrt(self.W[i].size(1))
            self.W[i].data.uniform_(-stdv, stdv)
            
            if self.bias is not None:
                self.bias[i].data.uniform_(-stdv, stdv)
            
    def channel_correlation(self, x):
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
    
    def init_Adjency_Degree_matrix2(self, x):
        if x.get_device() == -1:
            device = "cpu"
        else:
            device = x.get_device()
            
        if len(x.shape) == 3:
            x = x.permute(0, 2, 1)
            x = x.reshape([x.shape[0], x.shape[1], int(np.sqrt(x.shape[2])), int(np.sqrt(x.shape[2]))])
        
        unfold = nn.Unfold(3 + (self.distance - 1) * 2, padding = self.distance)
        fold = nn.Fold(x.shape[2], 3 + (self.distance - 1) * 2, padding = self.distance)
        base = torch.zeros([x.shape[0], 1, x.shape[2], x.shape[3]]).to(device)
        adjency = []

        with torch.no_grad():
            adjency_base = unfold(base).permute(0, 2, 1)
            num = adjency_base.shape[1]
            for i in range(num):
                adjency_base = unfold(base).permute(0, 2, 1)
                adjency_base[:, i, :] = 1
                adjency_base = fold(adjency_base.permute(0, 2, 1))
                adjency.append(adjency_base.view(x.shape[0], -1))
            adjency = torch.stack(adjency, dim = 1)
        
        self.A = adjency
        
        for i in range(1, self.power):
            self.A = torch.bmm(self.A, adjency)
        
        self.D = torch.diag_embed(torch.sum(self.A, dim = 2))
        D_inv_sqrt = torch.inverse(torch.sqrt(self.D))
        self.laplacian = torch.torch.bmm(torch.bmm(D_inv_sqrt, self.A), D_inv_sqrt)
            
    def init_Adjency_Degree_matrix(self, x):
        if x.get_device() == -1:
            device = "cpu"
        else:
            device = x.get_device()
            
        with torch.no_grad():
            cha_cor = self.channel_correlation(x)
            self.A = self.relu(cha_cor)
            self.A = torch.where(self.A > self.threshold,
                                 torch.tensor([1.]).to(device),
                                 torch.tensor([0.]).to(device))
            self.D = torch.diag_embed(torch.sum(self.A, dim = 2))
            D_inv_sqrt = torch.inverse(torch.sqrt(self.D))
            self.laplacian = torch.torch.bmm(torch.bmm(D_inv_sqrt, self.A), D_inv_sqrt)
            
    def forward(self, x):
        
        if len(x.shape) == 4:
            batch, channel, height, width = x.shape
            x_linear = x.view(batch, channel, -1)
            self.init_Adjency_Degree_matrix(x)
        else:
            batch, channel, feature = x.shape
            x_linear = x
            if self.grid_adjacency:
                self.init_Adjency_Degree_matrix2(x)
            else:
                self.init_Adjency_Degree_matrix(x)
                
        for i in range(self.layer):
            lx = torch.bmm(self.laplacian, x_linear)
            self._W = self.W[i].repeat(batch, 1, 1)
            if self.bias is not None:
                x_linear = torch.matmul(lx, self._W) + self.bias[i]
            else:
                x_linear = torch.matmul(lx, self._W)
            x_linear = self.relu(x_linear)
            
        return x_linear
    
if __name__ == '__main__':
    torch.manual_seed(0)
    a = torch.randn([1, 36, 5]).cuda()
    #a = torch.arange(25, dtype = torch.float).reshape([1, 1, 5, 5])
    #a = torch.cat([a, a, a], dim = 1)
    gnn = GNN([5,4,3], dist = 2, power = 2).cuda()
    
    a = gnn(a)

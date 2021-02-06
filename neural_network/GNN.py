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
    def __init__(self, feature, threshold = 0.005, bias = True, dist = 1, power = 1, grid_adjacency = False, channel_feature = True, direction = None):
        super(GNN, self).__init__()
        self.distance = dist
        self.grid_adjacency = grid_adjacency
        self.channel_feature = channel_feature
        self.direction = direction
        self.A = None
        self.D = None
        self.power = power
        self.laplacian = None
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.layer = len(feature) - 1
            
        if direction is None:
            self.W = nn.ParameterList([])
            for i in range(1, len(feature)):
                self.W.append(nn.Parameter(torch.randn(1, feature[i - 1], feature[i])))
                
        else:
            self.W = []
            for i in range(0, len(direction)):
                self.W.append(nn.ParameterList([]))
                for j in range(1, len(feature)):
                    self.W[i].append(nn.Parameter(torch.randn(1, feature[j - 1], feature[j])))
        self.pad = nn.ReplicationPad2d(self.distance)
        
            
        self.__init_W_bias__()
        
    def __init_W_bias__(self):
        if self.direction is None:
            for i in range(self.layer):
                stdv = 1. / math.sqrt(self.W[i].size(1))
                self.W[i].data.uniform_(-stdv, stdv)
        else:
            for i in range(len(self.direction)):
                for j in range(self.layer):
                    stdv = 1. / math.sqrt(self.W[i][j].size(1))
                    self.W[i][j].data.uniform_(-stdv, stdv)
            
    def correlation(self, x):
        if len(x.shape) == 4:
            batch, channel, height, width = x.shape
            x = x.view(batch, channel, -1)
            if self.channel_feature:
                x = x.permute(0, 2, 1)

        x_t = x.permute(0, 2, 1)
        x_2 = torch.pow(x, 2)
        x_2 = torch.sqrt(torch.sum(x_2, dim = 2, keepdim = True))
        x_t_2 = x_2.permute(0, 2, 1)
        Denominator_x = torch.bmm(x_2, x_t_2)
        Numerator_x = torch.bmm(x, x_t)
        norm_x = Numerator_x / Denominator_x
        return norm_x
    
    def init_Adjency_Degree_matrix2(self, x, _dir = None):
        if x.get_device() == -1:
            device = "cpu"
        else:
            device = x.get_device()
        
        x = self.pad(x)
        select = torch.arange(x.shape[2] * x.shape[3]).reshape(x.shape[2], x.shape[3])[1:-1, 1:-1].reshape(-1)
    
        unfold = nn.Unfold(3 + (self.distance - 1) * 2, padding = self.distance)
        fold = nn.Fold(x.shape[2], 3 + (self.distance - 1) * 2, padding = self.distance)
        base = torch.zeros([x.shape[0], 1, x.shape[2], x.shape[3]]).to(device)
        adjency = []
        with torch.no_grad():
            adjency_base = unfold(base).permute(0, 2, 1)
            num = x.shape[2] * x.shape[3]
            for i in range(num):
                adjency_base = unfold(base).permute(0, 2, 1)
                if _dir is None:
                    adjency_base[:, i, :] = 1
                else:
                    adjency_base[:, i, _dir] = 1
                adjency_base[:, i, adjency_base.shape[2] // 2] = 1
                adjency_base = fold(adjency_base.permute(0, 2, 1))
                adjency.append(adjency_base[:, :, 1 : -1, 1 : -1].reshape(x.shape[0], -1))
            adjency = torch.stack(adjency, dim = 1)
        
        self.A = adjency[:, select, :]

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
            cha_cor = self.correlation(x)
            self.A = self.relu(cha_cor)
            self.A = torch.where(self.A > self.threshold,
                                 torch.tensor([1.]).to(device),
                                 torch.tensor([0.]).to(device))
            self.D = torch.diag_embed(torch.sum(self.A, dim = 2))
            D_inv_sqrt = torch.inverse(torch.sqrt(self.D))
            self.laplacian = torch.torch.bmm(torch.bmm(D_inv_sqrt, self.A), D_inv_sqrt)
            
    def __init_matrix__(self, x, _dir = None):
        batch, channel, height, width = x.shape
        if self.channel_feature:
            x_linear = x.view(batch, channel, -1).permute(0, 2, 1)
            if self.grid_adjacency:
                self.init_Adjency_Degree_matrix2(x, _dir)
            else:
                self.init_Adjency_Degree_matrix(x)
            
        else:
            x_linear = x.view(batch, channel, -1)
            self.init_Adjency_Degree_matrix(x)
        return x_linear
            
    def forward(self, x):
        
        if x.get_device() == -1:
            device = "cpu"
        else:
            device = x.get_device()
        
        batch, channel, height, width = x.shape
        if self.direction is None:
            x_linear = self.__init_matrix__(x)
            
            for i in range(self.layer):
                lx = torch.bmm(self.laplacian, x_linear)
                self._W = self.W[i].repeat(batch, 1, 1)
                x_linear = torch.matmul(lx, self._W)
                x_linear = torch.nn.functional.instance_norm(x_linear)
                x_linear = self.relu(x_linear)
        else:
            x_result = []
            for step, _dir in enumerate(self.direction):
                x_linear = self.__init_matrix__(x, _dir)
                for i in range(self.layer):
                    lx = torch.bmm(self.laplacian, x_linear)
                    self._W = self.W[step][i].repeat(batch, 1, 1).to(device)
                    x_linear = torch.matmul(lx, self._W)
                    x_linear = torch.nn.functional.instance_norm(x_linear)
                    x_linear = self.relu(x_linear)
                
                x_result.append(x_linear)
            x_result = torch.stack(x_result, dim = 2)
            x_linear = x_result.reshape(batch, x_result.shape[1], -1)                   
        return x_linear
    
if __name__ == '__main__':
    torch.manual_seed(0)
    a = torch.randn([1, 3, 4, 4]).cuda()
    #a = torch.arange(25, dtype = torch.float).reshape([1, 1, 5, 5])
    #a = torch.cat([a, a, a], dim = 1)
    gnn = GNN([3,2,2], dist = 1, power = 1, grid_adjacency = True, channel_feature = True, direction = [0, 1, 2, 3]).cuda()
    
    a = gnn(a)
    print(a)

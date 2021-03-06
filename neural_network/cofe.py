#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:25:04 2020
@author: eddie
"""
import torch.nn as nn
import torch
import numpy as np

kernel_sample = {1 : [0], 
                 3 : [0, 1, 2, 3, 4], 
                 5 : [0, 2, 4, 10, 12]}

class cofeature_fast(nn.Module):
    def __init__(self, kernel_size = 3, stride = 1, dilate = 1, pad = 'reflect'):
        super(cofeature_fast, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilate = dilate
        self.relu = nn.ReLU()
        if pad == 'reflect':
            self.pad = nn.ReplicationPad2d(kernel_size // 2)
        
    def forward(self, x, y = None):
        x = self.pad(x)
        if y is not None:
            y = self.pad(y)
            
        batch, channel, height, width = x.size()

        center_idxs = [[],[]]
        for y_idx in range(self.kernel_size//2 + self.dilate - 1, height - self.kernel_size//2 - self.dilate + 1, self.stride):
            for x_idx in range(self.kernel_size//2 + self.dilate - 1, width - self.kernel_size//2 - self.dilate + 1, self.stride):
                center_idxs[0].append(y_idx)
                center_idxs[1].append(x_idx)
                
        center_idxs = np.asarray(center_idxs)
        kernel_count = center_idxs.shape[1]

        center_vector = x[:,:,center_idxs[0],center_idxs[1]]

        center_vector = torch.transpose(center_vector, 1, 2)
        center_vector = center_vector.contiguous().view(batch * kernel_count, channel, 1)

        cofe = []
        for y_idx in range(-(self.kernel_size//2 + self.dilate)+1, (self.kernel_size//2 + self.dilate - 1)+1, self.dilate):
            for x_idx in range(-(self.kernel_size//2 + self.dilate)+1, (self.kernel_size//2 + self.dilate - 1)+1, self.dilate):
                #if (y_idx + self.kernel_size//2) * self.kernel_size + x_idx + self.kernel_size//2 <= self.kernel_size * self.kernel_size // 2:
                if (y_idx + self.kernel_size//2) * self.kernel_size + x_idx + self.kernel_size//2 in kernel_sample[self.kernel_size]:
                    if y is not None:
                        side_vector = y[:,:,center_idxs[0]+y_idx, center_idxs[1]+x_idx]
                    else:
                        side_vector = x[:,:,center_idxs[0]+y_idx, center_idxs[1]+x_idx]
                        
                    side_vector = side_vector.transpose(1,2)
                    side_vector = side_vector.contiguous().view(-1, channel, 1)
                    side_vector_t = side_vector.permute(0, 2, 1)
                    
                    # calculate side_vector and center_vector similarity
                    A = torch.sum(side_vector * side_vector, dim = 1, keepdims = False)
                    B = torch.sum(center_vector * center_vector, dim = 1, keepdims = False)
                    similarity = side_vector.squeeze(2) * center_vector.squeeze(2) / torch.sqrt(A) / torch.sqrt(B)
                    similarity = torch.sum(similarity, dim = 1)
                    similarity = self.relu(similarity)
                    # original cofe
                    cofeature = torch.bmm(center_vector, side_vector_t) * similarity.unsqueeze(1).unsqueeze(1)
                    cofeature = cofeature.view(batch, kernel_count, -1)
                    cofeature = torch.sum(cofeature, dim=1, keepdim=False)
                    cofe.append(cofeature)
                    
        cofe = torch.stack(cofe)
        cofe = cofe.transpose(0,1)
        #cofe = cofe.contiguous().view(cofe.shape[0], -1, cofe.shape[3])
        cofe = nn.functional.normalize(cofe, dim=-1)
        return cofe
    
if __name__ == '__main__':
    c = cofeature_fast(3)
    a = torch.randn([2, 128, 14, 14])
    cofe = c(a)
    print(cofe.shape)
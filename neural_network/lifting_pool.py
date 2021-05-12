
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:51:49 2021
@author: eddie
"""
import numpy as np
import torch
import torch.nn as nn

class Lifting_down(nn.Module):
    def __init__(self, channel, kernel_size = 2, stride = None, pad_mode = 'discard', pad_place = [0, 1, 0, 1]):
        super(Lifting_down, self).__init__()
        self.pad_mode = pad_mode
        self.pad_place = pad_place
        self.kernel_size = kernel_size
        self.channel = channel
        self.stride = stride
        if self.stride is None:
            self.stride = kernel_size
        
        self.low_pass_filter_h = torch.nn.Parameter(torch.rand(channel, 1, 1, self.kernel_size))
        self.high_pass_filter_h = torch.nn.Parameter(torch.rand(channel, 1, 1, self.kernel_size))
        self.low_pass_filter_v = torch.nn.Parameter(torch.rand(channel, 1, self.kernel_size, 1))
        self.high_pass_filter_v = torch.nn.Parameter(torch.rand(channel, 1, self.kernel_size, 1))
        self.squeeze = nn.Conv2d(channel * 4, channel, 1, bias = False)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.SE = nn.Sequential(nn.Linear(channel, channel // 2),
                                nn.ReLU(),
                                nn.Linear(channel // 2, channel),
                                nn.Sigmoid())
        #self.filter_constraint()
    
    def __repr__(self):
        struct = "Lifting({}, kernel_size={}, stride={})".format(self.channel, self.kernel_size, self.stride)
        return struct
    
    def regular_term_loss(self):
        # low pass filter sum = 1
        constraint1 = torch.sum(torch.pow(torch.sum(self.low_pass_filter_h, dim = 3, keepdim = True) - 1, 2) +\
                                torch.pow(torch.sum(self.low_pass_filter_v, dim = 2, keepdim = True) - 1, 2), dim = 0).squeeze(-1)
        # high pass filter sum = 0 & sum((1 - weight) ** 2) = 0 => limit high pass to unit length
# =============================================================================
#         constraint2 = torch.sum(torch.sum(self.high_pass_filter_h, dim = 3) + torch.sum(self.high_pass_filter_v, dim = 2) +\
#             torch.sum(torch.pow(1 - self.high_pass_filter_h, 2), dim = 3) +\
#             torch.sum(torch.pow(1 - self.high_pass_filter_v, 2), dim = 2), dim = 0).squeeze(-1)
# =============================================================================
        constraint2 = torch.sum(torch.pow(1 - torch.sum(torch.pow(self.high_pass_filter_h, 2), dim = 3), 2) +\
                       torch.pow(1 - torch.sum(torch.pow(self.high_pass_filter_v, 2), dim = 2), 2), dim = 0).squeeze(-1)
        return (constraint1 + constraint2).squeeze(-1).squeeze(-1)
    
    # need call filter_constraint every step after optimizer.step() to make sure the weight is in constraint
    def filter_constraint(self):
        self.low_pass_filter_h.data = self.low_pass_filter_h / torch.sum(self.low_pass_filter_h, dim = 3, keepdim = True)
        self.high_pass_filter_h.data = self.high_pass_filter_h - torch.mean(self.high_pass_filter_h, dim = 3, keepdim = True)
        self.low_pass_filter_v.data = self.low_pass_filter_v / torch.sum(self.low_pass_filter_v, dim = 2, keepdim = True)
        self.high_pass_filter_v.data = self.high_pass_filter_v - torch.mean(self.high_pass_filter_v, dim = 2, keepdim = True)
    
    def energy_filter(self, x):
        batch, channel, height, width = x.shape
        x_energe = torch.mean(torch.mean(torch.pow(x, 2), dim = -1), dim = -1)
        x_energe_index = torch.argsort(-x_energe, dim = 1)
        x_energe_index = x_energe_index[:, :channel // 4 * 3]#:x_energe_index.shape[1] // 2]
        x_energe_index, _ = torch.sort(x_energe_index)
        x = x.reshape(-1, height, width)
        x_energe_index = x_energe_index + torch.arange(0, batch).reshape(-1, 1).to(x.get_device()) * channel
        x = x[x_energe_index.reshape(-1)]
        x = x.reshape(batch, -1, height, width)
        return x
    
    def attention(self, x):
        x_att = torch.mean(torch.mean(torch.pow(x, 2), dim = -1), dim = -1)
        #x_att = self.avg(x).squeeze(-1).squeeze(-1)
        x_att = self.SE(x_att)
        x = x * x_att.unsqueeze(-1).unsqueeze(-1) + x
        return x
    
    def forward(self, x):
        # pad the feature map
        batch, channel, height, width = x.shape
        if self.pad_mode == 'discard':
            x = x[:, :, :height - height % self.kernel_size, :width - width % self.kernel_size]
        elif self.pad_mode == 'pad0':
            x = torch.nn.functional.pad(x, pad = self.pad_place, mode = 'constant', value = 0)
        else:
            x = torch.nn.functional.pad(x, pad = self.pad_place, mode = self.pad_mode)
            
        assert self.low_pass_filter_h.shape[0] == x.shape[1], "low pass filter_h wrong channel number."
        assert self.high_pass_filter_h.shape[0] == x.shape[1], "high pass filter_h wrong channel number."
        assert self.low_pass_filter_v.shape[0] == x.shape[1], "low pass filter_v wrong channel number."
        assert self.high_pass_filter_v.shape[0] == x.shape[1], "high pass filter_v wrong channel number."
            
        # calculate the lifting weight different weight
        x_l = torch.nn.functional.conv2d(x, self.low_pass_filter_h, groups = x.shape[1], stride = (1, self.stride))
        x_h = torch.nn.functional.conv2d(x, self.high_pass_filter_h, groups = x.shape[1], stride = (1, self.stride))
        x_ll = torch.nn.functional.conv2d(x_l, self.low_pass_filter_v, groups = x_l.shape[1], stride = (self.stride, 1))
        x_hl = torch.nn.functional.conv2d(x_l, self.high_pass_filter_v, groups = x_l.shape[1], stride = (self.stride, 1))
        x_lh = torch.nn.functional.conv2d(x_h, self.low_pass_filter_v, groups = x_l.shape[1], stride = (self.stride, 1))
        x_hh = torch.nn.functional.conv2d(x_h, self.high_pass_filter_v, groups = x_l.shape[1], stride = (self.stride, 1))
        del x_l
        del x_h
        
        x_all = torch.cat([x_ll, x_hl, x_lh, x_hh], dim = 1)
        x_all = self.squeeze(x_all)
        x_all = self.attention(x_all)
        
        return x_all

def lifting_down(img, pad_mode = 'discard', pad_place = [0, 1, 0, 1]):
    if pad_mode == 'discard':
        img = img[:, :, :img.shape[2] // 2 * 2, :img.shape[3] // 2 * 2]
    elif pad_mode == 'pad0':
        img = torch.nn.functional.pad(img, pad = pad_place, mode = 'constant', value = 0)
    else:
        img = torch.nn.functional.pad(img, pad = pad_place, mode = pad_mode)

    h_img_odd = img[:, :, 1::2, :]
    h_img_even = img[:, :, 0::2, :]
    l = torch.div((h_img_even + h_img_odd), 2)
    h = torch.div((h_img_even - h_img_odd), 2)
    ll = torch.div((l[:, :, :, 0::2] + l[:, :, :, 1::2]), 2)
    hl = torch.div((l[:, :, :, 0::2] - l[:, :, :, 1::2]), 2)
    lh = torch.div((h[:, :, :, 0::2] + h[:, :, :, 1::2]), 2)
    hh = torch.div((h[:, :, :, 0::2] - h[:, :, :, 1::2]), 2)
    return ll, hl, lh, hh
    
def lifting_up(ll, hl, lh, hh):
    
    result_m = np.zeros([ll.shape[0], ll.shape[1] * 2, ll.shape[2] * 2, ll.shape[3]])
    map1 = ll + hl
    map2 = ll - hl
    map3 = lh + hh
    map4 = lh - hh
    
    map11 = map1 + map3
    map22 = map1 - map3
    map33 = map2 + map4
    map44 = map2 - map4
    
    result_m[:, 0::2, 0::2, :] = map11
    result_m[:, 0::2, 1::2, :] = map22
    result_m[:, 1::2, 0::2, :] = map33
    result_m[:, 1::2, 1::2, :] = map44
    return result_m

if __name__ == "__main__":
    # test 1    
# =============================================================================
#     image = torch.tensor([[[[30],[12], [16], [20]],
#                       [[28], [2], [18], [2]], 
#                       [[10],[12],[14],[16]],
#                       [[18],[20], [22], [24]]]], dtype = torch.float)
#     image = image.reshape(1, 1, 4, 4)
#     pool = Lifting_down(1)
#     x_ll, x_hl, x_lh, x_hh = pool(image)
#     pool.filter_constraint()
# =============================================================================
    
    # test 2
    image = torch.randn([2, 256, 4, 4]).cuda()
    pool = Lifting_down(256, kernel_size = 2).cuda()
    output = pool(image)
    print(pool.regular_term_loss() * 1e-4)
    pool.filter_constraint()
# =============================================================================
#     ll, hl, lh, hh = lifting_down(image, pad_mode = 'discard')
#     lifting_up(ll, hl, lh, hh)
# =============================================================================
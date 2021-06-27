
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:51:49 2021
@author: eddie
"""
import numpy as np
import torch
import torch.nn as nn

class LDW_down(nn.Module):
    def __init__(self, kernel_size = 2, stride = None):
        super(LDW_down, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 2
        
# =============================================================================
#         self.low_pass_filter_h_down = torch.nn.Parameter(torch.tensor([[[[0.5, 0.5]]]]))
#         self.high_pass_filter_h_down = torch.nn.Parameter(torch.tensor([[[[0.5, -0.5]]]]))
#         self.low_pass_filter_v_down = torch.nn.Parameter(torch.tensor([[[[0.5],
#                                                                    [0.5]]]]))
#         self.high_pass_filter_v_down = torch.nn.Parameter(torch.tensor([[[[0.5],
#                                                                     [-0.5]]]]))
#         self.low_pass_filter_h_up = torch.nn.Parameter(torch.tensor([[[[1., 1.]]]]))
#         self.high_pass_filter_h_up = torch.nn.Parameter(torch.tensor([[[[1., -1.]]]]))
#         self.low_pass_filter_v_up = torch.nn.Parameter(torch.tensor([[[[1.],
#                                                                    [1.]]]]))
#         self.high_pass_filter_v_up = torch.nn.Parameter(torch.tensor([[[[1.],
#                                                                     [-1.]]]]))
# =============================================================================
        self.low_pass_filter_h = torch.nn.Parameter(torch.rand(1, 1, 1, self.kernel_size))
        self.high_pass_filter_h = torch.nn.Parameter(torch.rand(1, 1, 1, self.kernel_size))
        self.low_pass_filter_v = torch.nn.Parameter(torch.rand(1, 1, self.kernel_size, 1))
        self.high_pass_filter_v = torch.nn.Parameter(torch.rand(1, 1, self.kernel_size, 1))
        #self.filter_constraint()
    
    def __repr__(self):
        struct = "Lifting_down(kernel_size={}, stride={})".format(self.kernel_size, self.stride)
        return struct
    
    def regular_term_loss(self):
        # low pass filter sum = 1
        constraint1 = torch.mean(torch.pow(torch.sum(self.low_pass_filter_h, dim = 3, keepdim = True) - 1, 2) +\
                                torch.pow(torch.sum(self.low_pass_filter_v, dim = 2, keepdim = True) - 1, 2), dim = 0).squeeze(-1)
        # high pass filter sum = 0 & sum((1 - weight) ** 2) = 0 => limit high pass to unit length
        constraint2 = torch.mean(torch.pow(1 - torch.sum(torch.pow(self.high_pass_filter_h, 2), dim = 3), 2) +\
                       torch.pow(1 - torch.sum(torch.pow(self.high_pass_filter_v, 2), dim = 2), 2) +\
                           torch.pow(torch.sum(self.high_pass_filter_h, dim = 3), 2) + torch.pow(torch.sum(self.high_pass_filter_v, dim = 2), 2), dim = 0).squeeze(-1)
            
        # constraint3 => H'H + L'L = 1
        vertical_sum = torch.sum(torch.pow(self.low_pass_filter_v, 2).squeeze(-1), dim = 2) + torch.sum(torch.pow(self.high_pass_filter_v, 2).squeeze(-1), dim = 2)
        horizontal_sum = torch.sum(torch.pow(self.low_pass_filter_h, 2).squeeze(2), dim = 2) + torch.sum(torch.pow(self.high_pass_filter_h, 2).squeeze(2), dim = 2)
        constraint3 = torch.mean(torch.pow(1 - vertical_sum, 2) + torch.pow(1 - horizontal_sum, 2), dim = 0)
        
        return (constraint1 + constraint2 + constraint3).squeeze(-1).squeeze(-1)
    
    def switch_data(self, x, y, dim):
        if x.get_device() == -1:
            device = "cpu"
        else:
            device = x.get_device()
        combine = torch.cat([x, y], dim = dim)
        idx_old = torch.arange(0, x.shape[dim]).to(device)
        idx_old = idx_old.repeat_interleave(2)
        idx_old[1::2] += x.shape[dim]
        if dim == 2:
            combine = combine[:, :, idx_old, :]
        elif dim == 3:
            combine = combine[:, :, :, idx_old]
        return combine
    
    def up(self, x):
        if x.get_device() == -1:
            device = "cpu"
        else:
            device = x.get_device()
        # pad the feature map
        batch, channel, height, width = x.shape
        
        # reconstruct ll + hl = l
        x_l_combine = self.switch_data(x[:, 0:channel // 4, :, :], x[:, channel // 4 : channel // 2, :, :], 2)
        x_l_combine = torch.nn.functional.pad(x_l_combine,
                                              pad = [0, 0, self.kernel_size // 2, self.kernel_size // 2],
                                              mode = 'reflect')
        x_l_e = torch.nn.functional.conv2d(x_l_combine,
                                         self.low_pass_filter_v.repeat(x_l_combine.shape[1], 1, 1, 1),
                                         groups = channel // 4,
                                         stride = (2, 1))
        x_l_o = torch.nn.functional.conv2d(x_l_combine[:, :, 1:, :],
                                         self.high_pass_filter_v.repeat(x_l_combine.shape[1], 1, 1, 1),
                                         groups = channel // 4,
                                         stride = (2, 1))
        x_l = self.switch_data(x_l_e, x_l_o, 2)
        
        # reconstruct lh + hh = h
        x_h_combine = self.switch_data(x[:, channel // 2 : channel // 4 * 3, :, :], x[:, channel // 4 * 3 : , :, :], 2)
        x_h_combine = torch.nn.functional.pad(x_h_combine,
                                              pad = [0, 0, self.kernel_size // 2, self.kernel_size // 2],
                                              mode = 'reflect')
        x_h_e = torch.nn.functional.conv2d(x_h_combine,
                                         self.low_pass_filter_v.repeat(x_h_combine.shape[1], 1, 1, 1),
                                         groups = channel // 4,
                                         stride = (2, 1))
        x_h_o = torch.nn.functional.conv2d(x_h_combine,
                                         self.high_pass_filter_v.repeat(x_h_combine.shape[1], 1, 1, 1),
                                         groups = channel // 4,
                                         stride = (2, 1))
        x_h = self.switch_data(x_h_e, x_h_o, 2)
        
        # reconstruct l + h = x
        x_combine = self.switch_data(x_l, x_h, 3)
        
        x_e = torch.nn.functional.conv2d(x_combine,
                                         self.low_pass_filter_h.repeat(x_h_combine.shape[1], 1, 1, 1),
                                         groups = channel // 4,
                                         stride = (1, 2),
                                         padding = [0, self.kernel_size // 2 if self.kernel_size != 2 else 0])
        x_o = torch.nn.functional.conv2d(x_combine,
                                         self.high_pass_filter_h.repeat(x_h_combine.shape[1], 1, 1, 1),
                                         groups = channel // 4,
                                         stride = (1, 2),
                                         padding = [0, self.kernel_size // 2 if self.kernel_size != 2 else 0])
        recover_x = self.switch_data(x_e, x_o, 3)
        return recover_x
    
    def forward(self, x):
        # pad the feature map
        batch, channel, height, width = x.shape
        x = torch.nn.functional.pad(x,
                                    pad = [self.kernel_size // 2 if self.kernel_size != 2 else 0, self.kernel_size // 2 if self.kernel_size != 2 else 0, 0, 0],
                                    mode = 'reflect')
        # calculate the lifting weight different weight
        x_l = torch.nn.functional.conv2d(x, 
                                         self.low_pass_filter_h.repeat(channel, 1, 1, 1),
                                         groups = channel, 
                                         stride = (1, self.stride))
        x_h = torch.nn.functional.conv2d(x[:, :, :, 1:], 
                                         self.high_pass_filter_h.repeat(channel, 1, 1, 1),
                                         groups = channel, 
                                         stride = (1, self.stride))
        
        x_l = torch.nn.functional.pad(x_l,
                                    pad = [0, 0, self.kernel_size // 2 if self.kernel_size != 2 else 0, self.kernel_size // 2 if self.kernel_size != 2 else 0],
                                    mode = 'reflect')
        x_h = torch.nn.functional.pad(x_h,
                                    pad = [0, 0, self.kernel_size // 2 if self.kernel_size != 2 else 0, self.kernel_size // 2 if self.kernel_size != 2 else 0],
                                    mode = 'reflect')
        
        
        x_ll = torch.nn.functional.conv2d(x_l, 
                                          self.low_pass_filter_v.repeat(channel, 1, 1, 1), 
                                          groups = x_l.shape[1], 
                                          stride = (self.stride, 1))
        x_hl = torch.nn.functional.conv2d(x_l[:, :, 1:, :], 
                                          self.high_pass_filter_v.repeat(channel, 1, 1, 1), 
                                          groups = x_l.shape[1], 
                                          stride = (self.stride, 1))
        x_lh = torch.nn.functional.conv2d(x_h, 
                                          self.low_pass_filter_v.repeat(channel, 1, 1, 1), 
                                          groups = x_h.shape[1], 
                                          stride = (self.stride, 1))
        x_hh = torch.nn.functional.conv2d(x_h[:, :, 1:, :], 
                                          self.high_pass_filter_v.repeat(channel, 1, 1, 1), 
                                          groups = x_h.shape[1], 
                                          stride = (self.stride, 1))
        del x_l
        del x_h
        
        x_all = torch.cat([x_ll, x_hl, x_lh, x_hh], dim = 1)
        return x_all
    
class Energy_attention(nn.Module):
    def __init__(self, in_cha):
        super(Energy_attention, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(in_cha)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.SE = nn.Sequential(nn.Linear(in_cha, in_cha // 4),
                                nn.BatchNorm1d(in_cha // 4),
                                nn.ReLU(inplace = True),
                                nn.Linear(in_cha // 4, in_cha),
                                nn.Sigmoid())
        
    def forward(self, x):
        x_norm = self.instance_norm(x)
        x_energy = self.avgpool(torch.pow(x_norm, 2)).squeeze(-1).squeeze(-1)
        x_energy = self.SE(x_energy)
        x = x * x_energy.unsqueeze(-1).unsqueeze(-1) + x
        
        return x

def lifting_down(img, pad_mode = 'discard', pad_place = [0, 1, 0, 1]):
    if pad_mode == 'discard':
        img = img[:, :, :img.shape[2] // 2 * 2, :img.shape[3] // 2 * 2]
    elif pad_mode == 'pad0':
        img = torch.nn.functional.pad(img, pad = pad_place, mode = 'constant', value = 0)
    else:
        img = torch.nn.functional.pad(img, pad = pad_place, mode = pad_mode)

    h_img_odd = img[:, :, :, 1::2]
    h_img_even = img[:, :, :, 0::2]
    l = torch.div((h_img_even + h_img_odd), 2)
    h = torch.div((h_img_even - h_img_odd), 2)
    ll = torch.div((l[:, :, 0::2, :] + l[:, :, 1::2, :]), 2)
    hl = torch.div((l[:, :, 0::2, :] - l[:, :, 1::2, :]), 2)
    lh = torch.div((h[:, :, 0::2, :] + h[:, :, 1::2, :]), 2)
    hh = torch.div((h[:, :, 0::2, :] - h[:, :, 1::2, :]), 2)
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
#     image = image.reshape(1, 1, 4, 4).cuda()
# =============================================================================
    image = torch.randn(1, 1, 8, 8)
    x_ll, x_hl, x_lh, x_hh = lifting_down(image)
    # test 2
    #image = torch.randn([2, 4, 8, 8]).cuda()
    pool_down = LDW_down(kernel_size = 3)
    print("image : ", image.shape)
    output = pool_down(image)
    print("output : ", output.shape)
    image = pool_down.up(output)
    print("image : ", image.shape)
    #print(pool_down.regular_term_loss())
# =============================================================================
#     ll, hl, lh, hh = lifting_down(image, pad_mode = 'discard')
#     lifting_up(ll, hl, lh, hh)
# =============================================================================
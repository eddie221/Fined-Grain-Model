
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
    def __init__(self, channel, kernel_size = 2, stride = None, pad_mode = 'discard', pad_place = [0, 1, 0, 1]):
        super(LDW_down, self).__init__()
        self.pad_mode = pad_mode
        self.pad_place = pad_place
        self.kernel_size = kernel_size
        self.channel = channel
        self.stride = stride
        
        if self.stride is None:
            self.stride = kernel_size
        
        self.instance_norm1 = torch.nn.InstanceNorm2d(channel)
# =============================================================================
#         self.low_pass_filter_h = torch.nn.Parameter(torch.tensor([[[[0.5, 0.5]]]]))
#         self.high_pass_filter_h = torch.nn.Parameter(torch.tensor([[[[0.5, -0.5]]]]))
#         self.low_pass_filter_v = torch.nn.Parameter(torch.tensor([[[[0.5],
#                                                                    [0.5]]]]))
#         self.high_pass_filter_v = torch.nn.Parameter(torch.tensor([[[[0.5],
#                                                                     [-0.5]]]]))
# =============================================================================
        self.low_pass_filter_h = torch.nn.Parameter(torch.rand(channel, 1, 1, self.kernel_size))
        self.high_pass_filter_h = torch.nn.Parameter(torch.rand(channel, 1, 1, self.kernel_size))
        self.low_pass_filter_v = torch.nn.Parameter(torch.rand(channel, 1, self.kernel_size, 1))
        self.high_pass_filter_v = torch.nn.Parameter(torch.rand(channel, 1, self.kernel_size, 1))
        #self.filter_constraint()
    
    def __repr__(self):
        struct = "Lifting({}, kernel_size={}, stride={})".format(self.channel, self.kernel_size, self.stride)
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
    
    def forward(self, x):
        # pad the feature map
        batch, channel, height, width = x.shape
        if self.pad_mode == 'discard':
            x = x[:, :, :height - height % self.kernel_size, :width - width % self.kernel_size]
        elif self.pad_mode == 'pad0':
            x = torch.nn.functional.pad(x, pad = self.pad_place, mode = 'constant', value = 0)
        else:
            x = torch.nn.functional.pad(x, pad = self.pad_place, mode = self.pad_mode)
            
        assert self.low_pass_filter_h.shape[0] == x.shape[1], "low pass filter_h wrong channel number. low pass filter_h get {}. x get {}".format(self.low_pass_filter_h.shape[0], x.shape[1])
        assert self.high_pass_filter_h.shape[0] == x.shape[1], "high pass filter_h wrong channel number."
        assert self.low_pass_filter_v.shape[0] == x.shape[1], "low pass filter_v wrong channel number."
        assert self.high_pass_filter_v.shape[0] == x.shape[1], "high pass filter_v wrong channel number."
        
        # calculate the lifting weight different weight
        x_l = torch.nn.functional.conv2d(x, self.low_pass_filter_h, groups = x.shape[1], stride = (1, self.stride))
        x_h = torch.nn.functional.conv2d(x, self.high_pass_filter_h, groups = x.shape[1], stride = (1, self.stride))

        x_ll = torch.nn.functional.conv2d(x_l, self.low_pass_filter_v, groups = x_l.shape[1], stride = (self.stride, 1))
        x_hl = torch.nn.functional.conv2d(x_l, self.high_pass_filter_v, groups = x_l.shape[1], stride = (self.stride, 1))
        x_lh = torch.nn.functional.conv2d(x_h, self.low_pass_filter_v, groups = x_h.shape[1], stride = (self.stride, 1))
        x_hh = torch.nn.functional.conv2d(x_h, self.high_pass_filter_v, groups = x_h.shape[1], stride = (self.stride, 1))
        del x_l
        del x_h
        
        x_all = torch.cat([x_ll, x_hl, x_lh, x_hh], dim = 1)
        return x_all
    
class LDW_up(nn.Module):
    def __init__(self, channel, kernel_size = 2, stride = None, pad_mode = 'discard', pad_place = [0, 1, 0, 1]):
        super(LDW_up, self).__init__()
        self.pad_mode = pad_mode
        self.pad_place = pad_place
        self.kernel_size = kernel_size
        self.channel = channel
        self.stride = stride
        
        if self.stride is None:
            self.stride = kernel_size
        
        self.conv1x1 = torch.nn.Conv2d(channel, channel, 1, bias = False)
        self.instance_norm1 = torch.nn.InstanceNorm2d(channel)
# =============================================================================
#         self.low_pass_filter_h = torch.nn.Parameter(torch.tensor([[[[0.5, 0.5]]]]))
#         self.high_pass_filter_h = torch.nn.Parameter(torch.tensor([[[[0.5, -0.5]]]]))
#         self.low_pass_filter_v = torch.nn.Parameter(torch.tensor([[[[0.5],
#                                                                    [0.5]]]]))
#         self.high_pass_filter_v = torch.nn.Parameter(torch.tensor([[[[0.5],
#                                                                     [-0.5]]]]))
# =============================================================================
        self.low_pass_filter_h = torch.nn.Parameter(torch.rand(channel // 4, 1, 1, self.kernel_size))
        self.high_pass_filter_h = torch.nn.Parameter(torch.rand(channel // 4, 1, 1, self.kernel_size))
        self.low_pass_filter_v = torch.nn.Parameter(torch.rand(channel // 4, 1, self.kernel_size, 1))
        self.high_pass_filter_v = torch.nn.Parameter(torch.rand(channel // 4, 1, self.kernel_size, 1))
        #self.filter_constraint()
    
    def __repr__(self):
        struct = "Lifting({}, kernel_size={}, stride={})".format(self.channel, self.kernel_size, self.stride)
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
    
    # need call filter_constraint every step after optimizer.step() to make sure the weight is in constraint
# =============================================================================
#     def filter_constraint(self):
#         self.low_pass_filter_h.data = self.low_pass_filter_h / torch.sum(self.low_pass_filter_h, dim = 3, keepdim = True)
#         self.high_pass_filter_h.data = self.high_pass_filter_h - torch.mean(self.high_pass_filter_h, dim = 3, keepdim = True)
#         self.low_pass_filter_v.data = self.low_pass_filter_v / torch.sum(self.low_pass_filter_v, dim = 2, keepdim = True)
#         self.high_pass_filter_v.data = self.high_pass_filter_v - torch.mean(self.high_pass_filter_v, dim = 2, keepdim = True)
# =============================================================================
    
    def forward(self, x):
        x = self.conv1x1(x)
        # pad the feature map
        batch, channel, height, width = x.shape
        if self.pad_mode == 'discard':
            x = x[:, :, :height - height % self.kernel_size, :width - width % self.kernel_size]
        elif self.pad_mode == 'pad0':
            x = torch.nn.functional.pad(x, pad = self.pad_place, mode = 'constant', value = 0)
        else:
            x = torch.nn.functional.pad(x, pad = self.pad_place, mode = self.pad_mode)

        # calculate the lifting weight different weight
        x_l_e = torch.nn.functional.conv2d(torch.cat([x[:, 0:channel // 4, :, :], x[:, channel // 4 : channel // 2, :, :]], dim = 2),
                                           self.low_pass_filter_v,
                                           groups = channel // 4,
                                           stride = (1, 1),
                                           dilation = (x.shape[2], 1))
        x_l_o = torch.nn.functional.conv2d(torch.cat([x[:, 0:channel // 4, :, :], x[:, channel // 4 : channel // 2, :, :]], dim = 2),
                                           self.high_pass_filter_v, 
                                           groups = channel // 4, 
                                           stride = (1, 1),
                                           dilation = (x.shape[2], 1))
        x_h_e = torch.nn.functional.conv2d(torch.cat([x[:, channel // 2 : channel * 3 // 4, :, :], x[:, channel * 3 // 4 :, :, :]], dim = 2), 
                                           self.low_pass_filter_v, 
                                           groups = channel // 4, 
                                           stride = (1, 1),
                                           dilation = (x.shape[2], 1))
        x_h_o = torch.nn.functional.conv2d(torch.cat([x[:, channel // 2 : channel * 3 // 4, :, :], x[:, channel * 3 // 4 :, :, :]], dim = 2), 
                                           self.high_pass_filter_v,
                                           groups = channel // 4,
                                           stride = (1, 1),
                                           dilation = (x.shape[2], 1))
        
        low_idx_old = torch.arange(0, x_l_o.shape[2]).to(x.get_device())
        low_idx_old = low_idx_old.repeat_interleave(2)
        low_idx_old[1::2] += x_l_o.shape[2]
        x_l = torch.cat([x_l_e, x_l_o], dim = 2)
        x_l = x_l[:, :, low_idx_old, :]
        
        high_idx_old = torch.arange(0, x_h_o.shape[2]).to(x.get_device())
        high_idx_old = high_idx_old.repeat_interleave(2)
        high_idx_old[1::2] += x_h_o.shape[2]
        x_h = torch.cat([x_h_e, x_h_o], dim = 2)
        x_h = x_h[:, :, high_idx_old, :]
        
        x_e = torch.nn.functional.conv2d(torch.cat([x_l, x_h], dim = 3),
                                         self.low_pass_filter_h,
                                         groups = channel // 4,
                                         stride = (1, 1),
                                         dilation = (1, x.shape[2]))
        x_o = torch.nn.functional.conv2d(torch.cat([x_l, x_h], dim = 3),
                                         self.high_pass_filter_h, 
                                         groups = channel // 4,
                                         stride = (1, 1),
                                         dilation = (1, x.shape[2]))
        idx_old = torch.arange(0, x.shape[3]).to(x.get_device())
        idx_old = idx_old.repeat_interleave(2)
        idx_old[1::2] += x_h_o.shape[2]
        recover_x = torch.cat([x_e, x_o], dim = 3)
        recover_x = recover_x[:, :, :, idx_old]
        
        return recover_x
    
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
        x_energy = torch.mean(torch.mean(torch.pow(x_norm, 2), dim = -1), dim = -1)
        
        x_energy = self.SE(x_energy)
        x = x * x_energy.unsqueeze(-1).unsqueeze(-1)
        
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
    image = torch.tensor([[[[30],[12], [16], [20]],
                      [[28], [2], [18], [2]], 
                      [[10],[12],[14],[16]],
                      [[18],[20], [22], [24]]]], dtype = torch.float)
    image = image.reshape(1, 1, 4, 4).cuda()
    x_ll, x_hl, x_lh, x_hh = lifting_down(image)
    # test 2
    #image = torch.randn([2, 4, 8, 8]).cuda()
    pool_down = Lifting_down(1, kernel_size = 2).cuda()
    pool_up = Lifting_up(4, kernel_size = 2).cuda()
    output = pool_down(image)
    print("output : ", output)
    image = pool_up(output)
    print("image : ", image)
    print(pool_down.regular_term_loss())
# =============================================================================
#     ll, hl, lh, hh = lifting_down(image, pad_mode = 'discard')
#     lifting_up(ll, hl, lh, hh)
# =============================================================================
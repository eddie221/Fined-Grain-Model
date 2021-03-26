#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:51:49 2021

@author: eddie
"""
import numpy as np
import torch

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
    #image = Image.open("./test.png")
    
    #image = np.array(image)[:, :, :3]
    image = torch.randn([1, 3, 3, 3])
# =============================================================================
#     image = torch.tensor([[[[30],[12], [16], [20]],
#                       [[28], [2], [18], [2]], 
#                       [[10],[12],[14],[16]],
#                       [[18],[20], [22], [24]]]])
# =============================================================================
# =============================================================================
#     image = np.arange(16).reshape(4, 4, 1)    
# =============================================================================
    ll, hl, lh, hh = lifting_down(image, pad_mode = 'discard')
    lifting_up(ll, hl, lh, hh)


# =============================================================================
# count = 0
# for i in range(W // 10240 + 1):
#     for j in range(H // 10240 + 1):
#         patch = np.array(image.read_region((i * 10240, j * 10240), 0, (10240, 10240)))
#         save_img = Image.fromarray(patch)
#         save_img.save('./19-19811A3-Ki67{}.png'.format(i * H // 10240 + j))
# =============================================================================

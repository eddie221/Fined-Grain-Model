#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 09:56:53 2020

@author: eddie
"""

BATCH_SIZE = 256
IMAGE_SIZE = 224
LR = 0.1
EPOCH = 160
INDEX = '20210530-1'
REMAEK = ''
NUM_CLASS = 1000
CUTMIX_PROB = 0
CON_MATRIX = False
KFOLD = 1
if KFOLD == 1:
    VAL_SPLIT = 0.2
PRETRAIN = False

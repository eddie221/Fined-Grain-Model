#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 09:56:53 2020

@author: eddie
"""

BATCH_SIZE = 64
IMAGE_SIZE = 224
LR = 0.0001
EPOCH = 400
INDEX = '20210615-1'
REMAEK = ''
NUM_CLASS = 10
CUTMIX_PROB = 0
CON_MATRIX = False
KFOLD = 1
if KFOLD == 1:
    VAL_SPLIT = 0.2
PRETRAIN = False

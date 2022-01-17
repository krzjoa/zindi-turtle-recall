#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 22:02:59 2022

@author: krzysztof
"""

import torch.nn as nn

turtle_net_1 = nn.Sequential(
        nn.Conv2d(3, 10, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d(3, stride = 2),  
        nn.Conv2d(10, 100, (3, 3), stride=2),
        nn.ReLU(),
        nn.MaxPool2d(3, stride = 2),
        nn.Conv2d(100, 1000, (3, 3), stride=2),
        nn.ReLU(),
        nn.MaxPool2d(3, stride = 2),
        nn.Conv2d(1000, 1500, (3, 3), stride=2),
        nn.ReLU(),
        nn.Conv2d(1500, 2266, (2, 2), stride=2),
        nn.Softmax(1)
    ).cuda()
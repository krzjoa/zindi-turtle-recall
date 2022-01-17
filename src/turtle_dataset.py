#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 22:04:54 2022

@author: krzysztof
"""

import torch 
import torch.utils.data as torch_data
import torch.nn as nn
import torch.optim as optim

import torchvision.io as io
import torchvision.transforms as T

import numpy as np


def get_target_by_turtle_id(turtle_id):
    idx = all_labels.index(turtle_id)
    return get_target_vector(idx)


def get_target_vector(idx):
    vec = torch.tensor(np.zeros(2266))
    vec[idx] = 1
    return vec


class TurtleDataset(torch_data.Dataset):
    
    def __init__(self, img_labels, img_dir, 
                 img_transform = None, transform_target = None):
        self.img_labels = pd.read_csv(img_labels)
        self.img_dir = img_dir
        self.img_transform = img_transform
        self.transform_target = transform_target
   
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        # Image path
        img_id = self.img_labels.iloc[idx, 0] # First col
        img_filename = img_id + ".JPG"
        img_path = os.path.join(self.img_dir, img_filename)
        
        # Reading an image
        image = io.read_image(img_path)
        image_location = self.img_labels.iloc[idx, 1]
        turtle_id = self.img_labels.iloc[idx, 2]
        
        if self.img_transform:
            image = self.img_transform(image)
        if self.transform_target:
            label = self.transform_target(label)
        
        return (image.to(torch.float) / 255).cuda(), image_location, get_target_by_turtle_id(turtle_id).cuda()        
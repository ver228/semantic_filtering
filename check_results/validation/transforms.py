#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 17:41:46 2019

@author: avelinojaver
"""

import numpy as np
import random

import torch
import math

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[:2]
            image = image[:, ::-1]
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            
            target["boxes"] = bbox
            
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image[::-1, :]
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            target["boxes"] = bbox
            
        return image, target
    
class RandomTranspose(object):
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            
            image = image.T
            bbox = target["boxes"]
            bbox = bbox[..., [1,0,3,2]]
            target["boxes"] = bbox
            
        return image, target
    
class RandomCrop(object):
    def __init__(self, roi_size = 512, prob = 0.5):
        self.prob = prob
        self.roi_size = roi_size

    def __call__(self, image, target):
        
        bbox = target["boxes"]
        
        n_test = 100
        for _ in range(n_test):
            xl, xr = 0, image.shape[1] - self.roi_size - 1
            yl, yr = 0, image.shape[0] - self.roi_size - 1
            
            yl = max(0, math.floor(bbox[:, 2].min()) - self.roi_size)
            xl = max(0, math.floor(bbox[:, 3].min()) - self.roi_size)
            
            yr = min(yr, math.ceil(bbox[:, 0].max()))
            xr = min(xr, math.ceil(bbox[:, 1].max()))
            
            
            x = int(random.uniform(xl, xr))
            y = int(random.uniform(yl, yr))
            
            valid_boxes = (bbox[:, 0] >= x) & (bbox[:, 2] < x + self.roi_size) 
            valid_boxes &= (bbox[:, 1] >= y) & (bbox[:, 3] < y + self.roi_size)
            
            if valid_boxes.any():
                break
            
        else:
            #I am just centering this to for it to include the first bbox
            x = max(0, int(bbox[0, 1] - self.roi_size))
            y = max(0, int(bbox[0, 0] - self.roi_size))
            
        
        
        img_crop = image[y:y+self.roi_size, x:x+self.roi_size].copy()
        bbox = bbox[valid_boxes].copy()
        bbox[:, [0, 2]] -= x
        bbox[:, [1, 3]] -= y
        
        target["boxes"] = bbox
        target['labels'] = target['labels'][valid_boxes].copy()
        
        return img_crop, target

class ToTensor(object):
    def __call__(self, image, target):
        image = np.rollaxis(image, 2, 0).copy()
        image = torch.from_numpy(image).float()
        
        target['boxes'] = torch.from_numpy(target['boxes']).float()
        target['labels'] = torch.from_numpy(target['labels']).long()
        
        
        return image, target


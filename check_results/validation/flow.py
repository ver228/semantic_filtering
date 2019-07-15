#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 17:41:46 2019

@author: avelinojaver
"""


from transforms import RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, ToTensor, Compose

import numpy as np
import cv2
import pandas as pd
import tqdm

def collate_simple(batch):
    batch = [(x,target) for x, target in batch if len(target['boxes'])> 0]
    return tuple(map(list, zip(*batch)))


def get_transforms(transform_type, roi_size = None):
    if not transform_type  or transform_type == 'none':
         transforms = []
    elif transform_type == 'all':
        transforms = [RandomCrop(roi_size), RandomHorizontalFlip(), RandomVerticalFlip()]
    elif transform_type == 'flips':
        transforms = [RandomHorizontalFlip(), RandomVerticalFlip()]
    elif transform_type == 'hflip':
        transforms = [RandomHorizontalFlip()]
    elif transform_type == 'vflip':
        transforms = [RandomVerticalFlip()]
    elif transform_type == 'crops':
        transforms = [RandomCrop(roi_size)]
        
    else:
        raise ValueError(f'Not implemented {transform_type}')
        
    transforms.append(ToTensor)
    
    transforms = Compose(transforms)
    return transforms

class BBBC042Dataset(object):
    
    train_range = (1, 999) 
    val_range = (1000, 1049)
    test_ind = 1050
    
    
    def __init__(self, 
                 root_dir, 
                 set_type = 'train',
                 max_samples = None,
                 transforms = None
                 ):
        self.root_dir = root_dir
        self.transforms = transforms if transforms is not None else ToTensor()
        
        images_dir = root_dir / 'images'
        
        img_files = [(int(x.stem), x) for x in images_dir.rglob('*.tif')]
        
        if set_type == 'train':
            img_files = [(xid, x) for xid, x in img_files if (xid >= self.train_range[0]) and (xid <= self.train_range[1])]
            expected_ranges = self.train_range
            
        elif set_type == 'val':
            img_files = [(xid, x) for xid, x in img_files if (xid >= self.val_range[0]) and (xid <= self.val_range[1])]
            expected_ranges = self.val_range
            
        elif set_type == 'test':
            img_files = [(xid, x) for xid, x in img_files if (xid >= self.test_ind)]
            expected_ranges = self.test_ind, max([x[0] for x in img_files]) 
        else:
            raise ValueError(f'`set_type` {set_type} is not valid.')
        
        
        
        
        img_files = sorted(img_files, key = lambda x : x[0])
        if max_samples is not None:
            img_files = img_files[:max_samples]
            expected_ranges = expected_ranges[0], min(expected_ranges[0] + max_samples, expected_ranges[1])
            
        
        self.data = []
        for img_id, fname in tqdm.tqdm(img_files, desc = 'Loading Data'):
            dat = self._load_file(fname)
            if dat is not None:
                self.data.append(dat)
        
        self.img_shape = self.data[0][1].shape
        
        assert all([(x[0] >= expected_ranges[0]) and x[0] <= expected_ranges[1] for x in self.data])
        
    def _load_file(self, fname):
        img_id = int(fname.stem)
        
        img = cv2.imread(str(fname), -1)[..., ::-1]
        img = img.astype(np.float32)
        img /= 128
        img -= 1 #Let's scale from [-128/128, 127/128]
        
        annotations_file = self.root_dir / 'positions' / f'{img_id}.txt'
        df = pd.read_csv(str(annotations_file), delim_whitespace=True, header = None)
        boxes = df.loc[:, 4:7].values.astype(np.float32)
        labels = np.ones((len(boxes),), dtype=np.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        if len(boxes):
            return img_id, img, target
        else:
            return None
    
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ind):
        img_id, img, target = self.data[ind]
        
        #I want to copy the data since i am preloading it
        img = img.copy()
        target = dict(target)
        img, target = self.transforms(img, target)
            
        return img, target
    
    
def _test_flow(gen):
    
    #%%
    for ii in range(10):
        X, target  = gen[ii]
        
        img = X.detach().cpu().numpy()
        img = np.rollaxis(img, 0, 3) 
        img = (img + 1)/2
        
        fig, ax = plt.subplots(1,1)
        plt.imshow(img)
        for bb in target['boxes']:
            cc = (bb[0], bb[1])
            w = bb[2] - bb[0]
            h = bb[3] - bb[1]
            rect = patches.Rectangle(cc, w, h, linewidth=2, edgecolor='r', facecolor='none', linestyle = '-')
            ax.add_patch(rect)
#%%
if __name__ == '__main__':
    import matplotlib.pylab as plt
    import matplotlib.patches as patches
    from pathlib import Path
    
    data_dir = Path('/Users/avelinojaver/Downloads/BBBC042/')
    #data_dir = Path.home() / 'workspace/datasets/BBBC/BBBC042'
    
    roi_size = 512
    transforms = get_transform(roi_size = roi_size)
    gen = BBBC042Dataset(data_dir, 
                         #max_samples = 100, 
                         transforms = transforms,
                         set_type = 'test')
    _test_flow(gen)
    
    
    #%%
    
    
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:19:08 2019

@author: avelinojaver
"""
from pathlib import Path
import cv2
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.patches as patches
import numpy as np
import random
from skimage.filters import threshold_otsu

if __name__ == '__main__':
    raw_root_dir = Path('/Users/avelinojaver/Downloads/BBBC042/images/')
    
    fnames = raw_root_dir.rglob('*.tif')
    fnames = [x for x in fnames if not x.name.startswith('.')]
    fnames = sorted(fnames)
    
    #%%
    for fname in fnames:
        annotations_file = str(fname).replace('/images/', '/positions/').replace('.tif', '.txt')
        df = pd.read_csv(str(annotations_file), delim_whitespace=True, header = None)
        
        img = cv2.imread(str(fname), -1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
        img = 255 - img
        
        
        fig, ax = plt.subplots(1,1, sharex=True, sharey=True)
        ax.imshow(img, cmap = 'gray')
        
        img_bgnd = img.copy()
        
        cell_crops = []
        for _, row in df.iterrows():
            x1, y1, x2, y2 = row[4:8]
            img_bgnd[y1:y2+1, x1:x2+1] = 0
            
            crop_ori = img[y1:y2+1, x1:x2+1].copy()
            cell_crops.append(crop_ori)
        
            
            cc = x1, y1
            ll = x2 - x1
            ww = y2 - y1
            rect = patches.Rectangle(cc, ll, ww, linewidth=2, edgecolor='r', facecolor='none', linestyle = '-')
            ax.add_patch(rect)
        
        
        break
    
    
    
    #%%
    
        
    #%%
    
    img_s = np.zeros((256, 256))
    crop_bgnd = get_random_crop(img_bgnd, 256)
    
    
    for _ in range(10):
        crop_size = random.randint(64, 128)
        cc = get_random_crop(img_bgnd, crop_size)
        
        xi = random.randint(0, crop_bgnd.shape[0] - crop_size)
        yi = random.randint(0, crop_bgnd.shape[1] - crop_size)
        
        
        if random.random() > 0.5:
            crop_bgnd[xi:xi + crop_size, yi:yi + crop_size] = cc
        else:
            bw = _segment_roi(cc)
            crop_bgnd[xi:xi + crop_size, yi:yi + crop_size][bw] = cc[bw]
            
        
    
    for _ in range(5):
        cc = random.choice(cell_crops)
        
        xi = random.randint(0, crop_bgnd.shape[0] - cc.shape[0])
        yi = random.randint(0, crop_bgnd.shape[1] - cc.shape[1])
        
        xf = xi + cc.shape[0]
        yf = yi + cc.shape[1]
        
        if random.random() > 0.5:
            crop_bgnd[xi:xf, yi:yf] = cc
        else:
            bw = _segment_roi(cc)
            crop_bgnd[xi:xf, yi:yf][bw] = cc[bw]
    
    plt.figure()
    plt.imshow(crop_bgnd)
    
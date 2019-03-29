#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:46:02 2019

@author: avelinojaver
"""
from pathlib import Path
import numpy as np
import cv2
from scipy.ndimage.filters import median_filter
import tables

from get_accuracy import get_labels, calculate_performance, read_GT

def _rescale(x):
    bot, mid, top = np.percentile(x, [2, 50, 99])
    x = (x-bot)/(top-bot)*255
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x
#%%
def save_removed_labels(img_dir, save_dir, ICF = None, is_rescale=False):
    #%%
    img_dir = Path(img_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    for fname in img_dir.rglob('*.png'):
        img, img_g, target_coords = read_GT(fname)
        
        if ICF is not None:
            img_g = img_g/ICF
            is_rescale = True
            #img_g = np.round(np.clip(img_g, 0, 255))
            #img_g = img_g.astype(np.uint8)
        
        if is_rescale:
            img_g =  _rescale(img_g)
        
        
        #
        #plt.figure()
        #plt.imshow(img_g, vmax=255, cmap='gray')
        #plt.title((bot, mid, top))
        
        
        cv2.imwrite(str(save_dir / fname.name), img_g)
        
#%%

if __name__ == '__main__':
    _debug = True
    #root_dir = '/Users/avelinojaver/Downloads/BBBC026_GT_images/'
    #save_dir = Path('/Users/avelinojaver/Downloads/BBBC026_GT_nolabel/')
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/papers/miccai2019/data/hepatocytes/BBBC026/'
    root_dir = Path(root_dir)
    
    plate_dir = root_dir / 'BBBC026_v1_images'
    
    save_removed_labels(root_dir / 'BBBC026_GT_images', root_dir / 'BBBC026_GT_nolabel_scaled', is_rescale=True)
    
    
    fnames = list(plate_dir.glob('*_?01_*.png')) + list(plate_dir.glob('*_?23_*.png')) 
    save_dir =  root_dir / 'BBBC026_control'
    save_dir.mkdir(exist_ok=True, parents=True)
    
    for fname in fnames:
        img = cv2.imread(str(fname), -1)
        img = _rescale(img)
        cv2.imwrite(str(save_dir / fname.name), img)
    #%%
    
    
    #%%
#    #%%
#    imgs = [cv2.imread(str(x), -1).astype(np.float32) for x in plate_dir.glob('*.png')]
#    img_m = np.mean(imgs, axis=0)
#    
#    #%%
#    #this is only to be able to use the opencv median filter
#    bot, top = img_m.min(), img_m.max()
#    img_n = ((img_m - bot) / (top-bot) * 255).astype(np.uint8)
#    ICF = cv2.medianBlur(img_n, 301)
#    
#    ROBUST_FACTOR = 2
#    robust_minimum = np.percentile(ICF, 2)
#    ICF[ICF < robust_minimum] = robust_minimum
#    ICF = ICF/robust_minimum
#    
#    
#    if _debug:
#        img = imgs[100]
#    
#        img_n = img/ICF
#        vmax = np.maximum(np.max(img_n), np.max(img))
#        
#        import matplotlib.pylab as plt
#        fig, axs = plt.subplots(1,2,sharex=True, sharey=True)
#        axs[0].imshow(img, vmax=vmax)
#        axs[1].imshow(img_n, vmax=vmax)
#   
#    #%%
#    
#    save_removed_labels(root_dir / 'BBBC026_GT_images', root_dir / 'BBBC026_GT_nolabel_illumcorr', ICF)
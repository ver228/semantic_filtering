#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 17:19:37 2019

@author: avelinojaver
"""
#%%
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.filters import threshold_otsu
import numpy as np
from pathlib import Path
import tqdm

if __name__ == '__main__':
    raw_root_dir = Path('/Users/avelinojaver/Downloads/BBBC042/images/')
    save_root_dir = Path.home() / 'workspace/denoising/data/BBBC042/cell_bgnd_divided_v2'
    #%%
    save_train = save_root_dir / 'train'
    save_test = save_root_dir / 'test'
    _is_debug = True
    
    #fname = Path('/Users/avelinojaver/Downloads/BBBC042/images/1000.tif')
    fnames = raw_root_dir.rglob('*.tif')
    fnames = [x for x in fnames if not x.name.startswith('.')]
    
    for fname in tqdm.tqdm(fnames):
    
        annotations_file = str(fname).replace('/images/', '/positions/').replace('.tif', '.txt')
        df = pd.read_csv(str(annotations_file), delim_whitespace=True, header = None)
        
        img = cv2.imread(str(fname), -1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
        img = 255 - img
        
        if _is_debug:
            fig, ax = plt.subplots(1,1, sharex=True, sharey=True)
            ax.imshow(img, cmap = 'gray')
        
        img_bgnd = img.copy()
        cell_crops = []
        for _, row in df.iterrows():
            x1, y1, x2, y2 = row[4:8]
            
            crop_ori = img[y1:y2+1, x1:x2+1].copy()
            
            img_bgnd[y1:y2+1, x1:x2+1] = 0
            
            crop_m = crop_ori.copy()
            for _ in range(5):
                crop_m = cv2.medianBlur(crop_m, ksize = 3)
            
            th = threshold_otsu(crop_m)
            
            _, bw = cv2.threshold(crop_m, th, 255, cv2.THRESH_BINARY)
            
            
            nlabels, labels, stats, centroids =  cv2.connectedComponentsWithStats(bw)
            largest_ind = np.argmax(stats[1:, -1]) + 1
            bw[labels != largest_ind] = 0
            
            #bw = cv2.dilate(bw, np.ones((3,3)))
            #bw = cv2.erode(bw, np.ones((3,3)))
            crop_cleaned = crop_ori.copy()
            crop_cleaned[bw==0] = 0
            
            if _is_debug:
                _, axs2 = plt.subplots(1,2, sharex=True, sharey=True)
                axs2[0].imshow(crop_ori)
                axs2[1].imshow(crop_cleaned)
                
                cc = x1, y1
                ll = x2 - x1
                ww = y2 - y1
                rect = patches.Rectangle(cc, ll, ww, linewidth=2, edgecolor='r', facecolor='none', linestyle = '-')
                ax.add_patch(rect)
            
            cell_crops.append(crop_cleaned)
        
        if _is_debug:
            break
        
        fnum = int(fname.stem)
        save_dir = save_train if fnum < 1000 else save_test
        
        bngd_save_name = save_dir / 'background' / f'{fnum}.tif'
        bngd_save_name.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(bngd_save_name), img_bgnd)
        
        for icell, cell_img in enumerate(cell_crops):
            cell_save_name = save_dir / 'foreground' / f'{fnum}_{icell}.tif'
            cell_save_name.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(cell_save_name), cell_img)
            
        
    
    
    
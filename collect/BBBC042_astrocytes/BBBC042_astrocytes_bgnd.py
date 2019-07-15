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
import random

def _segment_crop(crop_ori, max_dist_norm = 0.42, min_length_norm = 0.3):
    crop_m = crop_ori.copy()
    for _ in range(5):
        crop_m = cv2.medianBlur(crop_m, ksize = 3)
    
    th = threshold_otsu(crop_m[crop_ori>0])*0.9
    
    _, bw = cv2.threshold(crop_m, th, 255, cv2.THRESH_BINARY)
    
    #nlabels, labels, stats, centroids =  cv2.connectedComponentsWithStats(bw)
    _, cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    M = [cv2.moments(x) for x in cnts]
    m00, m10, m01  = map(np.array, zip(*[(x['m00'],x['m10'],x['m01']) for x in M]))
    roi_center = np.array(crop_m.shape)/2
    
    dist_x = (roi_center[1] - m10/m00)/roi_center[1]
    dist_y = (roi_center[0] - m01/m00)/roi_center[0]
    
    norm_dist_from_center = np.sqrt(dist_x**2 + dist_y**2)
    cnts = [x for ii, x in enumerate(cnts) if norm_dist_from_center[ii] < max_dist_norm]
    
    is_cleaned = False
    if cnts:
        cnt = max(cnts, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(cnt)
        if (w/crop_m.shape[0] > min_length_norm or h/crop_m.shape[1] > min_length_norm):
            bw = np.zeros(crop_m.shape, np.uint8)
            cv2.drawContours(bw,[cnt] , -1, 255, -1)
            is_cleaned = True
            
    crop_cleaned = crop_ori.copy()
    crop_cleaned[bw==0] = 0
    
    return crop_cleaned, is_cleaned

if __name__ == '__main__':
    n_bgnd_crops = 50
    _is_debug = False
    
    raw_root_dir = Path.home() / 'workspace/datasets/BBBC/BBBC042/images/'
    save_root_dir = Path.home() / 'workspace/denoising/data/BBBC042_bgnd/'
    #%%
    save_train = save_root_dir / 'train'
    save_test = save_root_dir / 'test'
    
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
        
        img_no_cells = img.copy()
        cell_crops = []
        for _, row in df.iterrows():
            x1, y1, x2, y2 = row[4:8]
            
            crop_ori = img[y1:y2+1, x1:x2+1].copy()
            
            img_no_cells[y1:y2+1, x1:x2+1] = 0
            
            crop_cleaned, is_cleaned = _segment_crop(crop_ori)
            if _is_debug:
                _, axs2 = plt.subplots(1,2, sharex=True, sharey=True)
                axs2[0].imshow(crop_ori)
                axs2[1].imshow(crop_cleaned)
                
                cc = x1, y1
                ll = x2 - x1
                ww = y2 - y1
                rect = patches.Rectangle(cc, ll, ww, linewidth=2, edgecolor='r', facecolor='none', linestyle = '-')
                ax.add_patch(rect)
            
            if is_cleaned and np.any(crop_cleaned>0):
                cell_crops.append(crop_cleaned)
        #%%
       
        bgnd_crops = []
        crop_size = (128, 128)
        for _ in range(n_bgnd_crops):
            xi = random.randint(0, img_no_cells.shape[0] - crop_size[0])
            yi = random.randint(0, img_no_cells.shape[1] - crop_size[1])
        
            img_crop = img_no_cells[xi:xi + crop_size[0], yi:yi + crop_size[1]]
            crop_cleaned, _ = _segment_crop(img_crop, max_dist_norm = 1.5)
            
            if np.any(crop_cleaned>0):
                bgnd_crops.append(crop_cleaned)
            
        
        bgnd_bw = cv2.adaptiveThreshold(img_no_cells, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,255, -25)
        bgnd_bw = cv2.dilate(bgnd_bw, np.ones((3,3)), iterations=2)
        img_bgnd = img_no_cells.copy()
        img_bgnd[bgnd_bw>0] = 0
        
        
        if _is_debug:
            for cc in bgnd_crops:
                plt.figure()
                plt.imshow(cc)
            
            _, axs_bw = plt.subplots(1,2, sharex=True, sharey=True)
            axs_bw[0].imshow(img)
            axs_bw[1].imshow(img_bgnd)
            
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
            
        
        for ii, b_crop in enumerate(bgnd_crops):
            save_name = save_dir / 'background_crops' / f'{fnum}_{ii}.tif'
            save_name.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(save_name), b_crop)
    
        
    
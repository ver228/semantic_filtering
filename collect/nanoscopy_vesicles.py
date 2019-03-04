#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 11:19:37 2019

@author: avelinojaver
"""

import json
import cv2
from pathlib import Path

import tqdm
import tifffile
import numpy as np

#%%
if __name__ == '__main__':
    tracks_file = '/Volumes/rescomp1/data/nanoscopy/raw_data/50min_bio_10fps_Airyscan_ProcessingSetting3-3_detections.txt'
    src_file = '/Volumes/rescomp1/data/nanoscopy/raw_data/50min_bio_10fps_Airyscan_ProcessingSetting3-3.tif'
    
    save_root = '/Volumes/rescomp1/data/denoising/data/vesicles_nanoscopy/'
    
    save_dir = Path(save_root) / Path(src_file).stem 
    #save_dir = Path(save_root) / ('test_' + Path(src_file).stem)
    #%%
    bgnd_dir = save_dir / 'background'
    fgnd_dir = save_dir / 'foreground'
    
    bgnd_dir.mkdir(parents=True, exist_ok=True)
    fgnd_dir.mkdir(parents=True, exist_ok=True)
    #%%
    imgs = tifffile.imread(src_file)
    #%%
    with open(tracks_file, 'r') as fid:
        coords = json.load(fid)
    
    inds = sorted(coords.keys(), key = lambda x : int(x))
    
    #%%
    _is_debug = False
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    ves_pad = 5
    
    nn = 2*ves_pad+1
    kernel_ves = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(nn, nn))
    kernel_ves = kernel_ves==0
    
    #%%
    for tt, img in tqdm.tqdm(zip(inds, imgs)):
        
        
        cc = coords[tt]
        cc = np.array(cc)
        
        mask = np.zeros_like(img, dtype = np.uint8)
        mask[cc[:, 0], cc[:, 1]] = 1
        mask = cv2.dilate(mask, kernel)
        
        bgnd = img.copy()
        bgnd[mask==1] = 0
        
        if _is_debug:
            import matplotlib.pylab as plt
            fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
            axs[0].imshow(img, interpolation = 'none')
            axs[0].plot(cc[:, 1], cc[:, 0], '.r')
            axs[1].imshow(bgnd, interpolation = 'none')
            #axs[1].plot(cc[:, 1], cc[:, 0], '.r')
        
        
        rois = [img[x-ves_pad:x+ves_pad+1, y-ves_pad:y+ves_pad+1].copy() for x,y in cc]
        
        bgnd_name = bgnd_dir / '{:03d}.tif'.format(int(tt))
        cv2.imwrite(str(bgnd_name), bgnd)
        
        ves_name_root =str(fgnd_dir / '{:03d}'.format(int(tt)))
        for i_ves, ves_img in enumerate(rois):
            save_name = f'{ves_name_root}_{i_ves:04d}.tif'
            
            
            ves_img[kernel_ves] = 0
            cv2.imwrite(save_name, ves_img)
        #break
        
        
        
    
    
    
    
    #%%
    
    
    
    
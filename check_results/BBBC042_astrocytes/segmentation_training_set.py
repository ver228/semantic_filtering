#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:12:10 2018

@author: avelinojaver
"""
import sys
from pathlib import Path

dname = Path(__file__).resolve().parents[2]
sys.path.append(str(dname))


from bgnd_removal.models import UNet
import torch
import numpy as np
import cv2
import pandas as pd
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#%%
if __name__ == '__main__':
    bn = 'BBBC042-v3-separated_unet_l1smooth_20190302_224236_adam_lr0.00032_wd0.0_batch32'
    
    n_epochs = 299#499#
    model_path = Path.home() / 'workspace/denoising/results/BBBC042' / bn / f'checkpoint-{n_epochs}.pth.tar'
    
    int_scale = (0,255)
    
    n_ch_in, n_ch_out  = 1, 1
    if '-separated' in bn:
        n_ch_out = 3
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out)
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    root_dir = Path('/Users/avelinojaver/Downloads/BBBC042/')
    fg_dir = Path('/Users/avelinojaver/Desktop/BBBC042_divided/train/foreground/')
    
    import tqdm
    for img_id in tqdm.tqdm([5, 420]):#range(800, 820)):
        annotations_file = root_dir / 'positions' / f'{img_id}.txt'
        fname = root_dir / 'images' / f'{img_id}.tif'
        
        if not (fname.exists() and annotations_file.exists()):
            continue
        
        df = pd.read_csv(str(annotations_file), delim_whitespace=True, header = None)
        img_ori = cv2.imread(str(fname), -1)[..., ::-1]
        img = cv2.cvtColor(img_ori, cv2.COLOR_RGB2GRAY);
        img = 255 - img
        
        
        #%%
        seg_ori = np.zeros_like(img)
        for irow, row in df.iterrows():
            crop_file = fg_dir / f'{img_id}_{irow}.tif'
            if crop_file.exists():
                crop = cv2.imread(str(crop_file), -1)
                
                x1, y1, x2, y2 = row[4:8]
                dd = seg_ori[ y1:y2+1, x1:x2+1].view()
                
                dd += crop
        
        
        
        #%%
        x = img[None].astype(np.float32)
        x = (x - int_scale[0])/(int_scale[1] - int_scale[0])
        
        with torch.no_grad():
            X = torch.from_numpy(x[None])
            Xhat = model(X)
        
        xhat = Xhat.squeeze().detach().numpy()
        xr = x.squeeze()
        
        #%%
        if xhat.ndim == 2:
            x2th = xhat
        else:
            x2th = xhat[0] 
        
        
        th = threshold_otsu(x2th)*0.8
        #th = 0.05
        x_th = x2th>th
        
        #%%
        fig, axs = plt.subplots(1, 3,sharex=True, sharey=True, figsize=(20, 8))
        
        axs[0].imshow(img_ori, cmap='gray')
        
        for _, row in df.iterrows():
            x1, y1, x2, y2 = row[4:8]
            cc = x1, y1
            ll = x2 - x1
            ww = y2 - y1
            rect = patches.Rectangle(cc, ll, ww, linewidth=2, edgecolor='r', facecolor='none', linestyle = '-')
            axs[0].add_patch(rect)
        
        axs[1].imshow(seg_ori , cmap='gray')
        axs[2].imshow(xhat[0] , cmap='gray')
        
            
        for ax in axs.flatten():
            ax.axis('off')
    #    
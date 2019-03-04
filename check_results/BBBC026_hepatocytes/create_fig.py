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
from skimage.filters import threshold_otsu
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#%%
if __name__ == '__main__':
    bns = ['BBBC026-separated_unet_l1smooth_20190226_082017_adam_lr0.00032_wd0.0_batch32',
           'BBBC026-hepatocytes_unet_l1smooth_20190226_082040_adam_lr0.00032_wd0.0_batch32'
           ]
    
    im2test = '/Users/avelinojaver/Downloads/BBBC026_v1_images/ADSASS092408-GHAD2-D6-20x_O09_s3_w1FB80F860-83B8-4E63-B7B1-D54D98716578.png'
    
    img = cv2.imread(im2test, -1)
    x = img.astype(np.float32)
    
    pix_top = np.percentile(x, 99)
    x = x/pix_top
        
    
    results = []
    for bn in bns:
        model_path = Path.home() / 'workspace/denoising/results/BBBC026' / bn / 'checkpoint.pth.tar'
        
        n_ch_in, n_ch_out  = 1, 1
        if '-separated' in bn:
            n_ch_out = 3
        model = UNet(n_channels = n_ch_in, n_classes = n_ch_out)
        
        
        state = torch.load(model_path, map_location = 'cpu')
        model.load_state_dict(state['state_dict'])
        model.eval()
        
        
        with torch.no_grad():
            X = torch.from_numpy(x[None, None])
            Xhat = model(X)
        
        xhat = Xhat[0].detach().numpy()
        results.append(xhat)
    
    #%%
    if xhat.ndim == 2:
        x2th = xhat
    else:
        #x2th = xhat[0]/xhat.sum(axis=0)
        x2th = xhat[0] 
    
    
    #th = 0.2
    th = threshold_otsu(x2th)
    x_th = x2th>th
    
    #%%
    
    fig, axs = plt.subplots(1, 3, figsize = (20, 6), sharex=True, sharey=True)
    
    axs[0].imshow(xr, cmap='gray')#, vmin=0, vmax=1)
    axs[0].set_title('Original')
    
    if cm_fib is not None and 'fibroblasts' in bn:
        #axs[1].plot(cm_fib[..., 0], cm_fib[..., 1], 'r.')
        axs[2].plot(cm_fib[..., 0], cm_fib[..., 1], 'r.')
    elif cm_hep is not None:
        #axs[1].plot(cm_hep[..., 0], cm_hep[..., 1], 'g.')
        axs[2].plot(cm_hep[..., 0], cm_hep[..., 1], 'g.')
    
    
    if xhat.ndim == 3:
        x_rgb = np.rollaxis(xhat, 0, 3)
    else:
        x_rgb = xhat.squeeze()
        
    axs[1].imshow(x2th, cmap='gray')#, vmin=0, vmax=1)
    axs[1].set_title('Prediction')
    
    axs[2].imshow(x_th)
    axs[2].set_title(f'Prediction > {th}')
    
    plt.suptitle(bn)
    #for ax in axs.flatten():
    #    ax.axis('off')
    
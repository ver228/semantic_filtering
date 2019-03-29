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
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/50.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/5.tif'
    
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1075.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1020.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1100.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1010.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1004.tif'
    
    
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1021.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1003.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1074.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1041.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1020.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1016.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1011.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1006.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1003.tif'
    
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1051.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1049.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1039.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1037.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1022.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1017.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1007.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1006.tif'
    
    
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1021.tif'
    fname = '/Users/avelinojaver/Downloads/BBBC042/images/1049.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1026.tif'
    
    annotations_file = str(fname).replace('/images/', '/positions/').replace('.tif', '.txt')
    df = pd.read_csv(str(annotations_file), delim_whitespace=True, header = None)
    
    img_ori = cv2.imread(fname, -1)[..., ::-1]
    img = cv2.cvtColor(img_ori, cv2.COLOR_RGB2GRAY);
    img = 255 - img
    
    #img = cv2.dilate(img, (3,3))
    
    img = img[None]
    
    x = img.astype(np.float32)
    #pix_top = np.percentile(x, 99)
    #x = x/pix_top
    
    #x = np.log(x+1)
    x = (x - int_scale[0])/(int_scale[1] - int_scale[0])
    
    with torch.no_grad():
        X = torch.from_numpy(x[None])
        Xhat = model(X)
    #%%
    xhat = Xhat.squeeze().detach().numpy()
    xr = x.squeeze()
    
    #%%
    if xhat.ndim == 2:
        x2th = xhat
    else:
        #x2th = xhat[0]/xhat.sum(axis=0)
        x2th = xhat[0] 
    
    
    th = threshold_otsu(x2th)*0.8
    #th = 0.05
    x_th = x2th>th
    
    #%%
    n_plots = 4 if xhat.shape[0] == 3 else 2
    fig, axs = plt.subplots(1, n_plots,sharex=True, sharey=True, figsize=(20, 8))
    
    axs[0].imshow(img_ori, cmap='gray')#, vmin=0, vmax=1)
    
    for _, row in df.iterrows():
        x1, y1, x2, y2 = row[4:8]
        cc = x1, y1
        ll = x2 - x1
        ww = y2 - y1
        rect = patches.Rectangle(cc, ll, ww, linewidth=2, edgecolor='r', facecolor='none', linestyle = '-')
        axs[0].add_patch(rect)
    
    if xhat.shape[0] == 3:
        
        axs[1].imshow(xhat[0] , cmap='gray')#, vmin=0, vmax=1)
        
        axs[2].imshow(xhat[2] , cmap='gray')#, vmin=0, vmax=1)
        
        axs[3].imshow(xhat[1] , cmap='gray')#, vmin=0, vmax=1)
    else:
        axs[1].imshow(xhat, cmap='gray')
    for ax in axs.flatten():
        ax.axis('off')
#    
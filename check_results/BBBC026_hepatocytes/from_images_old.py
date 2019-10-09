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


from semantic_filtering.models import UNet
import torch
import numpy as np
import cv2
from skimage.filters import threshold_otsu
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#%%
if __name__ == '__main__':
    #bn = 'sBBBC026-hepatocytes_l1smooth_20190219_175950_unet_adam_lr0.00032_wd0.0_batch32'
    #model_path =  Path('/Volumes/loco/workspace/denoising/results/') / bn.split('_')[0] / bn / 'checkpoint.pth.tar'
    
    
    #bn = 'BBBC026-hepatocytes_l1smooth_20190218_153253_unet_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-fibroblasts_l1smooth_20190218_162426_unet_adam_lr0.00032_wd0.0_batch32'
    
    #bn = 'BBBC026-hepatocytes-log_l1smooth_20190218_182937_unet_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-fibroblasts-log_l1smooth_20190219_022405_unet_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-hepatocytes_l1smooth_20190218_222644_unet_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-fibroblasts_l1smooth_20190219_062152_unet_adam_lr0.00032_wd0.0_batch32'
    
    #bn = 'BBBC026-hepatocytes-log_l1smooth_20190219_111711_unet_adam_lr0.00032_wd0.0_batch32'
    #bn = 'sBBBC026-hepatocytes_l1smooth_20190219_155630_unet_adam_lr0.00032_wd0.0_batch32'
    
    #bn = 'sBBBC026-fibroblasts_l1smooth_20190219_201226_unet_adam_lr0.00032_wd0.0_batch32'
    
    #bn = 'sBBBC026-fibroblasts_l1smooth_20190220_184127_unet_adam_lr0.00032_wd0.0_batch32'
    #bn = 'sBBBC026-hepatocytes_l1smooth_20190220_184125_unet_adam_lr0.00032_wd0.0_batch32'
    
    #model_path = Path.home() / 'workspace/denoising/results' / bn.split('_')[0] / bn / 'checkpoint.pth.tar'
    
    #bn = 'BBBC026-fibroblasts_unet_l1smooth_20190222_183808_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-separated_unet_l1smooth_20190222_184510_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-hepatocytes_unet_l1smooth_20190222_183622_adam_lr0.00032_wd0.0_batch32'
     
    #bn = 'BBBC026-separated_unet_l1smooth_20190223_221251_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-hepatocytes_unet_l1smooth_20190223_220901_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-fibroblasts_unet_l1smooth_20190223_220826_adam_lr0.00032_wd0.0_batch32'
    
    #bn = 'BBBC026-separated_unet_l1smooth_20190224_105903_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-hepatocytes_unet_l1smooth_20190224_105953_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-fibroblasts_unet_l1smooth_20190224_105911_adam_lr0.00032_wd0.0_batch32'
    
    
    #bn = 'BBBC026-separated_unet_l1smooth_20190225_235932_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-hepatocytes_unet_l1smooth_20190226_000010_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-fibroblasts_unet_l1smooth_20190225_235927_adam_lr0.00032_wd0.0_batch32'
    
    bn = 'BBBC026-separated_unet_l1smooth_20190226_082017_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-hepatocytes_unet_l1smooth_20190226_082040_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-fibroblasts_unet_l1smooth_20190226_082007_adam_lr0.00032_wd0.0_batch32'
    
    model_path = Path.home() / 'workspace/denoising/results/_old_BBBC026' / bn / 'checkpoint.pth.tar'
    model_path = Path.home() / 'workspace/denoising/results/_old_BBBC026' / bn / 'checkpoint-24.pth.tar'
    
    n_ch_in, n_ch_out  = 1, 1
    if '-separated' in bn:
        n_ch_out = 3
    
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out)
    
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/papers/miccai2019/data/hepatocytes/BBBC026')
    
    #fname = '/Users/avelinojaver/Downloads/BBBC026_v1_images/ADSASS092408-GHAD2-D6-20x_O23_s9_w14C780EEA-354A-4814-9D87-38CA9E3D5F39.png'
    #fname = '/Users/avelinojaver/Downloads/BBBC026_v1_images/ADSASS092408-GHAD2-D6-20x_O01_s6_w132859271-AB55-46F9-8029-6A1742DECF06.png'
    #fname = '/Users/avelinojaver/Downloads/BBBC026_v1_images/ADSASS092408-GHAD2-D6-20x_O09_s3_w1FB80F860-83B8-4E63-B7B1-D54D98716578.png'
    
    #fname = '/Users/avelinojaver/Downloads/BBBC026_GT_images/A03_s1.png'
    #fname = '/Users/avelinojaver/Downloads/BBBC026_GT_images/A07_s4.png'
    #fname = '/Users/avelinojaver/Downloads/BBBC026_GT_images/C09_s2.png'
    fname = root_dir / 'BBBC026_GT_images/M19_s6.png'     
    
    
    img = cv2.imread(str(fname), -1)
    
    if img.ndim == 2:
        cm_hep, cm_fib = None, None
    else:
        #this is the ground truth  
        img_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
        bad = (img[..., 0] == 255) & (img[..., 1] <= 10) & (img[..., 2] <= 10)
        fib =  (img[..., 0] <= 10) & (img[..., 1] == 255) & (img[..., 2] <= 10)
        hep =  (img[..., 0] <= 10) & (img[..., 1] <= 10) & (img[..., 2] == 255)
        
        cm_fib = cv2.connectedComponentsWithStats(fib.astype(np.uint8))[-1][1:].astype(np.int)
        cm_hep = cv2.connectedComponentsWithStats(hep.astype(np.uint8))[-1][1:].astype(np.int)
        
        peaks2remove = bad | fib | hep
        med = cv2.medianBlur(img_g, ksize= 11) + np.random.normal(0, 2, img_g.shape).round().astype(np.int)
        img_g[peaks2remove] = med[peaks2remove]
        img = img_g
    
    x = img[None].astype(np.float32)
    
    
    #int_scale = x.min(), x.max()
    #x = (x - int_scale[0])/(int_scale[1] - int_scale[0])
    
    if bn.split('_')[0].endswith('-log'):
        x = np.log(x+1)
        int_scale = (0, np.log(2**8))
        
    else:
        int_scale = (0,255)
    
    #x = (x - int_scale[0])/(int_scale[1] - int_scale[0]) 
    pix_top = np.percentile(x, 99)
    x = x/pix_top
    
    with torch.no_grad():
        X = torch.from_numpy(x[None])
        Xhat = model(X)
    
    xhat = Xhat.squeeze().detach().numpy()
    xr = x.squeeze()
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
    
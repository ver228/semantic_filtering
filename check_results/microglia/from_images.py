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
#%%
if __name__ == '__main__':
    
    #old
    #model_path = '/Volumes/rescomp1/data/denoising/_old_results/microglia_syntethic/microglia_synthetic_l1smooth_20181003_161017_unet_adam_lr0.0001_wd0.0_batch16/model_best.pth.tar'
    
    #new
    #bn = 'microglia-fluo_l2_20190131_164721_unet_adam_lr0.0001_wd0.0_batch12'
    #bn = 'microglia-fluo_l1_20190131_165128_unet_adam_lr0.0001_wd0.0_batch12'
    #bn = 'microglia-fluo_l1smooth_20190131_164723_unet_adam_lr0.0001_wd0.0_batch12'
    
    #bn = 'microglia-fluo_l2_20190201_172055_unet_adam_lr0.0001_wd0.0_batch12'
    #bn = 'microglia-fluo_l1_20190201_171947_unet_adam_lr0.0001_wd0.0_batch12'
    #bn = 'microglia-fluo_l1smooth_20190201_172003_unet_adam_lr0.0001_wd0.0_batch12'
    
    #bn = 'microglia-fluo-v2_l1smooth_20190204_191416_unet_adam_lr0.0001_wd0.0_batch12'
    #bn = 'microglia-fluo_l1smooth_20190204_184524_unet_adam_lr0.0001_wd0.0_batch12'
    
    #bn = 'microglia-fluo-clean_l1smooth_20190213_144050_unet_adam_lr0.0001_wd0.0_batch12'
    #bn = 'microglia-fluo-v2-clean_l1smooth_20190213_144254_unet_adam_lr0.0001_wd0.0_batch12'
    #model_path = Path.home() / 'workspace/denoising/results' / bn.split('_')[0] / bn / 'checkpoint.pth.tar'
    
    
    #bn = 'microglia-v2_unet_l1smooth_20190228_141742_adam_lr0.00012_wd0.0_batch12'
    #bn = 'microglia-v2-separated_unet_l1smooth_20190302_214016_adam_lr0.00012_wd0.0_batch12'
    #bn = 'microglia-v2-tight-separated_unet_l1smooth_20190307_085739_adam_lr0.00012_wd0.0_batch12'
    
    model_path = Path('/Volumes/loco/') / 'workspace/denoising/results' / 'microglia' / bn / 'checkpoint-299.pth.tar'
    #model_path = Path.home() / 'workspace/denoising/results/microglia' / bn / 'checkpoint.pth.tar'
    
    #model_path = f'/Volumes/rescomp1/data/microglia-fluo/{bn}/checkpoint-599.pth.tar'
    #model_path = f'/Volumes/rescomp1/data/denoising/results/microglia-fluo/{bn}/checkpoint.pth.tar'
    
    scale_log = (0, np.log(2**16))
    
    n_ch_in, n_ch_out  = 1, 1
    if '-separated' in bn:
        n_ch_out = 3
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out)
    
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/stills/2018.08.22_stills/180822_MicVid_40X_Stills/180822_MicVid_40X_Stills_20X_Stills_10pcPower_1/N - 6(fld 1- time 1 - 22858 ms).tif'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/stills/2018.08.22_stills/180822_MicVid_40X_Stills/180822_MicVid_40X_Stills_20X_Stills_10pcPower_1/J - 7(fld 1- time 1 - 35958 ms).tif'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/stills/2018.08.22_stills/180822_MicVid_40X_Stills/180822_MicVid_40X_Stills_20X_Stills_10pcPower_1/L - 8(fld 1- time 1 - 71667 ms).tif'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/stills/2018.08.22_stills/180815_MicVid_20X_Stills/180815_MicVid_20X_Stills_20X_Still_InjectionWells_2/J - 8(fld 8).tif'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/stills/2018.08.20_stills/180815_MicVid_20X_Stills/B - 9(fld 1 z 1- time 1 - 109319 ms).tif'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/stills/2018.08.20_stills/180815_MicVid_20X_Stills/B - 6(fld 1 z 3- time 1 - 684 ms).tif'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/stills/2018.08.20_stills/180815_MicVid_20X_Stills/D - 8(fld 1 z 2- time 1 - 59803 ms).tif'
    
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/stills/2018.08.20_stills/180815_MicVid_20X_Stills/H - 11(fld 1 z 3- time 2 - 153890 ms).tif'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/stills/2018.08.20_stills/180815_MicVid_20X_Stills/J - 9(fld 1 z 1- time 1 - 93555 ms).tif'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/stills/2018.08.20_stills/180815_MicVid_20X_Stills/L - 10(fld 1 z 3- time 2 - 132076 ms).tif'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/stills/2018.08.20_stills/180815_MicVid_20X_Stills/N - 10(fld 1 z 1- time 1 - 135117 ms).tif'
    
    fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/movies/2018.08.20_movies/180820_MicVid_20X_30s/180815_MicVid_20X_30s_20X_30s_4w/H - 12(fld 1 z 1- time 1 - 0 ms).tif'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/movies/2018.08.20_movies/180820_MicVid_20X_30s/180815_MicVid_20X_30s_20X_30s_4w/H - 12(fld 4 z 3- time 50 - 1470000 ms).tif'
    
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/movies/2018.08.20_movies/180820_MicVid_20X_30s/180815_MicVid_20X_30s_20X_30s_4w/H - 14(fld 1 z 1- time 1 - 0 ms).tif'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/movies/2018.08.20_movies/180820_MicVid_20X_30s/180815_MicVid_20X_30s_20X_30s_4w/H - 14(fld 4 z 3- time 50 - 1470000 ms).tif'
    
    
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/movies/2018.08.20_movies/180820_MicVid_20X_30s/180815_MicVid_20X_30s_20X_30s_4w/J - 11(fld 1 z 1- time 1 - 0 ms).tif'
    
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/movies/2018.08.20_movies/180820_MicVid_20X_30s/180815_MicVid_20X_30s_20X_30s_4w/J - 12(fld 1 z 1- time 1 - 0 ms).tif'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/movies/2018.08.20_movies/180820_MicVid_20X_30s/180815_MicVid_20X_30s_20X_30s_4w/J - 12(fld 4 z 3- time 50 - 1470000 ms).tif'
    
    
    #%%
    img = cv2.imread(fname, -1)[..., ::-1]
    
    x = img[None].astype(np.float32)
    
    x = np.log(x+1)
    x = (x - scale_log[0])/(scale_log[1] - scale_log[0])
    
    fig, axs = plt.subplots(1,2,sharex=True, sharey=True)
    axs[0].imshow(img, cmap='gray')
    axs[1].imshow(x[0], cmap='gray')
    for ax in axs.flatten():
        ax.axis('off')
            
    #%%
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
    
    
    th = threshold_otsu(x2th)*0.9
    x_th = x2th>th
    #%%
    if xhat.ndim == 2:
        fig, axs = plt.subplots(1,3,sharex=True, sharey=True)
    else:
        fig, axs = plt.subplots(1,4,sharex=True, sharey=True)
    
    axs[0].imshow(xr, cmap='gray')#, vmin=0, vmax=1)
    
    axs[1].imshow(x2th, cmap='gray')#, vmin=0.4)
    
    
    if xhat.ndim == 2:
        #th = 0.45
        th = threshold_otsu(xhat)*0.8
        axs[2].imshow(x_th)
        
        axs[0].set_title('Original')
        axs[1].set_title('Prediction')
        axs[2].set_title(f'Prediction > {th:.3}')
        
    else:
        axs[2].imshow(xhat[2], cmap='gray')
        axs[3].imshow(xhat[1], cmap='gray')
    
    
    for ax in axs.flatten():
        ax.axis('off')
            
    plt.suptitle(bn)        
    #%%
    
    
    
    
        
    
    
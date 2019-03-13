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
from bgnd_removal.flow import FluoMergedFlow

import torch
import numpy as np
import tqdm
import cv2
#%%
if __name__ == '__main__':
    #bn = 'fmnist-v2_unet_l1smooth_20190307_234839_adam_lr0.00032_wd0.0_batch32'
    #bn = 'fmnist-v2-separated_unet_l1smooth_20190307_234956_adam_lr0.00032_wd0.0_batch32'
    
    bn = 'fmnist-v2_unet_l1smooth_20190309_131918_adam_lr0.00032_wd0.0_batch32'
    #bn = 'fmnist-v2-separated_unet_l1smooth_20190309_131701_adam_lr0.00032_wd0.0_batch32'
    
    model_path = Path.home() / 'workspace/denoising/results/fmnist_v2' / bn / 'checkpoint.pth.tar'
    root_dir = Path.home() / 'workspace/denoising/data/MNIST_fashion/test'
    
    cells1_prefix = 'foreground'
    cells2_prefix = 'background_crops'
    bgnd_prefix = 'background'
    
    gen = FluoMergedFlow(root_dir = root_dir,
                            bgnd_prefix = bgnd_prefix,
                            cells1_prefix = cells1_prefix,
                            cells2_prefix = cells2_prefix,
                            crop_size = (128, 128),
                             is_log_transform = False,
                             int_scale = (0, 255),
                             img_ext = '*.png',
                             is_timeseries_dir = False,
                             n_cells_per_crop = 3,
                             n_bgnd_per_crop = 20,
                             int_factor = (0.9, 1.1),
                             bgnd_sigma_range = (0., 1.2),
                             frac_crop_valid = 0.25,
                             zoom_range = (0.9, 1.1),
                             noise_range = (0., 5.),
                             rotate_range = (0, 90),
                             max_overlap = 0.5,
                             is_separated_output = True
                             )  
    
    n_ch_in  = 1
    n_ch_out  = 3 if '-separated' in bn else 1
    
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    print(state['epoch'])
    
    #%%
    #is_plot = False; n_iter = 100
    is_plot = True; n_iter = 5
    
    
    #th = 0.1
    all_J = []
    
    _bad = []
    
    th_base = 0.05
    for ibatch, (xin, x_true) in tqdm.tqdm(enumerate(gen)):
        
       
        with torch.no_grad():
            X = torch.from_numpy(xin[None])
            Xhat = model(X)
        
        xin = xin.squeeze()
        x_true = x_true[0]
        
        xhat = Xhat[0].detach().numpy()
        x_pred = xhat[0]
        
        
        th_in = x_true > th_base
        
        med = np.median(x_pred)
        mad = np.median(np.abs(x_pred - med))
        
        th = med + 6*mad
        th = max(th_base, th)
        
        th_out = x_pred > th
        
        U = (th_in | th_out)
        I = (th_in & th_out)
        
        all_J.append((I.sum(), U.sum()))
        
        
        IoU = I.sum()/U.sum()
        
        if IoU < 0.8:
            _bad.append((xin, x_true, x_pred))
        
        if is_plot or IoU < 0.8:
            
            import matplotlib.pylab as plt
            fig, axs = plt.subplots(2,3,sharex=True, sharey=True, figsize = (14, 8))
            axs[0][0].imshow(xin, cmap='gray', vmin=0, vmax=1)
            axs[0][1].imshow(x_true, cmap='gray', vmin=0, vmax=1)
            
            axs[0][0].set_title('Input')
            axs[0][1].set_title('Ground Truth (GT)')
            
            if xhat.shape[0] == 3:
                axs[1][0].imshow(xhat[2], cmap='gray', interpolation=None)
                axs[1][1].imshow(xhat[0], cmap='gray', interpolation=None)
                axs[1][2].imshow(xhat[1], cmap='gray', interpolation=None)

                axs[0][2].imshow(th_in)
                #axs[1][2].imshow(th_out)
            else:
                axs[0][2].imshow(th_in)
            
                axs[1][0].imshow(x_pred - x_true)
                axs[1][1].imshow(x_pred, cmap='gray', vmin=0, vmax=1)
                axs[1][2].imshow(th_out)
            #axs[0][2].set_title(f'GT > {th}')
            #axs[1][0].set_title('GT - P')
            #axs[1][1].set_title('Prediction (P)')
            axs[1][2].set_title(f'P>{th}, IoU={IoU:.3}')
            
            for ax in axs.flatten():
                ax.axis('off')
        
            
        if ibatch >= n_iter:
            break
        
    I, U = map(np.sum, zip(*all_J))
    mIOU = I/U
    
    print(mIOU, bn)
    
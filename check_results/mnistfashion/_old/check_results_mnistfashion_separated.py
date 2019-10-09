#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:12:10 2018

@author: avelinojaver
"""
import sys
from pathlib import Path

dname = Path(__file__).resolve().parents[3]
sys.path.append(str(dname))


from semantic_filtering.models import UNet
from semantic_filtering.trainer import log_dir_root_dflt
from semantic_filtering.flow import MNISTFashionFlow

import torch
import numpy as np
import cv2
import tqdm
#%%
if __name__ == '__main__':
    n_ch_in, n_ch_out  = 1, 2
    
    
    bn = 'fmnist-separated-out_unet_l1smooth_20190222_155550_adam_lr0.00032_wd0.0_batch32'
    model_path = log_dir_root_dflt / 'fmnist' / bn / 'checkpoint.pth.tar'
    
    
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    mnistfashion_params = dict(
             fg_n_range = (1, 5),
             bg_n_range = (1, 25),
             int_range = (0.5, 1.1),
             epoch_size = 10240,
             output_size = 256,
             is_h_flip = True,
             is_v_flip = True,
             max_rotation = 90,
             max_overlap = 0.25
            )
    
    gen = MNISTFashionFlow(is_clean_output = True,  **mnistfashion_params)
    gen.test()
    #%%
    
    
    
    #%%
    #is_plot = False; n_iter = 1000
    is_plot = True; n_iter = 5
    
    
    th = 0.1
    all_J = []
    
    _bad = []
    for ibatch, (xin, x_true) in tqdm.tqdm(enumerate(gen)):
        
        with torch.no_grad():
            X = torch.from_numpy(xin[None])
            Xhat = model(X)
            
        xin = xin.squeeze()
        x_true = x_true.squeeze()
        x_pred = Xhat.squeeze().detach().numpy()
        
        th_in = x_true > th
        th_out = x_pred[0] > th
        
        U = (th_in | th_out)
        I = (th_in & th_out)
        
        all_J.append((I.sum(), U.sum()))
        
        
        IoU = I.sum()/U.sum()
        
        #if IoU < 0.8:
        #    _bad.append((xin, x_true, x_pred))
        
        if is_plot:
            
            import matplotlib.pylab as plt
            fig, axs = plt.subplots(2,3,sharex=True, sharey=True, figsize = (14, 8))
            axs[0][0].imshow(xin, cmap='gray', vmin=0, vmax=1)
            axs[0][1].imshow(x_true, cmap='gray', vmin=0, vmax=1)
            axs[0][2].imshow(th_in)
            
            
            axs[1][0].imshow(x_pred[1], cmap='gray', vmin=0, vmax=1)
            axs[1][1].imshow(x_pred[0], cmap='gray', vmin=0, vmax=1)
            
            axs[1][2].imshow(th_out)
            

            
            
            
            axs[0][0].set_title('Input')
            axs[0][1].set_title('Ground Truth (GT)')
            axs[0][2].set_title(f'GT > {th}')
            
            axs[1][0].set_title('Prediction (P1)')
            axs[1][1].set_title('Prediction (P2)')
            axs[1][2].set_title(f'P>{th}, IoU={IoU:.3}')
            
            for ax in axs.flatten():
                ax.axis('off')
        
        
        if ibatch >= n_iter:
            break
        
    I, U = map(np.sum, zip(*all_J))
    mIOU = I/U
    
    print(mIOU, bn)
    
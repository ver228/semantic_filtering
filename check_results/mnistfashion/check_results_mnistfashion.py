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
from bgnd_removal.trainer import log_dir_root_dflt
from bgnd_removal.flow import MNISTFashionFlow

import torch
import numpy as np
import cv2
#%%
if __name__ == '__main__':
    n_ch  = 1
    
    #bn = 'mnist-fg-fix_l1smooth_20190206_113138_unet_adam_lr0.0001_wd0.0_batch8'
    #bn = 'mnist-fg-fix_l2_20190206_113252_unet_adam_lr0.0001_wd0.0_batch8'
    
    #bn = 'mnist-fg-fix_l1smooth_20190205_140554_unet_adam_lr1e-05_wd0.0_batch24'
    #bn = 'mnist-bg-fix_l1smooth_20190205_140639_unet_adam_lr1e-05_wd0.0_batch24'
    #bn = 'mnist-fg-fix_l2_20190206_090121_unet_adam_lr1e-05_wd0.0_batch24'
    
    bn = 'fmnist-fg-fix_l1smooth_20190207_134929_unet_adam_lr0.00032_wd0.0_batch32'
    #bn = 'fmnist-fg-fix_l1smooth_20190207_135812_unet_adam_lr0.00016_wd0.0_batch16'
    #bn = 'fmnist-fg-fix_l1smooth_20190207_135814_unet_adam_lr8e-05_wd0.0_batch8'
    #bn = 'fmnist-fg-fix_l1smooth_20190207_135832_unet_adam_lr4e-05_wd0.0_batch4'
    
    #bn = 'fmnist-clean-out_l1smooth_20190207_142404_unet_adam_lr0.00032_wd0.0_batch32'
    
    model_path = log_dir_root_dflt / bn.split('_')[0] / bn / 'checkpoint.pth.tar'
    
    
    model = UNet(n_channels = n_ch, n_classes = n_ch)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    
#    argkws = dict(output_size = 256,
#                         epoch_size = 10,
#                         bg_n_range = (5, 25),
#                         int_range = (1., 1.),
#                         max_rotation = 0,
#                         is_v_flip = False)
    argkws = dict(output_size = 256)
    
    gen = MNISTFashionFlow(is_clean_output = True, fg_n_range = (1, 5), **argkws)
    gen.test()
    #%%
    import tqdm
    from skimage.measure import compare_ssim
    
    
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
        th_out = x_pred > th
        
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
            
            
            axs[1][0].imshow(x_pred - x_true)
            axs[1][1].imshow(x_pred, cmap='gray', vmin=0, vmax=1)
            
            axs[1][2].imshow(th_out)
            

            
            
            
            axs[0][0].set_title('Input')
            axs[0][1].set_title('Ground Truth (GT)')
            axs[0][2].set_title(f'GT > {th}')
            
            axs[1][0].set_title('GT - P')
            axs[1][1].set_title('Prediction (P)')
            axs[1][2].set_title(f'P>{th}, IoU={IoU:.3}')
            
            for ax in axs.flatten():
                ax.axis('off')
        
        
        if ibatch >= n_iter:
            break
        
    I, U = map(np.sum, zip(*all_J))
    mIOU = I/U
    
    print(mIOU, bn)
    #%%
    img_bg1 = gen._create_image(gen.bg_classes, gen.bg_n_range)
    img_bg2 = gen._create_image(gen.bg_classes, gen.bg_n_range)
    valid_mask = (img_bg1>0) | (img_bg2>0)
    img_fg1 = gen._create_image(gen.fg_classes, gen.fg_n_range, valid_mask)
    
    
    bb = plt.get_cmap('Reds')(img_fg1)
    bb[..., -1] = np.where(img_fg1>0.1, 0.5, 0)
    
    
    fig, axs = plt.subplots(1,3,sharex=True, sharey=True, figsize = (14, 4))
    axs[0].imshow(img_fg1, cmap='gray')
    axs[1].imshow(img_bg1, cmap='gray')
    axs[2].imshow(img_fg1 + img_bg1, cmap='gray')
    axs[2].imshow(bb)
    
    
    for ax in axs.flatten():
        ax.axis('off')
    
    fig, axs = plt.subplots(1,3,sharex=True, sharey=True, figsize = (14, 4))
    axs[0].imshow(img_fg1, cmap='gray')
    axs[1].imshow(img_bg2, cmap='gray')
    axs[2].imshow(img_fg1 + img_bg2, cmap='gray')
    axs[2].imshow(bb)
    for ax in axs.flatten():
        ax.axis('off')
   
    #%%
    
    img_fg1 = gen._create_image(gen.fg_classes, gen.fg_n_range)
    img_fg2 = gen._create_image(gen.fg_classes, gen.fg_n_range)
    valid_mask = (img_fg1>0) | (img_fg2>0)
    img_bg1 = gen._create_image(gen.bg_classes, gen.bg_n_range, valid_mask)
    
    
    
    
    
    fig, axs = plt.subplots(1,3,sharex=True, sharey=True, figsize = (14, 4))
    axs[0].imshow(img_fg1, cmap='gray')
    axs[1].imshow(img_bg1, cmap='gray')
    axs[2].imshow(img_fg1 + img_bg1, cmap='gray')
    
    bb = plt.get_cmap('Reds')(img_fg1)
    bb[..., -1] = np.where(img_fg1>0.1, 0.5, 0)
    axs[2].imshow(bb)
    
    
    for ax in axs.flatten():
        ax.axis('off')
    
    fig, axs = plt.subplots(1,3,sharex=True, sharey=True, figsize = (14, 4))
    axs[0].imshow(img_fg2, cmap='gray')
    axs[1].imshow(img_bg1, cmap='gray')
    axs[2].imshow(img_fg2 + img_bg2, cmap='gray')
    
    bb = plt.get_cmap('Reds')(img_fg2)
    bb[..., -1] = np.where(img_fg2>0.1, 0.5, 0)
    axs[2].imshow(bb)
    for ax in axs.flatten():
        ax.axis('off')
   
    #%%
    #from skimage.measure import compare_ssim
#    #%%
#    for xin, x_true, x_pred in _bad:
#        th_in = x_true > th
#        th_out = x_pred > th
#        
#        U = (th_in | th_out)
#        I = (th_in & th_out)
#        
#        fig, axs = plt.subplots(2,3,sharex=True, sharey=True, figsize = (14, 8))
#        axs[0][0].imshow(xin, cmap='gray', vmin=0, vmax=1)
#        axs[0][1].imshow(x_true, cmap='gray', vmin=0, vmax=1)
#        axs[0][2].imshow(th_in)
#        
#        
#        axs[1][0].imshow(x_pred - x_true)
#        axs[1][1].imshow(x_pred, cmap='gray', vmin=0, vmax=1)
#        
#        axs[1][2].imshow(th_out)
#        
#
#        
#        
#        
#        axs[0][0].set_title('Input')
#        axs[0][1].set_title('Ground Truth (GT)')
#        axs[0][2].set_title(f'GT > {th}')
#        
#        axs[1][0].set_title('GT - P')
#        axs[1][2].set_title('Prediction (P)')
#        axs[1][2].set_title(f'P>{th}, IoU={IoU:.3}')
        
    
    
    
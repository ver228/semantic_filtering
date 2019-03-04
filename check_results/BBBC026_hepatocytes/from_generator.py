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

#%%
from bgnd_removal.models import UNet
from bgnd_removal.flow import FluoMergedFlow, FluoSyntheticFlow
from torch.utils.data import DataLoader

import numpy as np
import torch
import tqdm

def _estimate_mIoU(model_path):
    model = UNet(n_channels = n_ch, n_classes = n_ch)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model = model.to(device)
    model.eval()
    
    
    root_dir = Path.home() / 'workspace/denoising/data/BBBC042/cell_bgnd_divided/train/'
    gen = FluoMergedFlow(root_dir = root_dir,
                         epoch_size = 4096,
                     crop_size = (128, 128),
                     is_log_transform = False,
                     int_scale = (0, 255),
                     fgnd_prefix = 'foreground',
                     bgnd_prefix = 'background',
                     img_ext = '*.tif',
                     is_timeseries_dir = False,
                     n_cells_per_crop = 3,
                     int_factor = (0.8, 1.2),
                     bgnd_sigma_range = (0., 1.)
                     )  
        
    loader = DataLoader(gen, 
                        batch_size=batch_size, 
                        shuffle=True,
                        num_workers = 4
                        )
    
    I_all = 0
    U_all = 0
    for Xin, Xout in tqdm.tqdm(loader):
        with torch.no_grad():
            Xin = Xin.to(device)
            
            Xhat = model(Xin)
        
            pred_bw = (Xhat.cpu()>th)
            target_bw = Xout > 0
            
            I = (target_bw & pred_bw)
            U = (target_bw | pred_bw)
            
            I_all += I.sum().item()
            U_all += U.sum().item()
    
    mIoU = I_all/U_all
    
    return mIoU


#%%
if __name__ == '__main__':
    n_ch  = 1
    
    #bn = 'BBBC026-hepatocytes_l1smooth_20190218_222644_unet_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-hepatocytes-log_l1smooth_20190219_111711_unet_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-hepatocytes-log_l1smooth_20190219_111711_unet_adam_lr0.00032_wd0.0_batch32'
    
    bn = 'sBBBC026-hepatocytes_l1smooth_20190219_175950_unet_adam_lr0.00032_wd0.0_batch32'
    #bn = 'sBBBC026-fibroblasts_l1smooth_20190219_201226_unet_adam_lr0.00032_wd0.0_batch32'
    
    #model_path = Path.home() / 'workspace/denoising/results' / bn.split('_')[0] / bn / 'checkpoint.pth.tar'
    model_path =  Path('/Volumes/loco/workspace/denoising/results/') / bn.split('_')[0] / bn / 'checkpoint.pth.tar'
    
    
    batch_size = 16
    th = 0.425
    epoch_size = 1000
    cuda_id = 0
    
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    
    model = UNet(n_channels = n_ch, n_classes = n_ch)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model = model.to(device)
    model.eval()
    
    
    is_log_transform = bn.split('_')[0].endswith('-log')
    
    if is_log_transform:
        int_scale = (0, np.log(2**8))
        
    else:
        int_scale = (0,255)
    
    
    #root_dir = Path.home() / 'workspace/denoising/data/BBBC026/hepatocytes/test/'
    #root_dir = '/Volumes/loco/workspace/denoising/data/BBBC026/fibroblasts/test/'
    #root_dir = '/Volumes/loco/workspace/denoising/data/BBBC026/hepatocytes/train/'
#    gen = FluoMergedFlow(root_dir = root_dir,
#                             crop_size = (256, 256),
#                             is_log_transform = is_log_transform,
#                             int_scale = int_scale,
#                             fgnd_prefix = 'foreground',
#                             bgnd_prefix = 'background',
#                             img_ext = '*.png',
#                             is_timeseries_dir = False,
#                             n_cells_per_crop = 10,
#                             int_factor = (0.5, 2.),
#                             bgnd_sigma_range = (0., 1.2),
#                             frac_crop_valid = 0.8,
#                             zoom_range = (0.75, 1.25),
#                             poisson_noise_range = (0., 5.),
#                             rotate_range = (0, 90),
#                             is_return_clean = True
#                             )  
    #%%
    root_dir = '/Volumes/loco/workspace/denoising/data/BBBC026/'
    
    
    if 'fibroblasts' in bn:
        fgnd_prefix = 'fibroblasts/train/foreground'
        bgnd_prefix = 'hepatocytes/train/foreground'
    elif 'hepatocytes' in bn:
        bgnd_prefix = 'fibroblasts/train/foreground'
        fgnd_prefix = 'hepatocytes/train/foreground'
    else:
        ValueError(bn)
                 
    
    gen = FluoSyntheticFlow(root_dir = root_dir,
                                bgnd_prefix = bgnd_prefix,
                                fgnd_prefix = fgnd_prefix,
                                 crop_size = (256, 256),
                                 is_log_transform = is_log_transform,
                                 int_scale = int_scale,
                                 img_ext = '*.png',
                                 is_timeseries_dir = False,
                                 n_cells_per_crop = 10,
                                 int_factor = (0.5, 2.),
                                 bgnd_sigma_range = (0., 1.2),
                                 frac_crop_valid = 0.8,
                                 zoom_range = (0.75, 1.25),
                                 poisson_noise_range = (0., 5.),
                                 rotate_range = (0, 90),
                                 bngd_base_range = (10, 40),
                                 is_return_clean = True
                             )  
    
    loader = DataLoader(gen, 
                        batch_size=batch_size, 
                        shuffle=True,
                        num_workers = 4
                        )
    
    for Xin, Xout in tqdm.tqdm(loader):
        with torch.no_grad():
            Xin = Xin.to(device)
            
            Xhat = model(Xin)
        
        xin = Xin.detach().cpu().numpy().squeeze(axis=1)
        xtarget = Xout.detach().cpu().numpy().squeeze(axis=1)
        xhat = Xhat.detach().cpu().numpy().squeeze(axis=1)
        #%%
        for (x,xt, y) in zip(xin, xtarget, xhat):
            fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
            axs[0].imshow(x, 'gray')
            axs[1].imshow(xt, 'gray')
            axs[2].imshow(y, 'gray')
        #%%
        break
    
    
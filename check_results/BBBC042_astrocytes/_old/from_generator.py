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
from bgnd_removal.flow import BFFlow
from torch.utils.data import DataLoader

import torch
import tqdm



#%%
if __name__ == '__main__':
    n_ch  = 1
    
    
    #bn = 'BBBC042_l1smooth_20190215_213644_unet_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC042_l1smooth_20190216_111113_unet_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC042_l1smooth_20190216_190337_unet_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC042_l1smooth_20190217_005342_unet_adam_lr0.00032_wd0.0_batch32'
    
    bn = 'BBBC042_l1smooth_20190220_184344_unet_adam_lr0.00032_wd0.0_batch32'
    
    model_path = Path.home() / 'workspace/denoising/results' / bn.split('_')[0] / bn / 'checkpoint.pth.tar'
    #model_path =  Path('/Volumes/loco/workspace/denoising/results/') / bn.split('_')[0] / bn / 'checkpoint.pth.tar'
    #model_path =  Path('/Volumes/loco/workspace/denoising/results/') / bn.split('_')[0] / bn / 'checkpoint-59.pth.tar'
    
    
    
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
    
    
    root_dir = Path.home() / 'workspace/denoising/data/BBBC042/cell_bgnd_divided_v2/train/'
    #root_dir = '/Volumes/loco/workspace/denoising/data/BBBC042/cell_bgnd_divided/train/'
    #root_dir = '/Volumes/loco/workspace/denoising/data/BBBC042/cell_bgnd_divided/test/'
    
    gen = BFFlow(epoch_size = 20480,
                     root_dir = root_dir,
                     crop_size = (256, 256),
                         is_log_transform = False,
                         int_scale = (0, 255),
                         fgnd_prefix = 'foreground',
                         bgnd_prefix = 'background',
                         img_ext = '*.tif',
                         is_timeseries_dir = False,
                         n_cells_per_crop = 3,
                         int_factor = (1., 1.),
                         bgnd_sigma_range = (0., 1.),
                         merge_type = 'replace',
                         frac_crop_valid = 0.,
                         is_return_clean = True,
                         noise_range = (0., 10.)
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
    
    
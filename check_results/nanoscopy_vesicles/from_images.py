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
#%%
if __name__ == '__main__':
    n_ch  = 1
    
   
    bn = 'nanoscopy-vesicles_l2_20190211_144325_unet_adam_lr0.00032_wd0.0_batch32'
    #bn = 'nanoscopy-vesicles-log_l2_20190211_150333_unet_adam_lr0.00032_wd0.0_batch32'
    
    #model_path = Path.home() / 'workspace/denoising/results' / bn.split('_')[0] / bn / 'checkpoint.pth.tar'
    model_path = Path.home() / 'workspace/denoising/results' / bn.split('_')[0] / bn / 'checkpoint-5.pth.tar'
    
    
    if '-log' in bn:
        is_log_transform = True
        int_scale = (0, np.log(2**16))
    
    else:
        is_log_transform = False
        int_scale = (0, 2**16-1)
    
    
    #model_path = f'/Volumes/rescomp1/data/microglia-fluo/{bn}/checkpoint-599.pth.tar'
    #model_path = f'/Volumes/rescomp1/data/denoising/results/microglia-fluo/{bn}/checkpoint.pth.tar'
    
    
    
    
    #gen = SyntheticFluoFlow()
    model = UNet(n_channels = n_ch, n_classes = n_ch)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    import tifffile
    import json
    
    tracks_file = None
    
#    src_file = '/Volumes/rescomp1/data/nanoscopy/raw_data/50min_bio_10fps_Airyscan_ProcessingSetting3-3.tif'
#    tracks_file = '/Volumes/rescomp1/data/nanoscopy/raw_data/50min_bio_10fps_Airyscan_ProcessingSetting3-3_detections.txt'
#    
#    src_file = '/Volumes/rescomp1/projects/nanoscopy_denoising/20181207/000 - Fas3 long test - Camera 02.tif'
    src_file = '/Volumes/rescomp1/projects/nanoscopy_denoising/20181207/005 - Fas3_2 long test - Camera 02.tif'
#    
    if tracks_file is not None:
        with open(tracks_file, 'r') as fid:
            coords = json.load(fid)
        
    
    imgs = tifffile.imread(src_file)
    #%%
    frame_number = 50
    
    img = imgs[frame_number]
    
    

    
    img = img[None]
    
    x = img.astype(np.float32)
    
    top, bot = x.max(), x.min()
    x = (x - bot)/(top-bot)
    
#    if is_log_transform:
#        x = np.log(x+1)
#    x = (x - int_scale[0])/(int_scale[1] - int_scale[0])
    
    with torch.no_grad():
        X = torch.from_numpy(x[None])
        Xhat = model(X)
    #%%
    
    xhat = Xhat.squeeze().detach().numpy()
    xr = x.squeeze()
    
    fig, axs = plt.subplots(1,3,sharex=True, sharey=True)
    
    axs[0].imshow(xr, cmap='gray')#, vmin=0, vmax=1)
    axs[1].imshow(np.log(xhat+1), cmap='gray')#, vmin=0, vmax=1)
    
    bot, top = xhat.min(), xhat.max()
    cc = (xhat-bot)/(top-bot)
    b_rgb = plt.get_cmap('magma')(cc)
    b_rgb[..., -1] = 0.5
    
    
    bot, top = xr.min(), xr.max()
    mm = (xr-bot)/(top-bot)
    mm = plt.get_cmap('gray')(mm)
    
    axs[2].imshow(mm)
    axs[2].imshow(b_rgb)
    
    
    if tracks_file is not None:
        cc = coords[str(frame_number)]
        cy, cx = map(np.array, zip(*cc))
        axs[2].plot(cx, cy, 'r.')
    #%%
    
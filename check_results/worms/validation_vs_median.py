#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:17:06 2019

@author: avelinojaver
"""
import sys
from pathlib import Path

dname = Path(__file__).resolve().parents[2]
sys.path.append(str(dname))

from semantic_filtering.models import UNet

from pathlib import Path
import tables
import torch
import numpy as np
import matplotlib.pylab as plt
#%%
if __name__ == '__main__':
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/manual_annotations/raw/Phase3/MaskedVideos/'
    root_dir = Path(root_dir)
    
    #fname = root_dir / 'Set2_N2_PS3398_CB369_PS3398_Ch2_03072018_170224.hdf5'
    fname = root_dir / 'Set2_CB369_PS3398_Ch1_22062018_122500.hdf5'
    #fname = root_dir / 'Set1_CB369_CB1490_Ch1_03072018_163429.hdf5'
    
    save_dir = Path.home() / 'OneDrive - Nexus365/papers/miccai2019/data/worms/'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with tables.File(fname, 'r') as fid:
        imgs = fid.get_node('/full_data')[:]
    
    img_m = np.max(imgs, axis=0)
    
    x = imgs[-1]
    dd = x.astype(np.float32) - img_m

    fig, axs = plt.subplots(1, 3, figsize = (20, 8), sharex = True, sharey = True)
    axs[0].imshow(x, cmap='gray')
    axs[1].imshow(img_m, cmap='gray')
    axs[2].imshow(dd, cmap='gray')
    
    plt.suptitle((len(imgs), fname.name))
    plt.xlim(550, 1650)
    plt.ylim(500, 1500)
    
    for ax in axs:
        ax.axis('off')
    
    fig.savefig(str(save_dir / 'median.pdf'))
    
    #%%
    
    bn = 'worms-divergent_l1smooth_20190201_011918_unet_adam_lr0.0001_wd0.0_batch36'
    model_path = f'/Volumes/rescomp1/data/denoising/results/worms-divergent/{bn}/checkpoint.pth.tar'
    
    scale_log = (0, 255)
    model = UNet(n_channels = 1, n_classes = 1)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    x = imgs[-1].astype(np.float32)
    x = (x - scale_log[0])/(scale_log[1] - scale_log[0])
    
    
    with torch.no_grad():
        X = torch.from_numpy(x[None, None])
        Xhat = model(X)
    
    xhat = Xhat.squeeze().detach().numpy()
    xr = x.squeeze()
    #%%
    fig, axs = plt.subplots(1, 3, figsize = (20, 8), sharex = True, sharey = True)
    axs[0].imshow(xr, cmap='gray', vmin=0, vmax=1)
    axs[1].imshow(xhat, cmap='gray', vmin=0, vmax=1)
    axs[2].imshow(xr-xhat, cmap='gray')
    
    plt.suptitle((len(imgs), fname.name))
    
    plt.xlim(550, 1650)
    plt.ylim(500, 1500)
    
    for ax in axs:
        ax.axis('off')
    fig.savefig(str(save_dir / 'NN.pdf'))
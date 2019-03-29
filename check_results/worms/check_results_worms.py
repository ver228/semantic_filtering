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
    #new
    
    #bn = 'worms-divergent_l1_20190201_011625_unet_adam_lr0.0001_wd0.0_batch36'
    #bn = 'worms-divergent_l2_20190201_011853_unet_adam_lr0.0001_wd0.0_batch36'
    bn = 'worms-divergent_l1smooth_20190201_011918_unet_adam_lr0.0001_wd0.0_batch36'
    
    model_path = f'/Volumes/rescomp1/data/denoising/results/worms-divergent/{bn}/checkpoint.pth.tar'
    
    
    scale_log = (0, 255)
    model = UNet(n_channels = 1, n_classes = 1)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    #fname = '/Volumes/rescomp1/data/denoising/data/c_elegans_divergent/test/MaskedVideos/CeNDR_Set2_220617/LKC34_worms5_food1-10_Set6_Pos5_Ch2_22062017_144037/2.tif'
    #fname = '/Volumes/rescomp1/data/denoising/data/c_elegans_divergent/test/MaskedVideos/CeNDR_Set2_060717/CX11314_worms10_food1-10_Set11_Pos4_Ch1_06072017_160129/0.tif'
    #fname = Path.home() / 'workspace/denoising/data/c_elegans_divergent/test/MaskedVideos/CeNDR_Set2_060717/MY16_worms10_food1-10_Set11_Pos4_Ch4_06072017_160117/4.tif'
    #fname = Path.home() / 'workspace/denoising/data/c_elegans_divergent/test/MaskedVideos/CeNDR_Set2_060717/JT11398_worms5_food1-10_Set7_Pos4_Ch3_06072017_144112/1.tif'
    
    #fname = Path.home() / 'workspace/denoising/data/c_elegans_divergent/test/MaskedVideos/CeNDR_Set2_060717/CB4856_worms5_food1-10_Set7_Pos4_Ch5_06072017_144104/3.tif'
    #fname = Path.home() / 'workspace/denoising/data/c_elegans_divergent/test/MaskedVideos/CeNDR_Set2_060717/ED3017_worms5_food1-10_Set12_Pos5_Ch2_06072017_162253/2.tif'
    #fname =  Path.home() / 'workspace/denoising/data/c_elegans_divergent/test/MaskedVideos/CeNDR_Set2_220617/MY16_worms5_food1-10_Set6_Pos5_Ch3_22062017_144023/2.tif'
    #fname = Path.home() / 'workspace/denoising/data/c_elegans_divergent/test/MaskedVideos/CeNDR_Set2_060717/EG4725_worms10_food1-10_Set9_Pos4_Ch4_06072017_152127/4.tif'
    
    #fname = '/Volumes/rescomp1/data/denoising/data/c_elegans_full/test/MMP/MaskedVideos/MMP_Set2_201217/VC20527_worms10_food1-10_Set2_Pos5_Ch3_20122017_152030/1.tif'
    
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/manual_annotations/raw/Phase3/MaskedVideos/Set1_CB369_CB1490_Ch1_03072018_163429.hdf5'
    #fname = '/Volumes/rescomp1/data/denoising/data/c_elegans_divergent/test/MaskedVideos/CeNDR_Set2_060717/CB4856_worms5_food1-10_Set7_Pos4_Ch5_06072017_144104/3.tif'
    fname = '/Volumes/rescomp1/data/denoising/data/c_elegans_divergent/test/MaskedVideos/CeNDR_Set2_060717/JU775_worms10_food1-10_Set9_Pos4_Ch2_06072017_152141/_0.tif'
    #fname = '/Volumes/rescomp1/data/denoising/data/c_elegans_divergent/test/MaskedVideos/CeNDR_Set2_060717/DL238_worms10_food1-10_Set3_Pos4_Ch1_06072017_124159/_0.tif'
    fname = Path(fname)
    
    if fname.suffix == '.hdf5':
        import tables
        with tables.File(fname, 'r') as fid:
            img = fid.get_node('/full_data')[0]
    else:
        img = cv2.imread(str(fname), -1)
    #cv2.resize(img, dsize = tuple([x//4 for x in img.shape]))
    
    fname_r = fname.parent / ('R_' + fname.stem + '.png')
    
    if fname_r.exists():
        target = cv2.imread(str(fname_r), -1)
        target_bw = (target>0)
    else:
        target_bw = np.zeros_like(img)
    
    #%%
    x = img[None].astype(np.float32)
    
    #x = np.log(x+1)
    x = (x - scale_log[0])/(scale_log[1] - scale_log[0])
    
    
    with torch.no_grad():
        X = torch.from_numpy(x[None])
        Xhat = model(X)
    
    xhat = Xhat.squeeze().detach().numpy()
    xr = x.squeeze()
    
    #%%
    #th = 0.08
    
    
    x_diff =  (xhat - xr) 
    
    #med = np.median(x_diff)
    #mad = np.median(np.abs(med - x_diff))
    #th = med + 6*mad
    
    th = 10/255
    pred_bw = x_diff > th
    I = (target_bw & pred_bw).sum()
    U = (target_bw | pred_bw).sum()
    
    fig, axs = plt.subplots(1,4,sharex=True, sharey=True)
    
    vmax = max(np.max(xr), np.max(xhat))
    vmin = min(np.min(xr), np.min(xhat))
    
    axs[0].imshow(xr, cmap='gray', vmin=0, vmax=1)
    axs[0].set_title('Input')
    axs[1].imshow(xhat, cmap='gray', vmin=0, vmax=1)
    axs[1].set_title('Output')
    axs[2].imshow(pred_bw, cmap='gray')
    axs[2].set_title(f'(Input  - Output) > {th}')
    
    axs[3].imshow(xr-xhat, cmap='gray')
    #axs[3].set_title('Target')
    
    for ax in axs:
        ax.axis('off')
        
        #%%
        
        
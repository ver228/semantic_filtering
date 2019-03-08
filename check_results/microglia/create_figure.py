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
import matplotlib.pylab as plt

#from skimage.filters import threshold_otsu
#%%
if __name__ == '__main__':
    scale_int = (0, np.log(2**16))
    
    model_names = {'noise2noise':'microglia-v2_unet_l1smooth_20190228_141742_adam_lr0.00012_wd0.0_batch12', 
              'demixer' : 'microglia-v2-separated_unet_l1smooth_20190302_214016_adam_lr0.00012_wd0.0_batch12'
              }
             
    fname = '/Users/avelinojaver/OneDrive - Nexus365/microglia/data/movies/2018.08.20_movies/180820_MicVid_20X_30s/180815_MicVid_20X_30s_20X_30s_4w/H - 12(fld 1 z 1- time 1 - 0 ms).tif'
    img_ori = cv2.imread(fname, -1)[..., ::-1]
    
    x = img_ori[None].astype(np.float32)
    x = np.log(x+1)
    x = (x - scale_int[0])/(scale_int[1] - scale_int[0])
    
    results = {}
    for k, bn in model_names.items():
    
        #model_path = Path('/Volumes/loco/') / 'workspace/denoising/results' / 'microglia' / bn / 'checkpoint-299.pth.tar'
        model_path = Path.home() / 'workspace/denoising/results/microglia' / bn / 'checkpoint.pth.tar'
    
        n_ch_in, n_ch_out  = 1, 1
        if '-separated' in bn:
            n_ch_out = 3
        model = UNet(n_channels = n_ch_in, n_classes = n_ch_out)
        
        
        state = torch.load(model_path, map_location = 'cpu')
        model.load_state_dict(state['state_dict'])
        model.eval()
        
        with torch.no_grad():
            X = torch.from_numpy(x[None])
            Xhat = model(X)
        
        xhat = Xhat[0].detach().numpy()
        
        results[k] = xhat

    #%%
    fig, axs = plt.subplots(1,4,sharex=True, sharey=True, figsize=(20, 5))
    
    axs[0].imshow(img_ori, cmap='gray', interpolation=None)
    axs[0].set_title('Original')
    axs[1].imshow(x[0], cmap='gray', interpolation=None)
    axs[1].set_title('Log-Trasform')
    axs[2].imshow(results['noise2noise'][0], cmap='gray', interpolation=None)
    axs[2].set_title('Noise2Noise')
    axs[3].imshow(results['demixer'][0], cmap='gray', interpolation=None)
    axs[3].set_title('Demixer')
    for ax in axs:
        ax.axis('off')
    
    ax.set_xlim(525, 1025)
    ax.set_ylim(0, 500)
    
    save_name =   '/Users/avelinojaver/OneDrive - Nexus365/papers/miccai2019/data/microglia/microglia.pdf'
    fig.savefig(save_name)
    
    
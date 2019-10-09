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
from semantic_filtering.models import UNet
from semantic_filtering.flow import FluoMergedFlow
from torch.utils.data import DataLoader

import torch
import numpy as np
import tqdm
import cv2

def _estimate_mIoU(model_path):
    model = UNet(n_channels = n_ch, n_classes = n_ch)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model = model.to(device)
    model.eval()
    
    
    gen = FluoMergedFlow(root_dir = root_dir,
                         is_return_truth = True,
                         epoch_size = epoch_size
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
    scale_log = (0, np.log(2**16))
    
    root_dir = Path.home() / 'workspace/denoising/data/microglia/cell_bgnd_divided/test/'
    results_root = Path.home() / 'workspace/denoising/results/'
    
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
    
    #old
    #model_path = '/Volumes/rescomp1/data/denoising/_old_results/microglia_syntethic/microglia_synthetic_l1smooth_20181003_161017_unet_adam_lr0.0001_wd0.0_batch16/model_best.pth.tar'
    
    #new
    #bn = 'microglia-fluo_l2_20190131_164721_unet_adam_lr0.0001_wd0.0_batch12'
    #bn = 'microglia-fluo_l1_20190131_165128_unet_adam_lr0.0001_wd0.0_batch12'
    #bn = 'microglia-fluo_l1smooth_20190131_164723_unet_adam_lr0.0001_wd0.0_batch12'
    
    
    bn = 'microglia-fluo_l2_20190201_172055_unet_adam_lr0.0001_wd0.0_batch12'
    #bn = 'microglia-fluo_l1_20190201_171947_unet_adam_lr0.0001_wd0.0_batch12'
    #bn = 'microglia-fluo_l1smooth_20190201_172003_unet_adam_lr0.0001_wd0.0_batch12'
 
    
    base_dir =  results_root / 'microglia-fluo' / bn 
    
    all_results = []
    for model_path in base_dir.glob('checkpoint*.pth.tar'):
        mIoU = _estimate_mIoU(model_path)
        
        ss = model_path.parent.name + '/' + model_path.name
        all_results.append((ss, mIoU))
        
        print((ss, mIoU))
        
    all_results = sorted(all_results, key = lambda x : x[0])
    
    for ss, mIoU in all_results:
        print(mIoU, ss)
        
    
        #out = zip(*[x.squeeze(dim=1).detach().numpy() for x in (Xin, Xout, Xhat)])
        
        
    
    
#    out = zip(*[x.squeeze(dim=1).detach().numpy() for x in (Xin, Xout, Xhat)])
#    for xin, xout, xhat in out:
#        fig, axs = plt.subplots(1,4, figsize = (16, 4), sharex=True, sharey=True)
#    
#        axs[0].imshow(xin, cmap='gray')#, vmin=0, vmax=1)
#        axs[1].imshow(xout, cmap='gray')#, vmin=0, vmax=1)
#        axs[2].imshow(xhat)
#        axs[3].imshow(xhat>0.425)
    
    #%%
    
    
    
    
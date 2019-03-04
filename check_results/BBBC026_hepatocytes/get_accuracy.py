#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:07:42 2019

@author: avelinojaver
"""
import sys
from pathlib import Path
dname = Path(__file__).resolve().parents[2]
sys.path.append(str(dname))

from bgnd_removal.models import UNet

import numpy as np
import torch
import cv2
import tqdm
import pandas as pd

from skimage.measure import regionprops

from get_counts import get_labels


#%%
if __name__ == '__main__':
    root_dir = Path.home() / 'workspace/datasets/BBBC026/BBBC026_GT_images'
    
    #th_min = 0.22
    #bn = 'sBBBC026-hepatocytes_l1smooth_20190220_184125_unet_adam_lr0.00032_wd0.0_batch32'
    #model_path = Path.home() / 'workspace/denoising/results' / bn.split('_')[0] / bn / 'checkpoint-199.pth.tar'
    
    
    
    
    th_min = 0.
    #bn = 'BBBC026-fibroblasts_unet_l1smooth_20190222_183808_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-separated_unet_l1smooth_20190222_184510_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-hepatocytes_unet_l1smooth_20190222_183622_adam_lr0.00032_wd0.0_batch32'
    
    #bn = 'BBBC026-separated_unet_l1smooth_20190223_221251_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-hepatocytes_unet_l1smooth_20190223_220901_adam_lr0.00032_wd0.0_batch32'
    
    #bn = 'BBBC026-separated_unet_l1smooth_20190224_105903_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-hepatocytes_unet_l1smooth_20190224_105953_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-fibroblasts_unet_l1smooth_20190224_105911_adam_lr0.00032_wd0.0_batch32'
    
    #bn = 'BBBC026-separated_unet_l1smooth_20190225_235932_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-hepatocytes_unet_l1smooth_20190226_000010_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-fibroblasts_unet_l1smooth_20190225_235927_adam_lr0.00032_wd0.0_batch32'
    
    bn = 'BBBC026-separated_unet_l1smooth_20190226_082017_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-hepatocytes_unet_l1smooth_20190226_082040_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-fibroblasts_unet_l1smooth_20190226_082007_adam_lr0.00032_wd0.0_batch32'
    
    model_path = Path.home() / 'workspace/denoising/results/BBBC026' / bn / 'checkpoint.pth.tar'
    
    n_ch_in, n_ch_out  = 1, 1
    if '-separated' in bn:
        n_ch_out = 3
    
    cuda_id = 0
    min_area = 300
    _debug = True
    
    save_name = Path.home() / 'workspace' / ('CELLS_' + bn + '.csv')
    
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model = model.to(device)
    model.eval()
    
    fnames = [x for x in Path(root_dir).glob('*.png') if not x.name.startswith('.')]
    fnames = sorted(fnames, key = lambda x : x.name)
    
    all_data = []
    for fname in tqdm.tqdm(fnames):
        
        img = cv2.imread(str(fname), -1)
        
        img_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
        bad = (img[..., 0] == 255) & (img[..., 1] <= 10) & (img[..., 2] <= 10)
        fib =  (img[..., 0] <= 10) & (img[..., 1] == 255) & (img[..., 2] <= 10)
        hep =  (img[..., 0] <= 10) & (img[..., 1] <= 10) & (img[..., 2] == 255)
        
        cm_fib = cv2.connectedComponentsWithStats(fib.astype(np.uint8))[-1][1:].astype(np.int)
        cm_hep = cv2.connectedComponentsWithStats(hep.astype(np.uint8))[-1][1:].astype(np.int)
        cm_bad = cv2.connectedComponentsWithStats(bad.astype(np.uint8))[-1][1:].astype(np.int)
        
        peaks2remove = bad | fib | hep
        med = cv2.medianBlur(img_g, ksize= 11) + np.random.normal(0, 2, img_g.shape).round().astype(np.int)
        img_g[peaks2remove] = med[peaks2remove]
        img = img_g
        
        x = img[None].astype(np.float32)
        
        pix_top = np.percentile(x, 99)
        xn = x/pix_top
        
        
        with torch.no_grad():
            X = torch.from_numpy(xn[None])
            X = X.to(device)
            Xhat = model(X)
    
        xhat = Xhat[0].detach().cpu().numpy()
        #%
        labels, th = get_labels(xhat[0], th_min = 0.)
        props = regionprops(labels)
        
        centroids = [x.centroid for x in props if x.area > min_area]
        
        #%%
        from scipy.spatial.distance import cdist
        
        
        cm_pred = np.array(centroids)[:, ::-1]
        #%%
        labels = np.zeros(len(cm_pred), np.int)
        
        max_dist = 20
        #dirty test, there are a lot of things that could go wrong here...
        for ilab, dat in enumerate([cm_fib, cm_bad, cm_hep]):
            if dat.size:
                dist = cdist(cm_pred, dat)
                true_ind = np.argmin(dist, axis=1)
                pred_ind = np.arange(cm_pred.shape[0])
                pair_dist = dist[pred_ind, true_ind]
                valid = pair_dist < max_dist
            
                labels[valid] = ilab + 1
        
        
        #%%
        counts = np.bincount(labels)
        recall = counts[-1]/len(cm_hep)
        precisions = counts/len(labels)
        
        print(fname.name, recall, precisions)
        
        
        #%%
        if _debug:
            if centroids:
                
                y, x = np.array(centroids).T
            else:
                x,y = [], []
            
            import matplotlib.pylab as plt
            fig, axs = plt.subplots(1,4,sharex=True, sharey=True, figsize = (20, 10))
            axs[0].imshow(img,  cmap = 'gray')#vmax = int(pix_top)
            
            colors = 'yrgm'
            for ilab, cc in enumerate(colors):
                good = labels == ilab
                if np.any(good):
                    axs[0].plot(x[good], y[good], 'o', color=cc)
                
            axs[0].plot(cm_hep[..., 0], cm_hep[..., 1], 'xm')
            #axs[0].plot(cm_fib[..., 0], cm_fib[..., 1], 'g.')
            
            axs[1].imshow(xhat[0])
            axs[2].imshow(xhat[-1])
            axs[3].imshow(xhat[1])
            #bn = fname.name.partition('_')[-1].rpartition('_')[0]
            
            plt.suptitle(fname.name)
            #%%
            
        
        
        
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

from scipy.spatial.distance import cdist
from skimage.measure import regionprops

from get_counts import get_labels
import matplotlib.pylab as plt

#%%
if __name__ == '__main__':
    root_dir = Path.home() / 'workspace/datasets/BBBC026/BBBC026_GT_images'
    
    bn = 'BBBC026-separated_unet_l1smooth_20190226_082017_adam_lr0.00032_wd0.0_batch32'
    
    
    n_epochs = 349
    model_path = Path.home() / 'workspace/denoising/results/BBBC026' / bn / f'checkpoint-{n_epochs}.pth.tar'
    
    save_dir = '/Users/avelinojaver/OneDrive - Nexus365/papers/miccai2019/data/hepatocytes/'
    save_dir = Path(save_dir)
    
    n_ch_in, n_ch_out  = 1, 1
    if '-separated' in bn:
        n_ch_out = 3
    
    cuda_id = 0
    min_area = 300
    _debug = True
    
    
    
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
    #%%
    fname = '/Users/avelinojaver/OneDrive - Nexus365/papers/miccai2019/data/hepatocytes/BBBC026/BBBC026_GT_images/M19_s6.png'
    img = cv2.imread(str(fname), -1)
    img = img[..., :3]
    
    img_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
    bad = (img[..., 0] == 255) & (img[..., 1] <= 10) & (img[..., 2] <= 10)
    fib =  (img[..., 0] <= 10) & (img[..., 1] == 255) & (img[..., 2] <= 10)
    hep =  (img[..., 0] <= 10) & (img[..., 1] <= 10) & (img[..., 2] == 255)
    
    #ground truth
    target_coords = {}
    target_coords['fib'] = cv2.connectedComponentsWithStats(fib.astype(np.uint8))[-1][1:].astype(np.int)
    target_coords['hep'] = cv2.connectedComponentsWithStats(hep.astype(np.uint8))[-1][1:].astype(np.int)
    target_coords['bad'] = cv2.connectedComponentsWithStats(bad.astype(np.uint8))[-1][1:].astype(np.int)
    
    #remove the labelled pixels and conver the image into gray scale
    peaks2remove = bad | fib | hep
    med = cv2.medianBlur(img_g, ksize= 11) + np.random.normal(0, 2, img_g.shape).round().astype(np.int)
    img_g[peaks2remove] = med[peaks2remove]
    
    #%%
    #calculate predictions
    x = img_g[None].astype(np.float32)
    pix_top = np.percentile(x, 99)
    xn = x/pix_top
    with torch.no_grad():
        X = torch.from_numpy(xn[None])
        X = X.to(device)
        Xhat = model(X)

    xhat = Xhat[0].detach().cpu().numpy()
    
    #%%
    if 'separate' in bn:
        prediction_maps = {key:xhat[ii] for ii,key in enumerate(['hep', 'bad', 'fib'])}
    elif 'fibroblast' in bn:
        prediction_maps = {'fib':xhat[0]}
    elif 'hepatocyte' in bn:
        prediction_maps = {'hep':xhat[0]}
        
    
    segmentation_labels = {}
    for key, pred in prediction_maps.items():
        th_min = 0.5 if key == 'bad' else 0.
        
        lab, _ = get_labels(pred, th_min = th_min, min_area=min_area)
        segmentation_labels[key] = lab
    #%%
    ax_lims = {'good1':[(640, 890), (125, 375)], 
               'good2':[(800, 1050), (500, 750)],
               #'bad1' : [(240, 490), (100, 350)],
               #'bad2' : [(1100, 1350), (410,660)],
               #'bad3': [(300, 550), (790,1040)]
               'bad1' : [(330, 480), (210, 360)],
               'bad2' : [(1140, 1290), (480,630)],
               'bad3': [(280, 430), (875,1025)]
               }
    #%%
    save_dir = '/Users/avelinojaver/OneDrive - Nexus365/papers/miccai2019/data/hepatocytes/'
    save_dir = Path(save_dir)
    for k, (xl, yl) in ax_lims.items(): 
    
        colors = dict(hep='r', fib='g', u='y', bad='b')
        fig, axs = plt.subplots(1,4,sharex=True, sharey=True, figsize = (20, 10))
        for i_plot, k_pred in enumerate(['hep', 'fib', 'bad']):
            pred = prediction_maps[k_pred]
            axs[i_plot + 1].imshow(pred,  cmap = 'gray')
            
        for ax in axs:
            ax.axis('off')
            
        img_rgb = img[..., 3::-1]
        axs[0].imshow(img_rgb)
        axs[0].set_xlim(xl)
        axs[0].set_ylim(yl)
    
        fig.savefig(str(save_dir / (k + '.pdf')))
    #%%
    import tables
    fname = '/Users/avelinojaver/OneDrive - Nexus365/papers/miccai2019/data/hepatocytes/ilastik_results/manual_scaled/M19_s6_Probabilities.h5'
    with tables.File(str(fname), 'r') as fid:
        preds = fid.get_node('/exported_data')[:]
    
    pred_maps = {'hep' : preds[..., 0], 'fib' : preds[..., 1], 'bad': preds[..., 2]}
    for k, (xl, yl) in ax_lims.items(): 
    
        colors = dict(hep='r', fib='g', u='y', bad='b')
        fig, axs = plt.subplots(1,4,sharex=True, sharey=True, figsize = (20, 10))
        for i_plot, k_pred in enumerate(['hep', 'fib', 'bad']):
            pred = pred_maps[k_pred]
            axs[i_plot + 1].imshow(pred,  cmap = 'gray')
            
        for ax in axs:
            ax.axis('off')
            
        img_rgb = img[..., 3::-1]
        axs[0].imshow(img_rgb)
        axs[0].set_xlim(xl)
        axs[0].set_ylim(yl)
        
        fig.savefig(str(save_dir / ('I_' + k + '.pdf')))
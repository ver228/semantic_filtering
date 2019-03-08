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


#%%
if __name__ == '__main__':
    root_dir = Path.home() / 'workspace/datasets/BBBC026/BBBC026_GT_images'
    
    bn = 'BBBC026-separated_unet_l1smooth_20190226_082017_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-hepatocytes_unet_l1smooth_20190226_082040_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-fibroblasts_unet_l1smooth_20190226_082007_adam_lr0.00032_wd0.0_batch32'
    
    
    n_epochs = 349#299#
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
    
    fnames = [x for x in Path(root_dir).glob('*.png') if not x.name.startswith('.')]
    fnames = sorted(fnames, key = lambda x : x.name)
    
    all_outputs = []
    for fname in tqdm.tqdm(fnames):
        #%%
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
        #img = img_g
        #%%
        x = img_g[None].astype(np.float32)
        
        pix_top = np.percentile(x, 99)
        xn = x/pix_top
        
        
        with torch.no_grad():
            X = torch.from_numpy(xn[None])
            X = X.to(device)
            Xhat = model(X)
    
        xhat = Xhat[0].detach().cpu().numpy()
        #%
        labels_hep, _ = get_labels(xhat[0], th_min = 0., min_area=min_area)
        pred_hep_l = labels_hep[cm_hep[...,1], cm_hep[...,0]]
        pred_hep_l = set(pred_hep_l[pred_hep_l>0])
            
        if xhat.shape[0] == 3:
            
            labels_fib, _ = get_labels(xhat[-1], th_min = 0., min_area=min_area)
            labels_bad, _ = get_labels(xhat[1], th_min = 0.5, min_area=min_area)
        
        
        
            pred_fib_l = labels_hep[cm_fib[...,1], cm_fib[...,0]]
            pred_fib_l = set(pred_fib_l[pred_fib_l>0])
            
            pred_bad_l = labels_hep[cm_bad[...,1], cm_bad[...,0]]
            pred_bad_l = set(pred_bad_l[pred_bad_l>0])
        else:
            pred_fib_l = []
            pred_bad_l = []
        
        
        TP = len(pred_hep_l)
        tot_true = cm_hep.shape[0]
        recall = TP /  tot_true
        
        all_labs = set(np.unique(labels_hep[labels_hep>0]))
        tot_pred = len(all_labs)
        precision = TP / tot_pred
        
        missing_l = set(all_labs) - (set(pred_hep_l) | set(pred_fib_l) | set(pred_bad_l))
       
        unmarked = len(missing_l) / tot_pred
        wrong_fib = len(pred_fib_l) / tot_pred
        wrong_bad = len(pred_bad_l) / tot_pred
        unknown = len(missing_l) / tot_pred
        
        #%%
        coords_by_label = {}
        
        props = regionprops(labels_hep)
        props_l = {x.label:x.centroid for x in props}
        
        
        coords_by_label['hep'] = np.array([props_l[x] for x in pred_hep_l])
        coords_by_label['fib'] = np.array([props_l[x] for x in pred_fib_l])
        coords_by_label['bad'] = np.array([props_l[x] for x in pred_bad_l])
        coords_by_label['u'] = np.array([props_l[x] for x in missing_l])
    
        #fraction of hep labeled as fibs that were also labeled as fibs
        cc = coords_by_label['fib'].round().astype(np.int)
        
        if cc.size > 0:
            labs = labels_fib[cc[..., 0], cc[..., 1]]
            also_fibs = np.mean(labs>0)
            
            coords_by_label['overlaps'] = cc[labs>0]
        else:
            also_fibs = 0
        
        #%%
        
        #overlap = len(hep_or_fib) / tot_pred
        
        _output = [f'{fname.name}',
                   f'recall : {recall:.4}',
                   f'precision : {precision:.4} -> fib {wrong_fib:.4}; bad {wrong_bad:.4}; unknown {unknown:.4}',
                   f'overlap : {also_fibs}']
        
        _output = '\n'.join(_output)
        
        all_outputs.append(_output)
        
        
        #%%
        colors = dict(hep='m', fib='r', overlaps='g', u='y', bad='c')
        
        if _debug:
            import matplotlib.pylab as plt
            
            
            
            
            if xhat.shape[0] == 3:
                n_figs = 4
            else:
                n_figs = 2
            
            fig, axs = plt.subplots(1,n_figs,sharex=True, sharey=True, figsize = (20, 10))
            
            axs[1].imshow(xhat[0],  cmap = 'gray')
            
            if len(axs) > 2:
                axs[2].imshow(xhat[-1],  cmap = 'gray')
                axs[3].imshow(xhat[1],  cmap = 'gray')
            
            plt.suptitle(fname.name)
            
            
            if False:
                axs[0].imshow(img_g,  cmap = 'gray')
                axs[0].plot(cm_hep[..., 0], cm_hep[..., 1], 'xm')
                for (k, coords) in coords_by_label.items():
                    if coords.size == 0:
                        continue
                    axs[0].plot(coords[..., 1], coords[..., 0], 'o', color=colors[k])
            else:
                img_rgb = img[..., 3::-1]
                axs[0].imshow(img_rgb)
        
            for ax in axs:
                ax.axis('off')
        #%%
        output2save = '*****\n'.join(all_outputs)
        
        
        save_dir.mkdir(parents=True, exist_ok=True)
    
        save_name = save_dir / f'accuracies_{n_epochs}_{bn}.txt'
        with open(save_name, 'w') as fid:
            fid.write(output2save)
        
        
            #%%
            
        
        
        
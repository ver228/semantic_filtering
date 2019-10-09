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

from semantic_filtering.models import UNet

import numpy as np
import torch
import cv2
import tqdm
import pandas as pd

from skimage.morphology import watershed
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
#from scipy.ndimage.morphology import binary_fill_holes

def get_labels(x2th, th_min = 0., min_area = 0., max_area = 1e20):
    #%%
    
    th = threshold_otsu(x2th)
    th = max(th_min, th)
    mask = (x2th>th).astype(np.uint8)
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    
    #fill holes
    mask_filled = np.zeros_like(mask)
    _, cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_filled, cnts, -1, 1, -1)
    mask = mask_filled
    
    #%%
    dist_t = cv2.distanceTransform(mask,cv2.DIST_L2,5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    
    mask_d = cv2.dilate(dist_t, kernel)
    local_max = cv2.compare(dist_t, mask_d, cv2.CMP_GE)
    
    non_plateau_mask = cv2.erode(dist_t, kernel)
    non_plateau_mask = cv2.compare(dist_t, non_plateau_mask, cv2.CMP_GT);
    local_max = cv2.bitwise_and(local_max, non_plateau_mask)
    local_max = cv2.dilate(local_max, kernel)
    ret, markers = cv2.connectedComponents(local_max)
    
    labels = watershed(-dist_t, markers, mask=mask)
    props = regionprops(labels)
    for p in props:
        if p.area < min_area or p.area >= max_area:
            min_row, min_col, max_row, max_col = p.bbox
            labels[min_row:max_row, min_col:max_col][p.image] = 0
    #%%
    
    return labels, th

#%%
if __name__ == '__main__':
    root_dir = Path.home() / 'workspace/datasets/BBBC026/BBBC026_v1_images'
    #bn = 'sBBBC026-hepatocytes_l1smooth_20190220_184125_unet_adam_lr0.00032_wd0.0_batch32'
    #model_path = Path.home() / 'workspace/denoising/results' / bn.split('_')[0] / bn / 'checkpoint-199.pth.tar'
    #bn = 'BBBC026-hepatocytes_unet_l1smooth_20190222_183622_adam_lr0.00032_wd0.0_batch32'
    
    
    n_epochs = 349
    
    #bn = 'BBBC026-separated_unet_l1smooth_20190224_105903_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-hepatocytes_unet_l1smooth_20190224_105953_adam_lr0.00032_wd0.0_batch32'
    
    bn = 'BBBC026-separated_unet_l1smooth_20190226_082017_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-hepatocytes_unet_l1smooth_20190226_082040_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-fibroblasts_unet_l1smooth_20190226_082007_adam_lr0.00032_wd0.0_batch32'
    
    model_path = Path.home() / 'workspace/denoising/results/BBBC026' / bn / f'checkpoint-{n_epochs}.pth.tar'
    n_ch_in, n_ch_out  = 1, 1
    if '-separated' in bn:
        n_ch_out = 3
    
    
    cuda_id = 0
    min_area = 0
    th_min = 0.
    _debug = False
    
    save_name = Path.home() / 'workspace' / f'CELLS_{n_epochs}_{bn}.csv'
    
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
    
    if _debug:
        fnames = [x for x in Path(root_dir).glob('*I01*.png') if not x.name.startswith('.')]
    else:
        fnames = [x for x in Path(root_dir).glob('*.png') if not x.name.startswith('.')]
    
    all_data = []
    for fname in tqdm.tqdm(fnames, desc = bn):
        #%%
        img = cv2.imread(str(fname), -1)
        x = img[None].astype(np.float32)
        
        pix_top = np.percentile(x, 99)
        xn = x/pix_top
        #%%
        
        with torch.no_grad():
            X = torch.from_numpy(xn[None])
            X = X.to(device)
            Xhat = model(X)
        
        xhat = Xhat[0].detach().cpu().numpy()
        
        if 'separate' in bn:
            prediction_maps = {key:xhat[ii] for ii,key in enumerate(['hep', 'bad', 'fib'])}
        elif 'fibroblast' in bn:
            prediction_maps = {'fib':xhat[0]}
        elif 'hepatocyte' in bn:
            prediction_maps = {'hep':xhat[0]}
        
        
        out_props = {}
        for k_class, pred in prediction_maps.items():
            labels, th = get_labels(pred, th_min = th_min, min_area = min_area)
            out_props[k_class] = regionprops(labels)
        
         
        
        if _debug:
            #%%
            centroids = [x.centroid for x in props_hep if x.area > min_area]
            if centroids:
                
                y, x = np.array(centroids).T
            else:
                x,y = [], []
            
            import matplotlib.pylab as plt
            fig, axs = plt.subplots(1,3,sharex=True, sharey=True, figsize = (15, 5))
            axs[0].imshow(img, vmax=int(pix_top))
            axs[0].plot(x, y, 'r.')
            #plt.imshow(labels)
            axs[1].imshow(xhat[0])
            axs[1].plot(x, y, 'r.')
            
            axs[2].imshow(labels['hep'])
            
            bn_file = fname.name.partition('_')[-1].rpartition('_')[0]
            plt.suptitle((th, bn_file))
            #%%
        bn_file, well_id, site_id, _ = fname.name.split('_')
        
        for k_class, props in out_props.items():
            cmy, cmx = zip(*[x.centroid for x in props])
            areas = [x.area for x in props]
            N = len(props)
            rows = list(zip([k_class]*N, [well_id]*N, [site_id]*N, cmx, cmy, areas))
            all_data += rows
            
    if not _debug:
        df = pd.DataFrame(all_data, columns = ['type', 'well_id', 'site_id', 'cm_x', 'cm_y', 'area'])
        df.to_csv(str(save_name), index = False)
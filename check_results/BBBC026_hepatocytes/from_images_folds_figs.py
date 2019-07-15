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
from skimage.morphology import watershed
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
import tqdm

import matplotlib.pyplot as plt
#%%
def get_labels(x2th, th_min = 0., min_area = 0., max_area = 1e20):
    
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
    
    return labels, th

def calculate_scores(segmentation_labels, target_coords):
    grouped_labels = {}
    coords_by_class = {}
    scores = {}
    for k_pred, pred_labels in  segmentation_labels.items():
        
        grouped_labels[k_pred] = {}
        coords_by_class[k_pred] = {}
        
        #get the labels center of mass
        props = regionprops(pred_labels)
        centroids_per_label = {x.label:x.centroid for x in props}
        
        #get all the labels and ignore label zero that corresponds to bgnd
        all_labs = np.unique(pred_labels)[1:]
        grouped_labels['u'] = set(all_labs)
        for k_target, coord_target in target_coords.items():
            intersected_labels = pred_labels[coord_target[...,1], coord_target[...,0]]
            intersected_labels = intersected_labels[intersected_labels>0]
            
            #remove labels from the `missing` class
            grouped_labels['u'] = grouped_labels['u'] - set(intersected_labels)
            
            #print(grouped_labels['u'])
            
            #save labels per class
            grouped_labels[k_pred][k_target] = intersected_labels
            
            #add coordinates per class
            intersected_coords = np.array([centroids_per_label[x] for x in intersected_labels])
            intersected_coords = intersected_coords if intersected_coords.size > 0 else np.zeros((0, 2), dtype = np.int) #i want to concatenate coordinates later, this makes that step easier
            coords_by_class[k_pred][k_target] = intersected_coords
        
        #add coordinates to the `missing` class
        coords_by_class[k_pred]['u'] = np.array([centroids_per_label[x] for x in grouped_labels['u']])
        
    
        TP = len(grouped_labels[k_pred][k_pred])
        FP = len(all_labs) - TP
        FN = target_coords[k_pred].shape[0] - TP
        
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        F1 = 2*P*R/(P+R)
        
        #tot_true = target_coords[k_pred].shape[0]
        #tot_pred = len(all_labs)#len(all_labs)
        
        tot_unlabelled = len(coords_by_class[k_pred]['u'])
        
        
        scores[k_pred] = {'R':R, 'P':P, 'F1':F1, 'TP':TP, 'FP':FP, 'FN':FN, 'tot_unlab' : tot_unlabelled}
    return scores, coords_by_class

def read_GT(fname):
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
    
    return img, img_g, target_coords
if __name__ == '__main__':
    bn_data = {
            'hep' : 'BBBC026-fold5/BBBC026-fold5_unet-filter_l1smooth_20190710_104126_adam_lr0.00032_wd0.0_batch32',
            'fib' : 'BBBC026-fibroblast-fold4/BBBC026-fibroblast-fold4_unet-filter_l1smooth_20190711_114025_adam_lr0.00032_wd0.0_batch32'
            }
    
    fname = Path.home() / 'workspace/datasets/BBBC/BBBC026/BBBC026_GT_images/M19_s6.png'
    img, img_g, target_coords = read_GT(fname)
    x = img_g[None].astype(np.float32)
    pix_top = np.percentile(x, 99)
    x = x/pix_top  
    
    predictions = {}
    for dat_type, bn in tqdm.tqdm(bn_data.items()):
        model_path = Path.home() / 'workspace/denoising/results/' / bn / 'checkpoint.pth.tar'
        
        n_ch_in, n_ch_out  = 1, 1
        model = UNet(n_channels = n_ch_in, n_classes = n_ch_out)
        
        
        state = torch.load(model_path, map_location = 'cpu')
        model.load_state_dict(state['state_dict'])
        model.eval()
        
        
        
        with torch.no_grad():
            X = torch.from_numpy(x[None])
            Xhat = model(X)
        
        xhat = Xhat.squeeze().detach().numpy()
        predictions[dat_type] = xhat
    #%%
    fig, axs = plt.subplots(1, 3, figsize = (20, 6), sharex=True, sharey=True)
    axs[0].imshow(img_g, cmap='gray')
    axs[0].plot(target_coords['fib'][:, 0], target_coords['fib'][:, 1], '.r')
    axs[0].plot(target_coords['hep'][:, 0], target_coords['hep'][:, 1], '.c')
    
    axs[1].imshow(predictions['hep'], cmap='gray')
    axs[2].imshow(predictions['fib'], cmap='gray')
    
    for ax in axs:
        ax.axis('off')
    #%%
            
            
#        #%%
#        labels, th = get_labels(x2th, min_area = min_area)
#        segmentation_labels = {target_label : labels}
#        scores, coords_by_class = calculate_scores(segmentation_labels, target_coords)
#        results.append((fold_id, scores))
#        #%%
#        fig, axs = plt.subplots(1, 3, figsize = (20, 6), sharex=True, sharey=True)
#        
#        axs[0].imshow(xr, cmap='gray')#, vmin=0, vmax=1)
#        axs[0].set_title('Original')
#        
#        
#        cm_good = coords_by_class[target_label][target_label]
#        cm_bad = np.concatenate((coords_by_class[target_label][noise_label], coords_by_class[target_label]['bad']))
#        
#        cm_extra = coords_by_class[target_label]['u']
#    
#        axs[2].plot(cm_bad[..., 1], cm_bad[..., 0], 'rx')
#        axs[2].plot(cm_good[..., 1], cm_good[..., 0], 'b.')
#        axs[2].plot(cm_extra[..., 1], cm_extra[..., 0], 'gx')
#        if xhat.ndim == 3:
#            x_rgb = np.rollaxis(xhat, 0, 3)
#        else:
#            x_rgb = xhat.squeeze()
#            
#        axs[1].imshow(x2th, cmap='gray')#, vmin=0, vmax=1)
#        axs[1].set_title('Prediction')
#        
#        axs[2].imshow(labels)
#        axs[2].set_title(f'Segmentation')
#        
#        plt.suptitle(bn)
#        #%%
#    results = sorted(results, key = lambda x : x[0])
#    cols = ['R', 'P', 'F1', 'TP' , 'FP', 'FN', 'tot_unlab']
#    dat = [[x[target_label][k] for k in cols] for _, x in results]
#    R, P, F1, TP, FP, FN, tot_unlab = zip(*dat)
#    
#    R_avg = np.mean(R)
#    P_avg = np.mean(P)
#    F1_avg = np.mean(F1)
#    
#    TP = sum(TP)
#    FN = sum(FN)
#    FP = sum(FP)
#    tot_unlab = sum(tot_unlab)
#    
#    P_pooled = TP/(TP+FP)
#    R_pooled = TP/(TP+FN)
#    F1_pooled = 2*R_pooled*P_pooled/(P_pooled + R_pooled)
#    #%%
#    print(f'Average : P={P_avg:.3}, R={R_avg:.3}, F1={F1_avg:.3}')
#    print(f'Pooled : P={P_pooled:.3}, R={R_pooled:.3}, F1={F1_pooled:.3}')
#    
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


from semantic_filtering.models import UNet
import torch
import numpy as np
import cv2
import pandas as pd
import tqdm
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#%%

def get_boxes(x2th, th = -1, min_area = 100):
    if th < 0:
        th = threshold_otsu(x2th)
    
    mask = (x2th>th).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    _, cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pred_bboxes = [cv2.boundingRect(x) for x in cnts if cv2.contourArea(x) > min_area]
    pred_bboxes = [(x, y, x + w, y + h) for (x,y,w,h) in pred_bboxes]
    
    return pred_bboxes, mask
    #%%
    
#def _get_candiates(x2th, th = -1):
#    if th < 0:
#        th = threshold_otsu(x2th)
#    
#    mask = (x2th>th).astype(np.uint8)
#        
#    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#    mask = cv2.dilate(mask, kernel, iterations=2)
#    
#    _, cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#    pred_bboxes = [cv2.boundingRect(x) for x in cnts]
#    return pred_bboxes, mask
#
#def _filter_boxes(pred_bboxes, min_bbox_size):
#    preds = []
#    for bb in pred_bboxes:
#        b_size = max(bb[2], bb[3])
#        if b_size > min_bbox_size:
#            dat = (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3])
#            preds.append(dat)
#    preds = np.array(preds)
#    return preds
#%%
if __name__ == '__main__':

    #bn = 'BBBC042-colour-v4-S5_unet-filter_l1smooth_20190709_202139_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC042-colour-v4_unet-filter_l1smooth_20190702_141954_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC042-colour-bgnd-S5_unet-filter_l1smooth_20190710_124947_adam_lr0.00032_wd0.0_batch32'
    
    #bn = 'BBBC042-colour-bgnd-S5_unet-filter_l1smooth_20190710_145222_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC042-colour-bgnd-S10_unet-filter_l1smooth_20190710_164420_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC042-colour-bgnd-S25_unet-filter_l1smooth_20190710_162339_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC042-colour-bgnd-S100_unet-filter_l1smooth_20190710_151403_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC042-colour-bgnd_unet-filter_l1smooth_20190710_150856_adam_lr0.00032_wd0.0_batch32'
    
    #bn = 'BBBC042-simple-S10_unet-filter_l1smooth_20190711_204742_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC042-simple_unet-filter_l1smooth_20190711_195905_adam_lr0.00032_wd0.0_batch32'
    
    #bn = 'BBBC042-colour-bgnd_unet-filter_l1smooth_20190712_110713_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC042-simple-bgnd_unet-filter_l1smooth_20190712_104448_adam_lr0.00032_wd0.0_batch32'
    
    bn = 'BBBC042-simple-bgnd_unet-filter_l1smooth_20190713_124210_adam-stepLR-30-0.1_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC042-simple-bgnd_unet-filter_l1smooth_20190713_162625_adam-_lr0.00032_wd0.0_batch32'
    
    subdir = bn.partition('_')[0]
    #model_path = Path.home() / 'workspace/denoising/results' / subdir / bn / f'model_best.pth.tar'
    model_path = Path.home() / 'workspace/denoising/results' / subdir / bn / f'checkpoint.pth.tar'
    #model_path = Path.home() / 'workspace/denoising/results' / subdir / bn / f'checkpoint-19.pth.tar'
    #model_path = Path.home() / 'workspace/denoising/results' / subdir / bn / f'checkpoint-24.pth.tar'
    #model_path = Path.home() / 'workspace/denoising/results' / subdir / bn / f'checkpoint-49.pth.tar'
    
    
    int_scale = (0,255)
    
    if 'colour' in bn:
        n_ch_in, n_ch_out  = 3, 3
    else:
        n_ch_in, n_ch_out  = 1, 1
    
    if '-decomposition' in bn:
        n_ch_out *= 3
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out)
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/50.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/5.tif'
    
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1075.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1020.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1100.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1010.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1004.tif'
    
    
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1021.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1003.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1074.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1041.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1020.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1016.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1011.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1006.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1003.tif'
    
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1051.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1049.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1039.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1037.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1022.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1017.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1007.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/1006.tif'
    
    
    
    root_dir = Path('/Users/avelinojaver/Downloads/BBBC042/')
    
    #files2check = [1016, 1021, 1049, 1026, 1003, 1074, 1041, 1016]
    #files2check = [1026, 1003, 1003,1004,1006,1007,1010,1011,1016,1017,1020,1021,1022,1037,1039,1041,1049,1051,1074,1075,1100]
    files2check = [1016]#range(1051, 1100)
    
    for ifname in tqdm.tqdm(files2check):
        fname = root_dir / 'images' / f'{ifname}.tif'
        annotations_file = root_dir / 'positions' / f'{ifname}.txt'
        df = pd.read_csv(str(annotations_file), delim_whitespace=True, header = None)
        
        img_ori = cv2.imread(str(fname), -1)[..., ::-1] #opencv reads the channels as BGR so i need to switch them
        
        
#        if n_ch_in == 1:
#            img = cv2.cvtColor(img_ori, cv2.COLOR_RGB2GRAY);
#            img = 255 - img
#            img = img[None]
#        else:
        img = np.rollaxis(img_ori, 2, 0)
        
        x = img.astype(np.float32)
        
        x = (x - int_scale[0])/(int_scale[1] - int_scale[0])
        
        with torch.no_grad():
            X = torch.from_numpy(x)
            
            if n_ch_in == 1:
                X = 0.299*X[0] + 0.587*X[1] + 0.114*X[2]
                X = 1 - X.unsqueeze(0)
            
            
            Xhat = model(X[None])
        #%%
        xhat = Xhat[0].detach().numpy()
        xhat = np.rollaxis(xhat, 0, 3)
        
        xr = x.squeeze()
        
        #%%
        if n_ch_in == 1:
            x2th = x2plot = xhat[..., 0]
            
        else:
            x2plot = xhat[..., :3]
            x2th = 1 - cv2.cvtColor(x2plot, cv2.COLOR_RGB2GRAY)
            
        #%%
#        pred_bboxes, mask = _get_candiates(x2th)
#        pred_bboxes = _filter_boxes(pred_bboxes, min_bbox_size= 30)
        pred_bboxes, mask = get_boxes(x2th)
        
        
        n_plots = (1,3) #if xhat.shape[0] > 3 else (1,3)
        fig, axs = plt.subplots(*n_plots,sharex=True, sharey=True, figsize=(20, 8))
        axs = axs.flatten()
        
        axs[0].imshow(img_ori, cmap='gray')#, vmin=0, vmax=1)
        
        for _, row in df.iterrows():
            x1, y1, x2, y2 = row[4:8]
            cc = x1, y1
            ll = x2 - x1
            ww = y2 - y1
            rect = patches.Rectangle(cc, ll, ww, linewidth=2, edgecolor='r', facecolor='none', linestyle = '-')
            axs[0].add_patch(rect)
        
        for x1, y1, x2, y2 in pred_bboxes:
            
            cc = x1, y1
            ll = x2 - x1
            ww = y2 - y1
            rect = patches.Rectangle(cc, ll, ww, linewidth=2, edgecolor='g', facecolor='none', linestyle = '-')
            axs[0].add_patch(rect)
        
        th = x2th < .25#0.175#threshold_otsu(x2th)*0.8
        
        if n_ch_in == 1:
            axs[1].imshow(x2plot, cmap='gray')
        else:
            axs[1].imshow(x2plot)
        
        
        
        
        axs[2].imshow(mask)
        
        
            
        for ax in axs.flatten():
            ax.axis('off')
    #%%
    ','.join(sorted(set([(x.rpartition('/')[-1][:-5]) for x in aa.split('\n') if x])))
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
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tqdm
#%%
if __name__ == '__main__':
    cuda_id = 0
    bn = 'BBBC042-v3-separated_unet_l1smooth_20190302_224236_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC042-small-separated_unet_l1smooth_20190302_221604_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC042_unet_l1smooth_20190226_003119_adam_lr0.00032_wd0.0_batch32'
    
    #bn = 'R_BBBC042-small_unet_l1smooth_20190228_191521_adam_lr0.00032_wd0.0_batch32'
    #bn = 'R_BBBC042-v3_unet_l1smooth_20190228_191527_adam_lr0.00032_wd0.0_batch32'
    
    #root_dir = '/Users/avelinojaver/Downloads/BBBC042/images/'
    root_dir = Path.home() / 'workspace/datasets/BBBC042/images'
    
    
    _debug = False
    
    n_epochs = 99#299#
    
    model_path = Path.home() / 'workspace/denoising/results/BBBC042' / bn / f'checkpoint-{n_epochs}.pth.tar'
    save_name = Path.home() / 'workspace' / f'BBOXES_{n_epochs}_{bn}.csv'
    
    int_scale = (0,255)
    min_area = 100
    
    n_ch_in  = 1
    if '-separated' in bn:
        n_ch_out = 3
        is_otsu = False
        
    else:
        n_ch_out = 1
        is_otsu = True
    
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
    
    root_dir = Path(root_dir)
    
    fnames = root_dir.glob('*.tif')
    fnames = [x for x in fnames if int(x.stem)>1000 and not x.name.startswith('.')]
    fnames = sorted(fnames, key = lambda x : int(x.stem))
    
    
    if _debug:
        fnames = fnames[:5]
        
    all_data = []
    for fname in tqdm.tqdm(fnames):
        img_ori = cv2.imread(str(fname), -1)[..., ::-1]
        img = cv2.cvtColor(img_ori, cv2.COLOR_RGB2GRAY);
        img = 255 - img
        
        
        x = img[None].astype(np.float32)
        x = (x - int_scale[0])/(int_scale[1] - int_scale[0])
        
        with torch.no_grad():
            X = torch.from_numpy(x[None])
            X = X.to(device)
            Xhat = model(X)
        
        xhat = Xhat.squeeze().detach().cpu().numpy()
        
        
        if xhat.ndim == 2:
            x2th = xhat
        else:
            x2th = xhat[0] 
        #%%
        if is_otsu:
            th = threshold_otsu(x2th)
        else:
            th = 0.035
        #
        
        x_th = (x2th>th).astype(np.uint8)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        mask = cv2.morphologyEx(x_th, cv2.MORPH_CLOSE, kernel, iterations=1)
        _, cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pred_bboxes = [cv2.boundingRect(x) for x in cnts]
        areas = [cv2.contourArea(x) for x in cnts]
        
        
        if not _debug:
            img_id = int(fname.stem)
            rows = [(img_id, area, *bb) for (area, bb) in zip(areas, pred_bboxes) if area > min_area]
            all_data += rows
        else:
            
            annotations_file = str(fname).replace('/images/', '/positions/').replace('.tif', '.txt')
            df = pd.read_csv(str(annotations_file), delim_whitespace=True, header = None)
            
            fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20, 8))
            
            axs[0].imshow(img_ori, cmap='gray')#, vmin=0, vmax=1)
            axs[0].set_title('Original')
            
            for _, row in df.iterrows():
                x1, y1, x2, y2 = row[4:8]
                cc = x1, y1
                ll = x2 - x1
                ww = y2 - y1
                rect = patches.Rectangle(cc, ll, ww, linewidth=2, edgecolor='r', facecolor='none', linestyle = '-')
                axs[0].add_patch(rect)
            
            
            
            for (x,y,w,h), area in zip(pred_bboxes, areas):
                if area > min_area:
                    rect = patches.Rectangle((x,y), w, h, linewidth=2, edgecolor='g', facecolor='none', linestyle = '-')
                    axs[0].add_patch(rect)
            
            axs[1].imshow(x2th, cmap='gray')#, vmin=0, vmax=1)
            axs[1].set_title('Prediction')
            
            
            axs[2].imshow(mask)
            axs[2].set_title(f'Prediction > {th}')
            for ax in axs.flatten():
                ax.axis('off')
        
            plt.suptitle((bn,model_path.name))
            #%%
    if not _debug:
        df = pd.DataFrame(all_data, columns = ['img_id', 'area', 'x', 'y', 'w', 'h'])
        df.to_csv(str(save_name), index = False)
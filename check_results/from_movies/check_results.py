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
import pandas as pd
#%%
if __name__ == '__main__':
    n_ch  = 3
    #bn = 'from-movies_l1smooth_20190207_223830_unet-ch3_adam_lr0.00032_wd0.0_batch32'
    #bn = 'from-movies-gap-1_l1smooth_20190208_113709_unet-ch3_adam_lr8e-05_wd0.0_batch8'
    #bn = 'from-movies-gap-4_l1smooth_20190208_113712_unet-ch3_adam_lr8e-05_wd0.0_batch8'
    #bn = 'from-movies-gap-16_l1smooth_20190208_113715_unet-ch3_adam_lr8e-05_wd0.0_batch8'
    #bn = 'from-movies-gap-64_l1smooth_20190208_113705_unet-ch3_adam_lr8e-05_wd0.0_batch8'
    #bn = 'from-movies-gap-256_l1smooth_20190208_113658_unet-ch3_adam_lr8e-05_wd0.0_batch8'
    
    #bn = 'toulouse-gap-16_l1smooth_20190219_225713_unet-ch3_adam_lr8e-05_wd0.0_batch8'
    bn = 'toulouse-gap-64_l1smooth_20190219_225719_unet-ch3_adam_lr8e-05_wd0.0_batch8'
    
    model_path = Path.home() / 'workspace/denoising/results' / bn.split('_')[0] / bn / 'checkpoint.pth.tar'
    
    #gen = SyntheticFluoFlow()
    model = UNet(n_channels = n_ch, n_classes = n_ch)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/survellance_videos/stanford_campus_dataset/video.mov'
    #fname = 'Volumes/rescomp1/data/datasets/stanford_campus_dataset/videos/nexus/video10/video.mov'
    #fname = '/Volumes/rescomp1/data/datasets/stanford_campus_dataset/videos/nexus/video11/video.mov'
    
    #fname = '/Volumes/rescomp1/data/datasets/stanford_campus_dataset/videos/deathCircle/video0/video.mov'
    #fname = '/Volumes/rescomp1/data/datasets/stanford_campus_dataset/videos/bookstore/video0/video.mov'
    #fname = '/Volumes/rescomp1/data/datasets/stanford_campus_dataset/videos/hyang/video0/video.mov'
    #fname = '/Volumes/rescomp1/data/datasets/stanford_campus_dataset/videos/gates/video0/video.mov'
    #fname = '/Volumes/rescomp1/data/datasets/stanford_campus_dataset/videos/coupa/video0/video.mov'
    #fname = '/Volumes/rescomp1/data/datasets/stanford_campus_dataset/videos/little/video0/video.mov'
    #fname = '/Volumes/rescomp1/data/datasets/stanford_campus_dataset/videos/nexus/video0/video.mov'
    #fname = '/Volumes/rescomp1/data/datasets/stanford_campus_dataset/videos/quad/video0/video.mov'
    #fname = '/Volumes/rescomp1/data/datasets/stanford_campus_dataset/videos/quad/video1/video.mov'
    #fname = '/Volumes/rescomp1/data/datasets/stanford_campus_dataset/videos/quad/video2/video.mov'
    #fname = '/Volumes/rescomp1/data/datasets/stanford_campus_dataset/videos/quad/video3/video.mov'
    #%%
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/survellance_videos/ToulouseCampusSurveillanceDataset/train/VideoScenario1LowResolution/F1C2LR.mp4'
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/survellance_videos/ToulouseCampusSurveillanceDataset/train/VideoScenario1LowResolution/F1C11LR.mp4'
    fname = '/Users/avelinojaver/OneDrive - Nexus365/survellance_videos/ToulouseCampusSurveillanceDataset/train/VideoScenario1LowResolution/F1C19LR.mp4'
    #%%
    ini_frame = 100
    annotations_file = Path(fname.replace('/videos/', '/annotations/')).parent / 'annotations.txt'
    
    if annotations_file.exists():
        
        columns = ['track_id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'oclutted', 'interpolated', 'type']
        df = pd.read_csv(str(annotations_file), delim_whitespace=True, header = None)
        df.columns = columns
        df = df[(df['lost'] == 0) & (df['oclutted'] ==0)]
        
        
        df_g = df.groupby('frame')
        
        
        
        df_frame = df_g.get_group(ini_frame)
    else:
        df_frame = None
        
    vcap = cv2.VideoCapture(str(fname))

    
    vcap.set(cv2.CAP_PROP_POS_FRAMES, ini_frame)
    ret, img = vcap.read()
    
    img = np.rollaxis(img, 2, 0)
    
    x = (img/255.).astype(np.float32)
    
    
    with torch.no_grad():
        X = torch.from_numpy(x[None])
        Xhat = model(X)
    
    xhat = Xhat.detach().numpy().squeeze()
    xhat = np.rollaxis(xhat, 0, 3)
    
    xr = np.rollaxis(x, 0, 3)
    
    #%%
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, axs = plt.subplots(2,2,sharex=True, sharey=True)
    axs = axs.flatten()
    
    axs[0].imshow(xr, cmap='gray')#, vmin=0, vmax=1)
    axs[0].set_title('Input')
    
    
    axs[1].imshow(xhat, cmap='gray')#, vmin=0, vmax=1)
    axs[1].set_title('Prediction')
    
    dd = np.abs(xhat-xr).sum(axis=-1)
    axs[2].imshow(dd)
    axs[2].set_title('|Prediction - Input|')
    
    
    th = 0.2
    mask = (dd > th).astype(np.uint8)
    mask = cv2.erode(mask, np.ones((3,3)), iterations = 1)
    mask = cv2.dilate(mask, np.ones((5,5)), iterations = 2)
    retval, labels, stats, centroids =  cv2.connectedComponentsWithStats(mask)
    
    axs[3].imshow(xr)
    axs[3].set_title('Bboxes')
    
    
    if df_frame:
        for _, row in df_frame.iterrows():
            cc = (row['xmin'], row['ymin'])
            ll = row['xmax'] - row['xmin']
            ww = row['ymax'] - row['ymin']
            rect = patches.Rectangle(cc, ll, ww, linewidth=2, edgecolor='r', facecolor='none', linestyle = ':')
            axs[3].add_patch(rect)
        for (cx, cy, w, h, _) in stats[1:]:
            rect = patches.Rectangle((cx, cy), w, h, linewidth=2, edgecolor='c', facecolor='none', linestyle = '-')
            axs[3].add_patch(rect)
        
        plt.suptitle(bn.split('_')[0])
        for ax in axs.flatten():
            ax.axis('off')
    #%%
    if False:
        vcap = cv2.VideoCapture(str(fname))
        imgs = []
        
        imgs2avg = 16
        for _ in range(imgs2avg):
            ret, img = vcap.read()
            imgs.append(img/255.)
            
        imgs = np.stack(imgs)
        
        bgnd = np.median(imgs, axis=0)
        
        
        img = imgs[0]
        dd = np.abs(img-bgnd).sum(axis=-1)
        mask = (dd > th).astype(np.uint8)
        mask = cv2.erode(mask, np.ones((3,3)), iterations = 1)
        mask = cv2.dilate(mask, np.ones((5,5)), iterations = 4)
        
        fig, axs = plt.subplots(2,2,sharex=True, sharey=True)
        axs = axs.flatten()
        
        axs[0].imshow(img, cmap='gray')#, vmin=0, vmax=1)
        axs[1].imshow(bgnd, cmap='gray')#, vmin=0, vmax=1)
        axs[2].imshow(dd)
        axs[3].imshow(mask)
        plt.suptitle(f'Bgnd = median {imgs2avg} images')
    
    
    
    
    
    
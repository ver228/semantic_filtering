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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tqdm
#%%
if __name__ == '__main__':
    bn = 'BBBC042-v3-separated_unet_l1smooth_20190302_224236_adam_lr0.00032_wd0.0_batch32'
    
    n_epochs = 299
    model_path = Path.home() / 'workspace/denoising/results/BBBC042' / bn / f'checkpoint-{n_epochs}.pth.tar'
    
    int_scale = (0,255)
    
    n_ch_in, n_ch_out  = 1, 1
    if '-separated' in bn:
        n_ch_out = 3
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out)
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/50.tif'
    #fname = '/Users/avelinojaver/Downloads/BBBC042/images/5.tif'
     
    root_dir = Path('/Users/avelinojaver/Downloads/BBBC042/')
    
    save_dir = '/Users/avelinojaver/OneDrive - Nexus365/papers/miccai2019/data/astrocytes/'
    save_dir = Path(save_dir)
    
    data = dict(
        good1 = (1026, (250, 500), (450, 700)),
        good2 = (1003, (410, 760), (0, 350)),
        good3 = (1074, (210, 560), (320, 670)),
        good4 = (1022, (280, 630), (190, 540)),
        good5 = (1016, (25, 375), (75, 425)),
        good6 = (1033, (50, 400), (150, 500)),
        bad1 = (1049, (700, 950), (450, 700)),
        bad2 = (1075, (400, 650), (330, 580)),
        bad3 = (1075, (150, 400), (150, 400)),
        bad4 = (1004, (50, 300), (220, 470)),
        bad5 = (1011, (550, 800), (150, 400)),
        bad6 = (1049, (100, 350), (350, 600)),
        bad7 = (1037, (250, 500), (200, 450))
        )
    
    #data = {'d' : (1033, None, None)}#(25, 375), (75, 425))}
    
    #maybe -> 1016,  
    #img_n = 1037
    #xl, yl = (250, 500), (200, 450)#None, None, ##, None#
    
    for k_name, (img_n, xl, yl) in tqdm.tqdm(data.items()):
        fname = root_dir / 'images' / f'{img_n}.tif'
        annotations_file = root_dir / 'positions' / f'{img_n}.txt'
        
        
        df = pd.read_csv(str(annotations_file), delim_whitespace=True, header = None)
        
        img_ori = cv2.imread(str(fname), -1)[..., ::-1]
        img = cv2.cvtColor(img_ori, cv2.COLOR_RGB2GRAY);
        img = 255 - img
        
        x = img[None].astype(np.float32)
        x = (x - int_scale[0])/(int_scale[1] - int_scale[0])
        
        with torch.no_grad():
            X = torch.from_numpy(x[None])
            Xhat = model(X)
        
        xhat = Xhat[0].detach().numpy()
        #%%
        fig, axs = plt.subplots(1, 4,sharex=True, sharey=True, figsize=(20, 8))
        
        axs[0].imshow(img_ori, cmap='gray')#, vmin=0, vmax=1)
        
        for _, row in df.iterrows():
            x1, y1, x2, y2 = row[4:8]
            cc = x1, y1
            ll = x2 - x1
            ww = y2 - y1
            rect = patches.Rectangle(cc, 
                                     ll, 
                                     ww, 
                                     linewidth=2, 
                                     edgecolor='r', 
                                     facecolor='none', 
                                     linestyle = ':')
            axs[0].add_patch(rect)
        
            
        axs[1].imshow(xhat[0] , cmap='gray')#, vmin=0, vmax=1)
        axs[2].imshow(xhat[2] , cmap='gray')#, vmin=0, vmax=1)
        axs[3].imshow(xhat[1] , cmap='gray')#, vmin=0, vmax=1)
        
        
        
        if xl is not None:
            axs[0].set_xlim(xl)
        if yl is not None:
            axs[0].set_ylim(yl)
        
        
        for ax in axs.flatten():
            ax.axis('off')
        #%%
        fig.savefig(str(save_dir / (k_name + '.pdf')))
        
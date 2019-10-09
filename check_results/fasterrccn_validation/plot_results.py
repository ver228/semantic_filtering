#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 13:24:49 2019

@author: avelinojaver
"""
from flow import BBBC042Dataset, collate_simple, get_transform
from train_fasterrcnn import get_model, get_scores

import tqdm
import numpy as np
import torch
from pathlib import Path
import time
from torch.utils.data import DataLoader

import matplotlib.pylab as plt
import matplotlib.patches as patches
    
#%%
if __name__ == '__main__':
    #bn = 'BBBC042-roi512_retinanet-resnet50_20190706_131158_adam-_lr1e-05_wd0.0_batch12'
    #bn = 'BBBC042-roi512_fasterrcnn_20190705_133510_sgd-stepLR-20-0.1_lr0.005_wd0.0005_batch16'
    #bn = 'BBBC042_fasterrcnn_20190704_214026_adam_lr0.0001_wd0.0_batch16'
    #bn = 'BBBC042-roi512_fasterrcnn-resnet50_20190706_175220_adam-_lr0.0001_wd0.0_batch16'
    #bn = 'V_BBBC042-max5-roi512_fasterrcnn-resnet50_20190708_172706_sgd-stepLR-250-0.1_lr0.005_wd0.0005_batch16'
    
    bn = 'V_BBBC042-max5-roi512_fasterrcnn-resnet50_20190709_164921_sgd-stepLR-250-0.1_lr0.005_wd0.0005_batch16'
    
    batch_size = 4
    device = cpu_device = torch.device("cpu")
    
    dd = bn[2:] if bn.startswith('V_') else bn
    model_name, _, backbone_name = dd.split('_')[1].partition('-')
    backbone_name = backbone_name if backbone_name else 'resnet50'
    
    model_path = Path.home() / 'workspace/localization/results/bbox_detection' / model_name / bn / 'checkpoint.pth.tar'
    state = torch.load(model_path, map_location = 'cpu')
    #%%
    
    
    img_size = (708, 990)
    model = get_model(model_name, backbone_name, img_size, pretrained_backbone = False)
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    
    data_dir = Path('/Users/avelinojaver/Downloads/BBBC042/')
    
    #transforms = get_transform(roi_size = img_size)
    flow_test = BBBC042Dataset(data_dir, set_type = 'test', max_samples = 12)
    
    
    data_loader = DataLoader(flow_test, 
                                  batch_size = batch_size,
                                  num_workers = 1,
                                  collate_fn = collate_simple
                                  )
    
    
    metrics = np.zeros(3)
    
    model_time_avg = 0
    pbar = tqdm.tqdm(data_loader)
    for images, targets in pbar:
        images = list(img.to(device) for img in images)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        
        
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time_avg += time.time() - model_time
        
    
        for pred, true in zip(outputs, targets) :
            pred_bb = pred['boxes'].detach().cpu().numpy()
            true_bb = true['boxes'].detach().cpu().numpy()
            TP, FP, FN, pred_ind, true_ind = get_scores(pred_bb, true_bb)
            metrics += TP, FP, FN
        
        for img, preds, target in zip(images, outputs, targets):
            img = img.detach().numpy()
            img = (img + 1)/2
            img = np.rollaxis(img, 0, 3)
            #print(img.shape)
            
            fig, ax = plt.subplots(1, 1)
            plt.imshow(img)
            for bb in target['boxes']:
                cc = (bb[0], bb[1])
                w = bb[2] - bb[0]
                h = bb[3] - bb[1]
                rect = patches.Rectangle(cc, w, h, linewidth=2, edgecolor='r', facecolor='none', linestyle = '-')
                ax.add_patch(rect)
            
            plt.imshow(img)
            for bb in preds['boxes']:
                cc = (bb[0], bb[1])
                w = bb[2] - bb[0]
                h = bb[3] - bb[1]
                rect = patches.Rectangle(cc, w, h, linewidth=2, edgecolor='c', facecolor='none', linestyle = '-')
                ax.add_patch(rect)
                
            
        
    TP, FP, FN = metrics
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = 2*P*R/(P+R)
    
    model_time_avg /= len(data_loader) 
    print(bn)
    print(f'R={R:.3} P={P:.3} F1={F1:.3}')
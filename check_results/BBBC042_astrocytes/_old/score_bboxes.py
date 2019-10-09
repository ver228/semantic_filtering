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
from scipy.optimize import linear_sum_assignment

if __name__ == '__main__':
    _debug = True
    min_bb_area = 1000
    #root_dir = Path.home() / 'workspace'
    #annotations_dir = Path.home() / 'workspace/datasets/BBBC042/positions'
    
    src_dir = Path.home() / 'OneDrive - Nexus365/papers/miccai2019/data/astrocytes/'
    annotations_dir = '/Users/avelinojaver/Downloads/BBBC042/positions/'
    
    
    annotations_dir = Path(annotations_dir)
    
    if not _debug:
        fnames =  src_dir.glob('BBOXES_*.csv')
    else:
        fnames =  src_dir.glob('BBOXES_399_BBBC042-v3*.csv')
    
    #%%
    def _sort_k(x):
         n_epochs, _, bn = x.name.partition('_')[-1].partition('_')
         return (bn, int(n_epochs))
    
    
    
    results = []
    for fname in sorted(fnames, key = _sort_k):
        df = pd.read_csv(str(fname))
        
        TP, FP, FN = 0, 0, 0
        for img_id, img_data in df.groupby('img_id'):
            #%%
            pred_bb_areas = (img_data['w']*img_data['h']).values
            good = pred_bb_areas > min_bb_area
            img_data = img_data[good]
            pred_bb_areas = pred_bb_areas[good]
            
            annotations_file = annotations_dir / f'{img_id}.txt'
            true_df = pd.read_csv(str(annotations_file), delim_whitespace=True, header = None)
            
            x1, y1, x2, y2 =  map(np.array, zip(true_df.loc[:, 4:7].values.T))
            true_bb_areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            
            
            
            pred_x1 = img_data['x'].values
            pred_x2 = pred_x1 + img_data['w'].values
            
            pred_y1 = img_data['y'].values
            pred_y2 = pred_y1 + img_data['h'].values
            
            xx1 = np.maximum(pred_x1[..., None], x1)
            yy1 = np.maximum(pred_y1[..., None], y1)
            xx2 = np.minimum(pred_x2[..., None], x2)
            yy2 = np.minimum(pred_y2[..., None], y2)
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            
            inter = w * h
            union = pred_bb_areas[..., None] + true_bb_areas - inter
            IoU = inter/union
            
            #frac_covered = 
            
            
            cost_matrix = inter.copy()
            cost_matrix[cost_matrix==0] = 1e-3
            cost_matrix = 1/cost_matrix
            pred_ind, true_ind = linear_sum_assignment(cost_matrix)
            
            good = inter[pred_ind, true_ind] > 0
            pred_ind, true_ind = pred_ind[good], true_ind[good]
            
            
            TP += pred_ind.size
            FP += inter.shape[0] - pred_ind.size
            FN += inter.shape[1] - pred_ind.size
            
            if _debug:
                img_file = annotations_dir.parent / f'images/{img_id}.tif'
                
                img_ori = cv2.imread(str(img_file), -1)[..., ::-1]
            
                fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(20, 20))
                axs.imshow(img_ori)
                
                
                ww = (x2 - x1).T
                hh = (y2 - y1).T
                
                was_found = np.zeros(x1.size)
                was_found[true_ind] = 1
                for (ff, x,y,w,h) in zip(was_found, x1.T, y1.T, ww, hh):
                    col = 'c' if ff > 0 else 'm'
                    
                    rect = patches.Rectangle((x,y), w, h, linewidth=2, edgecolor=col, facecolor='none', linestyle = '--')
                    axs.add_patch(rect)
                
                was_found = np.zeros(pred_x1.size)
                was_found[pred_ind] = 1
                for i_pred, ff in enumerate(was_found):
                    col = 'b' if ff > 0 else 'r'
                    
                    x, y = pred_x1[i_pred], pred_y1[i_pred]
                    w = pred_x2[i_pred] - x
                    h = pred_y2[i_pred] - y
                    rect = patches.Rectangle((x,y), w, h, linewidth=2, edgecolor=col, facecolor='none', linestyle = '-')
                    axs.add_patch(rect)
                
                
                fig.suptitle(img_id)
                
                if img_id > 1050:
                    break
        
        #%%
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        F1 = 2*P*R/(P+R)
        
        output = [fname.name, 
                  f'precision -> {P}',
                  f'recall -> {R}',
                  f'f1 -> {F1}'
              ]
        results.append('\n'.join(output))
    
    if not _debug:
        str2save = '\n----\n'.join(results)
        save_name = src_dir / 'results.txt'
        with open(save_name, 'w') as fid:
            fid.write(str2save)
    #%%
    

#            
#            fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20, 8))
#            
#            axs[0].imshow(img_ori, cmap='gray')#, vmin=0, vmax=1)
#            axs[0].set_title('Original')
#            
#            for _, row in df.iterrows():
#                x1, y1, x2, y2 = row[4:8]
#                cc = x1, y1
#                ll = x2 - x1
#                ww = y2 - y1
#                rect = patches.Rectangle(cc, ll, ww, linewidth=2, edgecolor='r', facecolor='none', linestyle = '-')
#                axs[0].add_patch(rect)
#            
#            
#            min_area = 250
#            for (x,y,w,h), area in zip(pred_bboxes, areas):
#                if area > min_area:
#                    rect = patches.Rectangle((x,y), w, h, linewidth=2, edgecolor='g', facecolor='none', linestyle = '-')
#                    axs[0].add_patch(rect)
#            
#            axs[1].imshow(x2th, cmap='gray')#, vmin=0, vmax=1)
#            axs[1].set_title('Prediction')
#            
#            
#            axs[2].imshow(x_th)
#            axs[2].set_title(f'Prediction > {th}')
#            for ax in axs.flatten():
#                ax.axis('off')
#        
#            plt.suptitle((bn,model_path.name))
#    
#    if not _debug:
#        df = pd.DataFrame(all_data, columns = ['img_id', 'area', 'x', 'y', 'w', 'h'])
#        df.to_csv(str(save_name), index = False)
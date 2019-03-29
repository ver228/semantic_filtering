#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:46:02 2019

@author: avelinojaver
"""
from pathlib import Path
import numpy as np
import cv2
import tables
from get_accuracy import get_labels, calculate_performance, read_GT
import pandas as pd
import tqdm
import matplotlib.pylab as plt
import seaborn as sns
from scipy.stats import ttest_ind
from skimage.measure import regionprops
import math
import warnings

#def remove_labels():
#    root_dir = '/Users/avelinojaver/Downloads/BBBC026_GT_images/'
#    root_dir = Path(root_dir)
#    
#    save_dir = Path('/Users/avelinojaver/Downloads/BBBC026_GT_nolabel/')
#    save_dir.mkdir(exist_ok=True, parents=True)
#    
#    for fname in root_dir.rglob('*.png'):
#        
#        img, img_g, target_coords = read_GT(fname)
#        cv2.imwrite(str(save_dir / fname.name), img_g)
        
def _filter_by_size(lab, min_size, max_size):
    np.warnings.filterwarnings('ignore')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        props = regionprops(lab)
        lab_filt = lab.copy()
        for p in props:
            #dd = math.sqrt(p.area/np.pi)*2 #calculate an estimated diameter from the area (following cell profiler code)
            
            #dd = (p.minor_axis_length + p.major_axis_length)/2
            #if dd < min_size or dd > max_size:
            
            if p.minor_axis_length < min_size or p.major_axis_length > max_size:
                lab_filt[p.coords[:,0], p.coords[:,1]] = 0
        return lab_filt


if __name__ == '__main__':
    
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/papers/miccai2019/data/hepatocytes/ilastik_results/manual_scaled/'
    root_dir_GT = '/Users/avelinojaver/OneDrive - Nexus365/papers/miccai2019/data/hepatocytes/BBBC026/BBBC026_GT_images'
    
    root_dir = Path(root_dir)
    root_dir_GT = Path(root_dir_GT)
    
    diameter_limits = {'fib' : (25, 120), 'hep' : (20, 60)}
    #diameter_limits = {'fib' : (20, 120), 'hep' : (20, 120)}
    
    #%%
    fnames = list(root_dir.glob('*.h5'))
    for fname in fnames:
    
        #fname_GT = root_dir_GT / (fname.stem.rpartition('_')[0])
        fname_GT = root_dir_GT / (fname.stem.rpartition('_')[0] + '.png')
        img, img_g, target_coords = read_GT(fname_GT)
        
        with tables.File(str(fname), 'r') as fid:
            preds = fid.get_node('/exported_data')[:]
        
        preds = preds.astype(np.float32)
        pred_maps = {'hep' : preds[..., 0], 'fib' : preds[..., 1]}
        
        
        kernel = (7,7)
        segmentation_labels = {}
        
        
        for k, mm in pred_maps.items():
            mm = cv2.blur(mm, kernel)
            
            #min_area, max_area = [(x/2)**2*np.pi for x in diameter_limits[k]]
            min_area, max_area = 300, 5000
            lab, _ = get_labels(mm, th_min = 0.5, min_area = min_area, max_area = max_area)
            
            #lab, _ = get_labels(mm, th_min = 0.5)
            #lab = _filter_by_size(lab, *diameter_limits[k])
            
            segmentation_labels[k] = lab
            
        results, coords_by_class = calculate_performance(segmentation_labels, target_coords)
        
        for k_target in ['hep', 'fib']:
        
            print(fname.name, k_target, results[k_target])
        
        
        fig, axs = plt.subplots(1,3,sharex=True, sharey=True, figsize=(15, 8))
        axs[0].imshow(img)
        axs[1].imshow(preds[..., 0], cmap='gray')
        axs[2].imshow(segmentation_labels[k_target])
        
        #axs[2].imshow(segmentation_labels[k_target])
        
        colors = dict(hep='r', fib='g', u='y', bad='b')
        gt_coords = target_coords[k_target]
        
        #axs[0].imshow(img_g,  cmap = 'gray')
        axs[1].plot(gt_coords[..., 0], gt_coords[..., 1], 'xc')
        for (k, coords) in coords_by_class[k_target].items():
            if coords.size == 0:
                continue
            axs[0].plot(coords[..., 1], coords[..., 0], 'o', color=colors[k])
    
    #%%
    ctr_root_dir = '/Users/avelinojaver/OneDrive - Nexus365/papers/miccai2019/data/hepatocytes/BBBC026/BBBC026_control/'
    ctr_root_dir = Path(ctr_root_dir)
    fnames = list(ctr_root_dir.glob('*.h5'))
    
    all_data = []
    for fname in tqdm.tqdm(fnames):
        with tables.File(str(fname), 'r') as fid:
            preds = fid.get_node('/exported_data')[:]
        
        preds = preds.astype(np.float32)
        pred_maps = {'hep' : preds[..., 0], 'fib' : preds[..., 1]}
        
        
        kernel = (7,7)
        #kernel = (11,11)
        segmentation_labels = {}
        for k, mm in pred_maps.items():
            mm = cv2.blur(mm, kernel)
            
            #min_area, max_area = [(x/2)**2*np.pi for x in diameter_limits[k]]
            min_area, max_area = 300, 5000
            lab, _ = get_labels(mm, th_min = 0.5, min_area = min_area, max_area = max_area)
            
            #lab, _ = get_labels(mm, th_min = 0.5)
            #lab = _filter_by_size(lab, *diameter_limits[k])
            
            segmentation_labels[k] = lab
         
        hep_counts = np.unique(segmentation_labels['hep']).size-1
        fib_counts = np.unique(segmentation_labels['fib']).size-1
        well_id, pos_id = fname.name.split('_')[1:3]
        
        
        all_data.append((well_id, pos_id, hep_counts, fib_counts))
        
    df = pd.DataFrame(all_data, columns=['well_id', 'pos_id', 'hep', 'fib'])
    
    
    
    
    k_counts = df.groupby('well_id').agg('sum')
    for k in ['hep', 'fib']:
        dat = [('neg' if w.endswith('01') else 'pos', v, w) for w,v in zip(k_counts[k].index, k_counts[k].values)]
        dat = pd.DataFrame(dat, columns = ['type', 'vals', 'well_id'])
        
        
        plt.figure()
        sns.boxplot('type', 'vals', data=dat)
        plt.title(k)
        
        pos = dat.loc[dat['type'] == 'pos', 'vals']
        neg = dat.loc[dat['type'] == 'neg', 'vals']
        
        pos_m = np.mean(pos)
        neg_m = np.mean(neg)
        
        pos_s = np.std(pos)
        neg_s = np.std(neg)
        
        Z = 1 - 3*(pos_s + neg_s)/(pos_m - neg_m)
        
        t, p = ttest_ind(pos, neg)
        ss = f"{k} -> Z'-score={Z:.3}  p={p:.3}"

        print(ss)
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:07:42 2019

@author: avelinojaver
"""
from pathlib import Path

import tqdm
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind

#%%
if __name__ == '__main__':
    min_area = 300
    max_area = 5000
    
    
    #bn = 'sBBBC026-hepatocytes_l1smooth_20190220_184125_unet_adam_lr0.00032_wd0.0_batch32'
    
    #bn = 'CELLS_349_BBBC026-separated_unet_l1smooth_20190224_105903_adam_lr0.00032_wd0.0_batch32'
    #bn = 'CELLS_349_BBBC026-hepatocytes_unet_l1smooth_20190224_105953_adam_lr0.00032_wd0.0_batch32'
    
    #bn = 'BBBC026-separated_unet_l1smooth_20190226_082017_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-hepatocytes_unet_l1smooth_20190226_082040_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-fibroblasts_unet_l1smooth_20190226_082007_adam_lr0.00032_wd0.0_batch32'
    
    bn = 'CELLS_349_BBBC026-separated_unet_l1smooth_20190226_082017_adam_lr0.00032_wd0.0_batch32'
    #bn = 'CELLS_349_BBBC026-hepatocytes_unet_l1smooth_20190226_082040_adam_lr0.00032_wd0.0_batch32'
    #bn = 'CELLS_349_BBBC026-fibroblasts_unet_l1smooth_20190226_082007_adam_lr0.00032_wd0.0_batch32'
    
    root_dir = Path.home() / 'OneDrive - Nexus365/papers/miccai2019/data/hepatocytes/'
    
    
    fname = root_dir / f'{bn}.csv'
    df_ori = pd.read_csv(str(fname))
    df_ori = df_ori[(df_ori['area'] >= min_area) & (df_ori['area'] <= max_area)]
        
    for k_class in ['hep', 'fib']:
        df = df_ori[df_ori['type'] == k_class]
        
        if df.size == 0:
            continue
        
        plt.figure()
        plt.hist(df['area'], 100)
        
        #%%
        #df_negative = df[df['well_id'].str.contains('01')]
        #df_positive = df[df['well_id'].str.contains('23')]
        
        
        #counts_per_well = df.groupby('well_id').count()
        gg = df.groupby(['well_id', 'site_id'])
        
        all_counts = {}
        for (well_id, site_id), dat in gg:
            if not well_id in all_counts:
                all_counts[well_id] = []
            all_counts[well_id].append(len(dat))
        
        #%%
        _negative = [(k, v) for k,v in all_counts.items() if '01' in k]
        _positive = [(k, v) for k,v in all_counts.items() if '23' in k]
        
        #%%
        dat_s_n = [('neg', x) for k, d in _negative for x in d]
        dat_s_p = [('pos', x) for k, d in _positive for x in d]
        df_m = pd.DataFrame(dat_s_n + dat_s_p, columns = ['type', 'vals'])
        
        plt.figure()
        sns.boxplot('type', 'vals', data=df_m)
        #%%
        dat_n = [('neg', np.sum(d)) for k, d in _negative]
        dat_p = [('pos', np.sum(d)) for k, d in _positive]
        
        #dat_n = [('neg', np.median(d)) for k, d in _negative]
        #dat_p = [('pos', np.median(d)) for k, d in _positive]
        
        
        df_plates = pd.DataFrame(dat_n + dat_p, columns = ['type', 'vals'])
        
        pos = [x[1] for x in dat_p]
        neg = [x[1] for x in dat_n]
        
        pos_m = np.mean(pos)
        neg_m = np.mean(neg)
        
        pos_s = np.std(pos)
        neg_s = np.std(neg)
        
        Z = 1 - 3*(pos_s + neg_s)/(pos_m - neg_m)
        #%%
        
        t, p = ttest_ind(pos, neg)
        ss = f"Z'-score={Z:.3}  p={p:.3}"
        
        #%%
        save_name = root_dir / f'{k_class}_{bn}.pdf'
        
        plt.figure()
        sns.boxplot('type', 'vals', data=df_plates)
        
        #plt.ylim([0, 180])
        plt.title(ss)
        plt.savefig(str(save_name))
        
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:24:10 2019

@author: avelinojaver
"""
import os
from pathlib import Path

import numpy as np
import tqdm
import pickle
from collections import defaultdict

import matplotlib.pylab as plt
import matplotlib.patches as patches

if __name__ == '__main__':
    
    bn2check_dict = {
            
#    'less_bgnd' :
#                ['BBBC042-colour-v4-S5_unet-filter_l1smooth_20190709_202139_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-colour-v4-S10_unet-filter_l1smooth_20190709_202129_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-colour-v4-S25_unet-filter_l1smooth_20190710_000135_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-colour-v4-S100_unet-filter_l1smooth_20190710_001136_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-colour-v4_unet-filter_l1smooth_20190709_223456_adam_lr0.00032_wd0.0_batch32'
#                ],
#
#    'more_bgnd' :
#                ['BBBC042-colour-bgnd-S5_unet-filter_l1smooth_20190710_145222_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-colour-bgnd-S10_unet-filter_l1smooth_20190710_164420_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-colour-bgnd-S25_unet-filter_l1smooth_20190710_162339_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-colour-bgnd-S100_unet-filter_l1smooth_20190710_151403_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-colour-bgnd_unet-filter_l1smooth_20190711_180845_adam_lr0.00032_wd0.0_batch32'
#                ],
#    'simple' :
#                [
#                'BBBC042-simple-S5_unet-filter_l1smooth_20190712_045727_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-simple-S10_unet-filter_l1smooth_20190711_204742_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-simple-S25_unet-filter_l1smooth_20190711_204744_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-simple-S100_unet-filter_l1smooth_20190712_043730_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-simple_unet-filter_l1smooth_20190711_195905_adam_lr0.00032_wd0.0_batch32'
#                ],
    'Ours' :
                [
                'BBBC042-simple-bgnd-S5_unet-filter_l1smooth_20190712_144153_adam_lr0.00032_wd0.0_batch32',
                'BBBC042-simple-bgnd-S10_unet-filter_l1smooth_20190712_160608_adam_lr0.00032_wd0.0_batch32',
                'BBBC042-simple-bgnd-S25_unet-filter_l1smooth_20190712_105659_adam_lr0.00032_wd0.0_batch32',
                'BBBC042-simple-bgnd-S100_unet-filter_l1smooth_20190712_144219_adam_lr0.00032_wd0.0_batch32',
                'BBBC042-simple-bgnd_unet-filter_l1smooth_20190713_162625_adam-_lr0.00032_wd0.0_batch32',
                ],
#    'simple_step' :
#                [
#                'BBBC042-simple-bgnd-S5_unet-filter_l1smooth_20190713_123452_adam-stepLR-30-0.1_lr0.00032_wd0.0_batch32',
#                'BBBC042-simple-bgnd-S10_unet-filter_l1smooth_20190713_144738_adam-stepLR-30-0.1_lr0.00032_wd0.0_batch32',
#                'BBBC042-simple-bgnd-S25_unet-filter_l1smooth_20190713_161618_adam-stepLR-30-0.1_lr0.00032_wd0.0_batch32',
#                'BBBC042-simple-bgnd-S100_unet-filter_l1smooth_20190713_124826_adam-stepLR-30-0.1_lr0.00032_wd0.0_batch32',
#                'BBBC042-simple-bgnd_unet-filter_l1smooth_20190713_124210_adam-stepLR-30-0.1_lr0.00032_wd0.0_batch32',
#                ]
    }
        
    #%%    
    
    results = defaultdict(list)
    for set_type, bn2check in bn2check_dict.items():
        fig, ax = plt.subplots(1,1)
            
        for bn in tqdm.tqdm(bn2check):
            num = bn.partition('-S')[-1].partition('_')[0]
            if num:
                num = int(num)
            else:
                num = 1000
            
            subdir = bn.partition('_')[0]
            model_dir = model_path = Path.home() / 'workspace/denoising/results' / subdir / bn
            
            results_file = model_dir / 'scores_otsu_val.p'
            with open(results_file, 'rb') as fid:
                metrics_test, IoU_cutoffs, epochs2check, thresh2check, min_bbox_sizes = pickle.load(fid)
                
            metric_test = metrics_test[1]
            TP, FP, FN = metric_test
            P = TP/(TP+FP)
            R = TP/(TP+FN)
            F1 = 2*P*R/(P+R)
            
            ith, isize = 0, 2
            
            ax.plot(epochs2check, F1[:, ith, isize], label = num)
            plt.title(set_type)
        plt.legend()
    #%%
    fig, ax = plt.subplots(1,1, figsize = (3, 3))
    icut = 0
    set_type = 'test'
    fname = Path.home() / f'workspace/box_detection_BBBC042_{set_type}.p'
    with open(fname, 'rb') as fid:
        results = pickle.load(fid)
    
    lengends_dict = {'retinanet_adam' : 'Retinanet', 
                     'fasterrcnn_sgd' : 'FasterRCNN',
                     'fasterrcnn_sgd_augmentation': 'FasterRCNN +\n Augmentations'}
    #plt.figure()
    #for set_type, scores in results.items():
    for lab in ['retinanet_adam',  'fasterrcnn_sgd', 'fasterrcnn_sgd_augmentation']:
        scores = results[lab]
        res_y = []
        res_x = []
        for bn, metrics in scores.items():
            num = bn.partition('-max')[-1].partition('-')[0]
            if num:
                num = int(num)
            else:
                num = 1000
            res_x.append(num)
            TP, FP, FN = metrics.T
            P = TP/(TP+FP)
            R = TP/(TP+FN)
            F1 = 2*P*R/(P+R)
            
            res_y.append((P, R, F1))
            
            
        res_y = np.array(res_y)
        res_x = np.array(res_x)
        
        ind = np.argsort(res_x)
        res_y = res_y[ind]
        res_x = res_x[ind]
        
        plt.plot(res_x, res_y[:, -1, icut], 'o-', label = lengends_dict[lab])
        
    #%%
    results = defaultdict(list)
    for set_type, bn2check in bn2check_dict.items():
    
        for bn in tqdm.tqdm(bn2check):
            num = bn.partition('-S')[-1].partition('_')[0]
            if num:
                num = int(num)
            else:
                num = 1000
            
            subdir = bn.partition('_')[0]
            model_dir = model_path = Path.home() / 'workspace/denoising/results' / subdir / bn
            
            results_file = model_dir / 'scores_otsu_val.p'
            with open(results_file, 'rb') as fid:
                metrics_val,  IoU_cutoffs, epochs2check, thresh2check, min_bbox_sizes = pickle.load(fid)
            
            results_file = model_dir / 'scores_otsu_test.p'
            with open(results_file, 'rb') as fid:
                metrics_test = pickle.load(fid)[0]
                
            res = []
            print(bn)
            for metric_val, metric_test in zip(metrics_val, metrics_test):
                
                TP, FP, FN = metric_val
                #%%
                P = TP/(TP+FP)
                R = TP/(TP+FN)
                F1 = 2*P*R/(P+R)
                #%%
                iepoch,  ith, isize = np.unravel_index(np.nanargmax(F1), F1.shape)
                
                iepoch,  ith, isize = 19, 0, 2
                
                th = thresh2check[ith]
                min_box = min_bbox_sizes[isize]
                epoch = epochs2check[iepoch]
                
                F1_val = F1[iepoch,  ith, isize]
                P_val = P[iepoch,  ith, isize]
                R_val = R[iepoch, ith, isize]
                
                TP, FP, FN = metric_test[:, iepoch,  ith, isize]
                P_test = TP/(TP+FP)
                R_test = TP/(TP+FN)
                F1_test = 2*P_test*R_test/(P_test + R_test)
                
                
                print(f'P {P_val:.3f} | R {R_val:.3f} | F1 {F1_val:.3f} | epoch{epoch} | th{th:.3f} | min_box{min_box}') 
                      
                res.append([(P_val, R_val, F1_val), (P_test, R_test, F1_test)])
            
            results[set_type].append((num, res))
    
    #plt.figure()
    for set_type, scores in results.items():
        res_x, res_all = zip(*scores)
        
        #use the second icut that corresponds to u
        res_cutoffs = list(zip(*res_all))
        (P_val, R_val, F1_val), (P_test, R_test, F1_test) = map(lambda x : zip(*x), zip(*res_cutoffs[icut]))
        
        res_y = F1_test
        #res_y = P_test
        #res_y = R_test
        
        plt.plot(res_x, res_y, 's-', label = set_type)
    plt.xscale('log')
    
    #%%
    
    
    plt.xscale('log')
    plt.legend()
    plt.xlabel('Number of Training Images')
    plt.ylabel('F1-score')
    plt.ylim([0, 0.8])
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:58:14 2019

@author: avelinojaver
"""
from pathlib import Path
import pickle
import numpy as np

import matplotlib.pylab as plt
if __name__ == '__main__':
    set_type = 'test'
    
    fname = Path.home() / f'workspace/box_detection_BBBC042_{set_type}.p'
    with open(fname, 'rb') as fid:
        results = pickle.load(fid)
    #%%
    
    plt.figure()
    for set_type, scores in results.items():
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
            
            res_y.append(F1)
        res_y = np.array(res_y)
        res_x = np.array(res_x)
        
        ind = np.argsort(res_x)
        res_y = res_y[ind]
        res_x = res_x[ind]
        
        plt.plot(res_x, res_y[:, 0], 'o-', label = set_type)
    plt.xscale('log')
    plt.legend()
        
        
    
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:48:03 2019

@author: avelinojaver
"""

import math
from pathlib import Path
import random

def divided_data(data_dict):
    test_frac = 0.1 
    v_keys = list(data_dict.keys())
    random.shuffle(v_keys)
    
    tot = len(v_keys)
    test_ii = math.ceil(tot*test_frac)
    
    test_k = v_keys[:test_ii]
    train_k = v_keys[test_ii:]
    
    return test_k, train_k


if __name__ == '__main__':    
    #root_dir = Path.home() / 'OneDrive - Nexus365/microglia/data/cell_bgnd_divided/'
    root_dir = Path.home() / 'workspace/denoising/data/microglia/cell_bgnd_divided'
    
    cell_root_dir = root_dir / 'cell_images'
    d_cell_root_dir = root_dir / 'cell_images_dilated'
    bgnd_root_dir = root_dir / 'bgnd_images'
    
    def get_files(_root_dir):
        return [x for x in _root_dir.rglob('*.tif') if not x.name.startswith('.')]
    
    cell_files = get_files(cell_root_dir)
    d_cell_files = get_files(d_cell_root_dir)
    bgnd_files = get_files(bgnd_root_dir)
    
    
    bgnd_data = {}
    for x in bgnd_files:
        
        bn = x.parent.name + '/' + x.name.split('_z')[0]
        if bn not in bgnd_data:
            bgnd_data[bn] = []
        bgnd_data[bn].append(x)
    
    cell_data = {}
    for x in cell_files:
        if x.name.startswith('.'):
            continue
        
        bn = x.parents[1].name + '/' + x.parent.name.partition('_')[-1].partition('_z')[0]
        if bn not in cell_data:
            cell_data[bn] = []
        cell_data[bn].append(x)
    
    d_cell_data = {}
    for x in d_cell_files:
        if x.name.startswith('.'):
            continue
        
        bn = x.parents[1].name + '/' + x.parent.name.partition('_')[-1].partition('_z')[0]
        if bn not in d_cell_data:
            d_cell_data[bn] = []
        d_cell_data[bn].append(x)
      #%% 
    test_k, train_k = divided_data(bgnd_data)
    
    
    def _divided(data, keys):
        return [x for k in keys if k in data for x in data[k]]
    
    
    cell_test = _divided(cell_data, test_k)
    d_cell_test = _divided(d_cell_data, test_k)
    bgnd_test = _divided(bgnd_data, test_k)
    
    cell_train = _divided(cell_data, train_k)
    d_cell_train = _divided(d_cell_data, train_k)
    bgnd_train = _divided(bgnd_data, train_k)
    
    
    
    assert len(cell_test) + len(cell_train) == len(cell_files)
    assert len(d_cell_test) + len(d_cell_train) == len(d_cell_files)
    assert len(bgnd_test) + len(bgnd_train) == len(bgnd_files)
    
    #%%
    for x in bgnd_train + cell_train + d_cell_train:
        
        new_fname = root_dir / 'train' / str(x).replace(str(root_dir), '')[1:]
        new_fname.parent.mkdir(exist_ok = True, parents = True)
        x.rename(new_fname)
        
    #%%
    for x in bgnd_test + cell_test + d_cell_test:
        
        new_fname = root_dir / 'test' / str(x).replace(str(root_dir), '')[1:]
        new_fname.parent.mkdir(exist_ok = True, parents = True)
        x.rename(new_fname)
    #%%
    
        
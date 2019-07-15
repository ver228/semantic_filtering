#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 22:42:04 2019

@author: avelinojaver
"""

import shutil
from pathlib import Path
import tqdm
#root_dir = Path.home() / 'workspace/denoising/data/BBBC042_colour/'
#root_dir = Path.home() / 'workspace/denoising/data/BBBC042_colour_more_bgnd/'
#root_dir = Path.home() / 'workspace/denoising/data/BBBC042_v2/'
root_dir = Path.home() / 'workspace/denoising/data/BBBC042_bgnd/'

for max_id in [5, 10, 25, 100]:
    src_dir = root_dir / 'train'
    save_dir = root_dir / f'train_S{max_id}'
    
    fnames = src_dir.rglob('*.tif')
    
    for fname in tqdm.tqdm(fnames, desc = f'{root_dir.name} {max_id}'):
        img_id = int(fname.stem.split('_')[0])
        
        if img_id <= max_id: #the set starts with index 1...
            dname = fname.parent.name
            
            dst_dir = save_dir / dname 
            
            dst_dir.mkdir(parents = True, exist_ok = True)
            shutil.copyfile(fname, dst_dir / fname.name)


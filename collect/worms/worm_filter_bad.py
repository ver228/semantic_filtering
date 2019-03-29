#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:26:13 2019

@author: avelinojaver
"""
from pathlib import Path
import cv2
import numpy as np
import os
import tqdm
#%%
if __name__ == '__main__':
    data_root = Path.home() / 'workspace/denoising/data/c_elegans_divergent/test/'
    save_root =  Path.home() / 'workspace/denoising/data/c_elegans_divergent/check_test/'
    
    
    if False:
        for fname in tqdm.tqdm(data_root.rglob('*.tif')):
            if fname.name.startswith('.'):
                continue
            
            img = cv2.imread(str(fname), -1)
            
            fname_r = fname.parent / ('R_' + fname.stem + '.png')
            target = cv2.imread(str(fname_r), -1)
            
            
            
            img_rgb = np.stack([img]*3, axis=2)
            img_rgb[..., 0] = np.where(target>0, img*0.5 + 125 , img)
            
            
            save_f = Path(str(fname).replace(str(data_root), str(save_root)))
            save_f.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(save_f), img_rgb)
    
    #%%
    if True:
        def _get_files(dname):
            return [str(x).replace(str(dname), '') for x in dname.rglob('*.tif') if not x.name.startswith('.') and not x.name.startswith('_')]
        
        fnames_rgb = _get_files(save_root)
        fnames = _get_files(data_root)
        
        #%%
        bad_files = [data_root / x[1:] for x in set(fnames) - set(fnames_rgb)]
        
        for x in bad_files:
            x.rename(x.parent / ('_' + x.name))
        
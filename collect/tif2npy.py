#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 18:37:18 2019

@author: avelinojaver
"""
from pathlib import Path
import cv2
import numpy as np
import tqdm

def convert(root_dir, ext='.tif'):
    fnames = Path(root_dir).rglob(f'*{ext}')
    fnames = [x for x in fnames if not x.name.startswith('.')]
    
    for fname in tqdm.tqdm(fnames):
        img = cv2.imread(str(fname), -1)
        np.save(fname.with_suffix('.npy'), img)
        

if __name__ == '__main__':
    import fire
    fire.Fire(convert)
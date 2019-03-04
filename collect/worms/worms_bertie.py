#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 18:05:43 2018

@author: avelinojaver
"""

from pathlib import Path
import tqdm
import tables
import os
import cv2
import numpy as np
import pandas as pd
#%%
def _process_dir(row, dist_th = 100, min_pix_change = 15):
    prefix_, fnames = row
    
    imgs = np.array([cv2.imread(str(x), -1).astype(np.float32) for x in fnames])
    if len(imgs) < 2:
        return pd.DataFrame([])
    
    imgs_diff = np.diff(imgs, axis=0)
    
    imgs_diff = np.diff(imgs, axis=0)
    imgs_diff_m = [cv2.medianBlur(x, 7) for x in imgs_diff]
    
    img_shape = imgs_diff_m[0].shape
    max_coords = [np.unravel_index(np.argmax(x), img_shape) for x in imgs_diff_m]
    pix_max = [img[cc] for img,cc in zip(imgs_diff_m, max_coords)]
    
    
    min_coords = [np.unravel_index(np.argmin(x), img_shape) for x in imgs_diff_m]
    pix_min = [img[cc] for img,cc in zip(imgs_diff_m, min_coords)]
    
    largest_pix_change = np.maximum(pix_max, np.abs(pix_min))
    
    R = np.sqrt(np.sum((np.array(max_coords) - np.array(min_coords))**2, axis=1))
    valid_inds, =  np.where((R > dist_th) & (largest_pix_change>=min_pix_change))
    
    valid_files = []
    for ind in valid_inds:
        max_c, min_c =  max_coords[ind], min_coords[ind]
        row_max = (prefix_, ind, ind+1, *max_c)
        row_min = (prefix_, ind, ind+1, *min_c)
        valid_files += [row_max, row_min]
        
    valid_files = pd.DataFrame(valid_files, columns = ['prefix', 'prev_ind', 'after_ind', 'target_coord_y', 'target_coord_x'])
    
    return valid_files
#%%
def _process_debug(row):
    try:
        return _process_dir(row)
    except:
        return row
#%%

if __name__ == '__main__':
    save_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie/valid_files.csv'
    
    root_src_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie/')
    
    fnames = list(root_src_dir.rglob('*.tif'))
    
    
    videos_dict = {}
    for fname in fnames:
        dname = fname.parent
        
        
        prefix_ = str(dname).replace(str(root_src_dir), '')
        prefix_ = prefix_[1:] if prefix_[0] == os.sep else prefix_
        
        
        if not prefix_ in  videos_dict:
            videos_dict[prefix_] = []
        videos_dict[prefix_].append(fname)
        
    videos_data = [(k, sorted(x, key = lambda x : int(x.stem))) for k,x in videos_dict.items()]
    
    #%%
    from multiprocessing import Pool
    with Pool(6) as p:
        mapper = p.imap_unordered(_process_dir, videos_data)
        results = list(tqdm.tqdm(mapper, total=len(videos_data)))
    results = pd.concat(results, ignore_index = True)
    
    #%%
    results.to_csv(save_file, index=False)
    
    #%%
    
    
#    #%%
#    row = results.iloc[-3]
#    
#    dname = root_src_dir / row['prefix'] 
#    
#    prev_ind, after_ind = row['prev_ind'], row['after_ind'] 
#    
#    img1 = cv2.imread(str(dname / f'{prev_ind}.tif'), -1)
#    img2 = cv2.imread(str(dname / f'{after_ind}.tif'), -1)
#    
#    fig, axs = plt.subplots(1,3, sharex=True, sharey=True)
#    axs[0].imshow(img1, cmap='gray')
#    axs[1].imshow(img2, cmap='gray')
#    
#    x = img1.astype(np.float32)-img2
#    
#    axs[2].imshow(x, cmap='gray')
#    axs[2].plot(row['min_coord_x'] , row['min_coord_y'] , 'rx')
#    
#    axs[2].plot(row['max_coord_x'] , row['max_coord_y'] , 'bx')
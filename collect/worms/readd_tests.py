#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 18:05:43 2018

@author: avelinojaver
"""


from pathlib import Path
import tqdm
import tables
import cv2
import math
import pandas as pd
import numpy as np

DIVERGENT_SET = ['N2',
 'CB4856',
 'DL238',
 'JU775',
 'MY16',
 'MY23',
 'CX11314',
 'ED3017',
 'EG4725',
 'LKC34',
 'JT11398',
 'JU258']



#%%
if __name__ == '__main__':
    rr = Path.home() / 'workspace/denoising/data/c_elegans_divergent/test'
    dnames = set([x.parent for x in rr.rglob('*.tif') if not x.name.startswith('.')])
    
    
    src_root_dir = Path.home() / 'workspace/WormData/screenings/CeNDR'
    fnames = [x for x in src_root_dir.rglob('*.hdf5') if 'MaskedVideos' in str(x)]
    src_fnames = {x.stem:x for x in fnames if any([x.name.startswith(s) for s in DIVERGENT_SET])}
    #%%
    for dname in tqdm.tqdm(dnames):
        bn = dname.name
        fname = src_fnames[bn]
        
    
        with tables.File(str(fname), 'r') as fid:
            imgs = fid.get_node('/full_data')[:]
            save_interval = int(fid.get_node('/full_data')._v_attrs['save_interval'])
            
        path_r = str(fname.parent / fname.stem).replace(str(src_root_dir), '')[1:]
        
        for ii, img in enumerate(imgs):
            save_name = dname / '{}.tif'.format(ii)
            cv2.imwrite( str(save_name), img)        
    #%%
    
    
    
    
    
#    
#    test_frac = 0.1
#    
#    test_ind = math.ceil(len(fnames)*test_frac)
#    fnames_test = fnames[:test_ind]
#    fnames_train = fnames[test_ind:]
#    
#    
#    #%%

#      
#        
#        feat_file = str(fname).replace('/MaskedVideos/', '/Results/').replace('.hdf5', '_featuresN.hdf5')
#        
#        with pd.HDFStore(str(feat_file), 'r') as ske_file_id:
#            trajectories_data = ske_file_id['/trajectories_data']
#        #group the data by frame to easily be able to access to it
#        traj_group_by_frame = trajectories_data.groupby('frame_number')
#        #%%
#        for ii, img in enumerate(imgs):
#            frame_number = int(ii*save_interval)
#            #read the data from all the worms in this frame
#            frame_data = traj_group_by_frame.get_group(frame_number)
#            labeled_mask = img2bw(img, frame_data)
#            
#            save_name = new_dname / 'R_{}.png'.format(ii)
#            cv2.imwrite( str(save_name), labeled_mask)
#            #%%
#    for fname in tqdm.tqdm(fnames_train):
#        with tables.File(str(fname), 'r') as fid:
#            imgs = fid.get_node('/full_data')[:]
#            save_interval = int(fid.get_node('/full_data')._v_attrs['save_interval'])
#            
#        path_r = str(fname.parent / fname.stem).replace(str(src_root_dir), '')[1:]
#        new_dname = save_root_dir / 'train' / path_r
#        
#        new_dname.mkdir(parents=True, exist_ok=True)
#        
#        for ii, img in enumerate(imgs):
#            save_name = new_dname / '{}.tif'.format(ii)
#            cv2.imwrite( str(save_name), img)
#      
#    
#    

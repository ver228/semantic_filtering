#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 18:05:43 2018

@author: avelinojaver
"""

from tierpsy.analysis.ske_create.helperIterROI import getAllImgROI
from tierpsy.analysis.ske_create.getSkeletonsTables import getWormMask

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

def img2bw(img, frame_data):
    #this functions crop all the ROIs of all the worms located in the image
    worms_in_frame = getAllImgROI(img, frame_data)
    
    #now i want to pass this into a binary mask with the segmentation
    labeled_mask = np.zeros_like(img)
    for iworm, (ind, (worm_img, roi_corner)) in enumerate(worms_in_frame.items()):
        
        row_data = trajectories_data.loc[ind]
        #here is where the ROI is really segmented. It uses threshold plus some morphology operations to clean a bit
        worm_mask, worm_cnt, cnt_area = getWormMask(worm_img, 
                                     row_data['threshold'], 
                                     strel_size=5,
                                     min_blob_area=row_data['area'] / 2, 
                                     is_light_background = True
                                     )
        
        #now I want to put the ROI mask into the larger labelled image
        xi,yi = roi_corner
        ss = worm_mask.shape
        
        #here I am saving only the pixels located to the worm. 
        #This is safer than assigned the whole ROI in case of overlaping regions.
        m_roi = labeled_mask[yi:yi+ss[0], xi:xi+ss[1]].view()
        m_roi[worm_mask>0] += 1
        
    return labeled_mask

#%%
if __name__ == '__main__':
    src_root_dir = Path.home() / 'workspace/WormData/screenings/CeNDR'
    save_root_dir = Path.home() / 'workspace/denoising/data/c_elegans_divergent'
    
    fnames = [x for x in src_root_dir.rglob('*.hdf5') if 'MaskedVideos' in str(x)]
    fnames = [x for x in fnames if any([x.name.startswith(s) for s in DIVERGENT_SET])]
    
    test_frac = 0.1
    
    test_ind = math.ceil(len(fnames)*test_frac)
    fnames_test = fnames[:test_ind]
    fnames_train = fnames[test_ind:]
    
    
    #%%
    for fname in tqdm.tqdm(fnames_test):
        with tables.File(str(fname), 'r') as fid:
            imgs = fid.get_node('/full_data')[:]
            save_interval = int(fid.get_node('/full_data')._v_attrs['save_interval'])
            
        path_r = str(fname.parent / fname.stem).replace(str(src_root_dir), '')[1:]
        new_dname = save_root_dir / 'test' / path_r
        
        new_dname.mkdir(parents=True, exist_ok=True)
        
        for ii, img in enumerate(imgs):
            save_name = new_dname / '{}.tif'.format(ii)
            cv2.imwrite( str(save_name), img)
      
        
        feat_file = str(fname).replace('/MaskedVideos/', '/Results/').replace('.hdf5', '_featuresN.hdf5')
        
        with pd.HDFStore(str(feat_file), 'r') as ske_file_id:
            trajectories_data = ske_file_id['/trajectories_data']
        #group the data by frame to easily be able to access to it
        traj_group_by_frame = trajectories_data.groupby('frame_number')
        #%%
        for ii, img in enumerate(imgs):
            frame_number = int(ii*save_interval)
            #read the data from all the worms in this frame
            frame_data = traj_group_by_frame.get_group(frame_number)
            labeled_mask = img2bw(img, frame_data)
            
            save_name = new_dname / 'R_{}.png'.format(ii)
            cv2.imwrite( str(save_name), labeled_mask)
            #%%
    for fname in tqdm.tqdm(fnames_train):
        with tables.File(str(fname), 'r') as fid:
            imgs = fid.get_node('/full_data')[:]
            save_interval = int(fid.get_node('/full_data')._v_attrs['save_interval'])
            
        path_r = str(fname.parent / fname.stem).replace(str(src_root_dir), '')[1:]
        new_dname = save_root_dir / 'train' / path_r
        
        new_dname.mkdir(parents=True, exist_ok=True)
        
        for ii, img in enumerate(imgs):
            save_name = new_dname / '{}.tif'.format(ii)
            cv2.imwrite( str(save_name), img)
      
    
    

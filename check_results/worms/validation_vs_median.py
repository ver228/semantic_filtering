#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:17:06 2019

@author: avelinojaver
"""

from pathlib import Path
import cv2
import numpy as np
import tqdm
#%%
if __name__ == '__main__':
    #root_dir = '/Volumes/rescomp1/data/denoising/data/c_elegans_divergent/test/MaskedVideos/'
    #root_dir = '/Volumes/rescomp1/data/denoising/data/c_elegans_full/test/Drug_Screening/MaskedVideos/MK_Olazapine_220817/'
    #root_dir = '/Volumes/rescomp1/data/denoising/data/c_elegans_full/test/Drug_Screening/MaskedVideos/MK_Screening_Amisulpride_Chlopromazine_CB4856_240817/'
    #root_dir = '/Volumes/rescomp1/data/denoising/data/c_elegans_full/test/Serena_WT_Screening/MaskedVideos/Agg_15.1_180303/'
    #root_dir = '/Volumes/rescomp1/data/denoising/data/c_elegans_full/test/MMP/MaskedVideos/MMP_Set1_011217/'
    
    #root_dir = '/Volumes/rescomp1/data/denoising/data/bertie_c_elegans/2017_06_28/'
    #root_dir = '/Volumes/rescomp1/data/denoising/data/c_elegans_full/test/MMP/MaskedVideos/MMP_Set2_201217/'
    
    #root_dir = '/Volumes/rescomp1/data/denoising/data/c_elegans_full/test/MMP/MaskedVideos/MMP_Set2_201217/VC20527_worms10_food1-10_Set2_Pos5_Ch3_20122017_152030'
    #root_dir = '/Volumes/rescomp1/data/denoising/data/c_elegans_full/test/Lidia/MaskedVideos/Optogenetics-day1/'
    #root_dir = '/Volumes/rescomp1/data/denoising/data/c_elegans_full/test/Lidia/MaskedVideos/Optogenetics-day14/'
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie/2017_06_28/'
    #root_dir = '/Volumes/rescomp1/data/denoising/data/c_elegans_full/test/Pratheeban/First_Set/MaskedVideos/Old_Adult/'
    
    root_dir = Path(root_dir)
    
    fnames = root_dir.rglob('*.tif')
    fnames = [x for x in fnames if not x.name.startswith('.')]
    
    data = {}
    for fname in fnames:
        bn = fname.parent.name
        if not bn in data:
            data[bn] = []
        data[bn].append(fname)
        
     #%%
    for bn, fnames in tqdm.tqdm(data.items()):
        imgs = [cv2.imread(str(x), -1) for x in fnames]
        img_m = np.max(imgs, axis=0)
        
        x = imgs[-1]
        dd = x.astype(np.float32) - img_m
    
        fig, axs = plt.subplots(1, 3, figsize = (20, 8), sharex = True, sharey = True)
        axs[0].imshow(x)
        axs[1].imshow(img_m)
        axs[2].imshow(dd)
        
        plt.suptitle((len(fnames), bn))
        
        
    #%%
    import tables
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/manual_annotations/raw/Phase3/MaskedVideos/'
    #root_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies/'
    root_dir = Path(root_dir)
    for fname in root_dir.glob('*.hdf5'):
    
        with tables.File(fname, 'r') as fid:
            imgs = fid.get_node('/full_data')[:]
        
        img_m = np.max(imgs, axis=0)
        
        x = imgs[-1]
        dd = x.astype(np.float32) - img_m
    
        fig, axs = plt.subplots(1, 3, figsize = (20, 8), sharex = True, sharey = True)
        axs[0].imshow(x)
        axs[1].imshow(img_m)
        axs[2].imshow(dd)
        
        plt.suptitle((len(imgs), fname.name))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 13:47:54 2019

@author: avelinojaver
"""

import tables
import cv2
import tqdm
import numpy as np
from pathlib import Path
TABLE_FILTERS = tables.Filters(
    complevel=5,
    complib='blosc',
    shuffle=True,
    fletcher32=True)
#%%
if __name__ == '__main__':
    #fname = '/Users/avelinojaver/OneDrive - Nexus365/survellance_videos/stanford_campus_dataset/video.mov'
    root_dir = '/Volumes/rescomp1/data/datasets/stanford_campus_dataset/videos/'
    root_dir = Path(root_dir)
    
    bad_files = []
    
    fnames = [x for x in root_dir.rglob('*.mov') if not x.name.startswith('.')]
    
    for fname in tqdm.tqdm(fnames):
        fname = str(fname)
        #save_name = fname.parent / fname.stem + '.hdf5'
        vcap = cv2.VideoCapture(str(fname))
        
        width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tot_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        
        batch_size = 16
        imgs = []
        for ii in tqdm.tqdm(range(batch_size)):
            ret, img = vcap.read()
            
            
            imgs.append(img)
            
        imgs = np.array(imgs).astype(np.float32)
        
        img1 = imgs[0]/255
        img2 = imgs[15]/255
        
        dd = np.abs(img1 - img2).sum(axis=-1)
        
        fig, axs = plt.subplots(1,3, sharex=True, sharey=True)
        axs[0].imshow(img1)
        axs[1].imshow(img2)
        axs[2].imshow(dd)
    #%%
    def auto_canny(image, sigma = 0.05):
        # compute the median of the single channel pixel intensities
        v = np.median(image)
 
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        return edged
 
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
    laplacian = cv2.Laplacian(img_gray,cv2.CV_32F)
    fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
    axs[0].imshow(img_gray)
    
    edged = auto_canny(img_gray)
    axs[1].imshow(edged)
    
    #%%
    
    
    
    
#    with tables.File(save_name, 'w') as fid:
#        imgs = fid.create_carray('/',
#                          'frames',
#                          tables.UInt8Atom(),
#                          shape = (tot_frames, height, width, 3),
#                          chunkshape = (1, height, width, 3),
#                          filters = TABLE_FILTERS
#                          )
        
        
        
            
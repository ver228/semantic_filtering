#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 18:05:43 2018

@author: avelinojaver
"""

from pathlib import Path
import cv2
import pandas as pd
import os
import random
import numpy as np
from torch.utils.data import Dataset 

_root_dir = Path.home() / 'workspace/denoising_data/bertie_c_elegans/'

class FromTableFlow(Dataset):
    def __init__(self, 
                 root_dir = _root_dir,
                 src_file = None,
                 cropping_size = 256,
                 is_log_transform = False,
                 scale_int = (0, 255),
                 expand_factor = 1,
                 is_augment = True,
                 is_to_align = False
                 ):
        
        self.cropping_size = cropping_size
        self.is_log_transform = is_log_transform
        self.scale_int = scale_int
        self.expand_factor = expand_factor
        self.is_to_align = is_to_align
        self.is_augment = is_augment
        
        if src_file is None:
            src_file = Path(_root_dir) / 'valid_files.csv'
        
        self.src_df = pd.read_csv(src_file)
        
        
        
        root_dir = str(root_dir)
        root_dir = root_dir if root_dir.endswith('/') else root_dir + os.sep
        
        self.src_df['dirname'] = root_dir + self.src_df['prefix']
        del self.src_df['prefix']
        
        self.n_fields = len(self.src_df)
        
    def __len__(self):
        return self.n_fields*self.expand_factor
    
    def __getitem__(self, ind):
        irow = random.choice(self.src_df.index) #select a set
        row_data = self.src_df.loc[irow]
        
        
        
        fname1 = Path(row_data['dirname']) / '{}.tif'.format(row_data['prev_ind'])
        fname2 = Path(row_data['dirname']) / '{}.tif'.format(row_data['after_ind'])
        
        X = cv2.imread(str(fname1), -1).astype(np.float32)
        Y = cv2.imread(str(fname2), -1).astype(np.float32)
        
        pos_coord = row_data['target_coord_x'], row_data['target_coord_y']
        
        if self.is_augment:
            X, Y = self._augment(X, Y, pos_coord)
        
        if self.is_log_transform:
            X = np.log(X+1)
            Y = np.log(Y+1)
        
        X = (X-self.scale_int[0])/(self.scale_int[1]-self.scale_int[0])
        Y = (Y-self.scale_int[0])/(self.scale_int[1]-self.scale_int[0])
            
        return X[None], Y[None]
    
    def _augment(self, X, Y, pos_coord):
        #randomize if the previous is going to predict the after or the after the previous
        if random.random() < 0.5:
            Y, X = X, Y
        
        w,h = X.shape
        if random.random() < 0.5:
            #random cropping
            ix = random.randint(0, w - self.cropping_size - 1)
            iy = random.randint(0, h - self.cropping_size - 1)
        
        else:
            rr = self.cropping_size
            #random crop along a region of interest
            xlim_l = max(0, pos_coord[1] - rr)
            xlim_r = min(w - self.cropping_size -1, pos_coord[1])
            
            ylim_l = max(0, pos_coord[0] - rr)
            ylim_r = min(h - self.cropping_size -1, pos_coord[0])
            
            ix = random.randint(xlim_l, xlim_r)
            iy = random.randint(ylim_l, ylim_r)
        
            
        
        X = X[ix:ix+self.cropping_size, iy:iy+self.cropping_size]
        Y = Y[ix:ix+self.cropping_size, iy:iy+self.cropping_size]
        
        #horizontal flipping
        if random.random() < 0.5:
            X = X[::-1]
            Y = Y[::-1]
        
        #vertical flipping
        if random.random() < 0.5:
            X = X[:, ::-1]
            Y = Y[:, ::-1]
        return X, Y


if __name__ == '__main__':
    import tqdm
    #src_root_dir = Path.home() / 'workspace/WormData/full_images/'
    #src_root_dir = Path.home() / 'workspace/drosophila_eggs/'
    #src_root_dir = Path.home() / 'workspace/denoising_data/c_elegans/train'
        
    #%%
    gen = FromTableFlow(is_log_transform = False, scale_int = (0, 255))
    #gen = BasicFlow(src_root_dir, is_log_transform = True, scale_int = (0, 16), cropping_size=128)
    #%%
    for kk in tqdm.tqdm(range(10)):
        
        X, Y = gen[kk]
        assert X.shape == (1, gen.cropping_size,gen.cropping_size)
        #%%
        fig, axs = plt.subplots(1,3, sharex=True, sharey=True)
        vmax = max(X.max(), Y.max())
        vmin = min(X.min(), Y.min())
        
        axs[0].imshow(X[0], vmin=0, vmax=1, cmap='gray', interpolation='None')
        axs[1].imshow(Y[0], vmin=0, vmax=1, cmap='gray', interpolation='None')
        
        rr = np.abs(Y[0].astype(np.float32) - X[0])
        axs[2].imshow(rr, interpolation='None')
        for ax in axs:
            ax.axis('off')
        
        #axs[2].imshow((X-Y)[0],  interpolation='None')
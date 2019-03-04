#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 18:05:43 2018

@author: avelinojaver
"""

from pathlib import Path
import cv2
import pandas as pd
import tqdm
import random
import numpy as np
from torch.utils.data import Dataset 

def _get_file_list(root_dir, file_ext = '.tif'):
    root_dir = Path(root_dir)
    all_data = []
    for fname in tqdm.tqdm(root_dir.rglob('*' + file_ext)):
        if fname.name.startswith('.'):
            continue
        
        nn = len(root_dir.parts)
        
        group_name = fname.parts[nn]
        path_r = str(fname.parent)
        bn = fname.name
        
        ss = bn[:-4]
        if '_t' in bn: 
            dd = [x[1:] for x in ss.split('_') if x[0]=='t']
            frame = int(dd[0])
        else:
            frame = int(ss)
        
        all_data.append((bn, frame, path_r, group_name))
    
    df = pd.DataFrame(all_data, columns = ['base_name', 'frame', 'path', 'set_type'])
    df['field_id'] = df['path'].map({k:ii for ii, k in enumerate(df['path'].unique())})
    
    file_lists = []
    for set_type, dat in df.groupby('set_type'):
        set_data = []
        for field_id, filed_dat in dat.groupby('field_id'):
            field_data = []
            for _, row in filed_dat.sort_values(by='frame').iterrows():
                fname = Path(row['path']) / row['base_name']
                field_data.append(fname)
            set_data.append(field_data)
        file_lists.append(set_data)
    return file_lists    
    
class BasicFlow(Dataset):
    def __init__(self, 
                 root_dir,
                 cropping_size = 256,
                 is_log_transform = True,
                 scale_int = (0, 16),
                 expand_factor = 10,
                 is_augment = True,
                 is_to_align = False,
                 max_samples_per_set = None,
                 samples_per_epoch = None
                 ):
        self.cropping_size = cropping_size
        self.is_log_transform = is_log_transform
        self.scale_int = scale_int
        self.expand_factor = expand_factor
        self.is_to_align = is_to_align
        self.is_augment = is_augment
        self.samples_per_epoch = samples_per_epoch
        
        self.file_lists = _get_file_list(root_dir)
        
        if max_samples_per_set is not None:
            assert max_samples_per_set >= 1
            print(f'Randomly selecting {max_samples_per_set} samples per set.')
            new_file_list = []
            for set_data in self.file_lists:
                random.shuffle(set_data)
                ss = set_data[:max_samples_per_set]
                new_file_list.append(ss)
            
            self.file_lists = new_file_list
                
        self.n_fields = len([m for x in self.file_lists for m in x])
        self.frame_step = 1
        
    def __len__(self):
        if self.samples_per_epoch is None:
            return self.n_fields*self.expand_factor
        else:
            return self.samples_per_epoch
    
    def __getitem__(self, ind):
        set_data = random.choice(self.file_lists) #select a set
        field_data = random.choice(set_data) #select a field
        
        N = len(field_data)
        t2 = random.randint(self.frame_step, N-1)
        
        fname1 = field_data[t2-self.frame_step]
        fname2 = field_data[t2]
        
        X = cv2.imread(str(fname1), -1).astype(np.float32)
        Y = cv2.imread(str(fname2), -1).astype(np.float32)
        
        if self.is_augment:
            X, Y = self._augment(X, Y)
        
        if self.is_log_transform:
            X = np.log(X+1)
            Y = np.log(Y+1)
        
        X = (X-self.scale_int[0])/(self.scale_int[1]-self.scale_int[0])
        Y = (Y-self.scale_int[0])/(self.scale_int[1]-self.scale_int[0])
            
        return X[None], Y[None]
    
    def _augment(self, X, Y):
        #randomize if the previous is going to predict the after or the after the previous
        if random.random() < 0.5:
            Y, X = X, Y
        
        #random cropping
        w,h = X.shape
        ix = random.randint(0, w-self.cropping_size)
        iy = random.randint(0, h-self.cropping_size)
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
    
    #src_root_dir = Path.home() / 'workspace/WormData/full_images/'
    #src_root_dir = Path.home() / 'workspace/drosophila_eggs/'
    src_root_dir = Path.home() / 'workspace/denoising/data/c_elegans_divergent/train/'
     
    
    #%%
    gen = BasicFlow(src_root_dir, 
                    is_log_transform = False, 
                    scale_int = (0, 255),
                    max_samples_per_set = None)
    #gen = BasicFlow(src_root_dir, is_log_transform = True, scale_int = (0, 16), cropping_size=128)
    #%%
    for kk in range(10):
        X, Y = gen[kk]
        fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
        vmax = max(X.max(), Y.max())
        vmin = min(X.min(), Y.min())
        
        axs[0].imshow(X[0], vmin=0, vmax=1, cmap='gray', interpolation='None')
        axs[1].imshow(Y[0], vmin=0, vmax=1, cmap='gray', interpolation='None')
        
        
        for ax in axs:
            ax.axis('off')
        
        #axs[2].imshow((X-Y)[0],  interpolation='None')
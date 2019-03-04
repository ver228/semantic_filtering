#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 15:00:47 2019

@author: avelinojaver
"""
import cv2
import numpy as np
import random
import math

from pathlib import Path
from torch.utils.data import Dataset 
#%%
_root_dir = Path.home() / 'workspace/datasets/stanford_campus_dataset/videos/'

class FromMoviesFlow(Dataset):
    def __init__(self, 
                 root_dir = _root_dir,
                 crop_size = 256,
                 epoch_size = 2000,
                 frame_gap = 10,
                 zoom_range = (0.5, 1.5),
                 search_ext = '*.mov',
                 is_random_v_flip = True,
                 is_SSD = True
                 ):
        
        self.crop_size = crop_size
        self.zoom_range = zoom_range
        self.epoch_size = epoch_size
        self.frame_gap = frame_gap
        self.is_random_v_flip = is_random_v_flip
        
        self.root_dir = Path(root_dir)
        
        movie_files = self.root_dir.rglob(search_ext)
        movie_files = [x for x in movie_files if not x.name.startswith('.')] 
        
        
        
        
        
        if is_SSD:
            movie_files_d = {}
            for fname in movie_files:
                prefix = fname.parents[1].name
                if prefix not in movie_files_d:
                    movie_files_d[prefix] = []
                movie_files_d[prefix].append(fname)
                
            test_keys = ['quad']
            test_frac = 0.1
            
            
            test_data = {}
            train_data = {}
            for k in movie_files_d.keys():
                if k in test_keys:
                    test_data[k] = movie_files_d[k]
                else:
                    dd = movie_files_d[k]
                    dd = sorted(dd, key = lambda x : int(x.parent.name[5:]))
                    ind_test = math.ceil(len(dd)*test_frac)
                    
                    test_data[k] = dd[:ind_test]
                    train_data[k] = dd[ind_test:]
            
            self.test_files = test_data
            self.train_files = train_data
        
        else:
            self.train_files = {'all' : movie_files}
            self.test_files = {}
        
        
        self.train()
        
        
    def _sample(self, fname):
        vcap = cv2.VideoCapture(str(fname))
        
        width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        tot_frames = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
    
        ini_frame = random.randint(0, int(tot_frames - self.frame_gap))
        
        
        vcap.set(cv2.CAP_PROP_POS_FRAMES, ini_frame)    
        ret, img1 = vcap.read()
        
        
        vcap.set(cv2.CAP_PROP_POS_FRAMES, ini_frame + self.frame_gap)    
        ret, img2 = vcap.read()
        
        if img1 is None or img2 is None:
            return None
        
        #crop and zoom
        zoom_factor = random.uniform(*self.zoom_range)
        
        z_crop_size = self.crop_size/zoom_factor
        
        z_crop_size = int(min(min(height, width), z_crop_size))
        
        
        xi = random.randint(0, max(0, height - z_crop_size))
        yi = random.randint(0, max(0, width - z_crop_size))
        
        img1_crop = img1[xi:xi+z_crop_size, yi:yi+z_crop_size]
        img2_crop = img2[xi:xi+z_crop_size, yi:yi+z_crop_size]
        
        rr = (self.crop_size, self.crop_size)
        img1_crop = cv2.resize(img1_crop, rr)
        img2_crop = cv2.resize(img2_crop, rr)
        
        #random flips
        if self.is_random_v_flip and (random.random() > 0.5):
            img1_crop = img1_crop[::-1]
            img2_crop = img2_crop[::-1]
        
        if (random.random() > 0.5):
            img1_crop = img1_crop[:, ::-1]
            img2_crop = img2_crop[:, ::-1]
        
        #invert as augmentation
        if random.random() > 0.5:
            img2_crop, img1_crop = img1_crop, img2_crop
    
        return img1_crop, img2_crop
    
    def __len__(self):
        return self.epoch_size
    
    def __getitem__(self, ind):
        for _ in range(10):
            s_key = random.choice(list(self.movie_files.keys()))
            fname = random.choice(self.movie_files[s_key])
            _out = self._sample(fname)
            if _out is not None:
                break
        else:
            raise ValueError('Cannot Read File!!!')
        _out = [np.rollaxis(x, 2, 0) for x in _out]
        _out = [x.astype(np.float32)/255. for x in _out]
        
        return _out
    
    def train(self):
        self.movie_files = self.train_files
    
    def test(self):
        self.movie_files = self.test_files
    
#%%        
if __name__ == '__main__':
    import tqdm
    #root_dir = '/Users/avelinojaver/Downloads/ToulouseCampusSurveillanceDataset/train'
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/survellance_videos/ToulouseCampusSurveillanceDataset/train'
    gen = FromMoviesFlow(root_dir = root_dir,
                         search_ext = '*.mp4',
                         frame_gap = 16,
                         is_random_v_flip = False,
                         is_SSD = False)
    
    #frame_gap = 10
    #crop_size = 256
    #zoom_range = (0.75, 1.5)
    
    for ii, (x,y) in enumerate(gen):#enumerate(tqdm.tqdm(gen)):
        
        y = np.rollaxis(y, 0, 3)
        x = np.rollaxis(x, 0, 3)
        
        fig, axs = plt.subplots(1, 3, sharex= True, sharey=True)
        axs[0].imshow(x)
        axs[1].imshow(y)
        axs[2].imshow(np.abs(x - y).sum(axis=-1))
        
        if ii >= 32:
            break
    
    
    #%%   
#    all_imgs = []
#    for _ in range(5):
#        ret, img = vcap.read()
#        
#        
#        if not ret:
#            raise ValueError("Couldn't read the frame")
#
#        all_imgs.append(img/255.)
##        plt.figure()
##        plt.imshow(img)
#    
#    
#    img_m = np.median(all_imgs, axis=0)
##    all_imgs = np.stack(all_imgs)
    #vcap.set(cv2.CAP_PROP_POS_MSEC, 60166)
    #ret, img2 = vcap.read()
        
    
#    #%%
#    fig, axs = plt.subplots(1,3, sharex= True, sharey=True)
#    axs[0].imshow(np.abs(all_imgs[0] - img_m))
#    axs[1].imshow(img_m)
#    axs[2].imshow(all_imgs[0])
    
    
    
    
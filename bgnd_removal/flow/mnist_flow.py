#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 15:48:22 2018

@author: avelinojaver
"""
from pathlib import Path
import struct
import random

import numpy as np
import cv2
from torch.utils.data import Dataset



_root_dir = Path.home() / 'workspace/datasets/'

mnistfashion_params_v1 = dict(
        fg_classes = [8],
         fg_n_range = (0, 5),
         bg_n_range = (1, 25),
         int_range = (0.5, 1.1),
         is_fix_bg = False,
         is_clean_output = False,
         epoch_size = 10016,
         root_dir = None, 
         output_size = 256,
         is_h_flip = True,
         is_v_flip = True,
         max_rotation = 90,
         max_overlap = 0.25
        )

class MNISTFashionFlow(Dataset):
    label_names = {
            0:'T-shirt/top',
            1:'Trouser',
            2:'Pullover',
            3:'Dress',
            4:'Coat',
            5:'Sandal',
            6:'Shirt',
            7:'Sneaker',
            8:'Bag',
            9:'Ankle boot'
            }
    
    def __init__(self, 
                 fg_classes = [8],
                 fg_n_range = (0, 5),
                 bg_n_range = (1, 25),
                 int_range = (0.5, 1.1),
                 is_fix_bg = False,
                 is_clean_output = False,
                 is_separated_output = False,
                 epoch_size = 10016,
                 root_dir = None, 
                 output_size = 256,
                 is_h_flip = True,
                 is_v_flip = True,
                 max_rotation = 90,
                 max_overlap = 0.25
                 ):
        
        if root_dir is None:
            root_dir = _root_dir / 'MNIST_fashion'
        
        self.root_dir = root_dir 
        
        self.train_img_file = root_dir / 'raw' / 'train-images-idx3-ubyte'
        self.train_lbl_file = root_dir / 'raw' / 'train-labels-idx1-ubyte'
        self.test_img_file = root_dir / 'raw' / 't10k-images-idx3-ubyte'
        self.test_lbl_file = root_dir / 'raw' / 't10k-labels-idx1-ubyte'
        
        self.train_insets = self._read_insets(self.train_img_file, self.train_lbl_file)
        self.test_insets = self._read_insets(self.test_img_file, self.test_lbl_file)
        
        
        #set sizes ranges
        self.out_size_x, self.out_size_y  = output_size, output_size
        
        #labels dist
        self.epoch_size = epoch_size
        self.fg_classes = fg_classes
        self.bg_classes = list(set(self.label_names.keys()) - set(self.fg_classes))
        
        self.fg_n_range = fg_n_range
        self.bg_n_range = bg_n_range
        self.int_range = int_range
        self.is_h_flip = is_h_flip
        self.is_v_flip = is_v_flip
        self.max_rotation = max_rotation
        self.max_overlap = max_overlap
        
        self.is_fix_bg = is_fix_bg
        self.is_clean_output = is_clean_output
        self.is_separated_output = is_separated_output
        
        self.train()
        
    def _read_insets(self, fname_images, fname_labels):
        with open(str(fname_images), 'rb') as fid:
            zero, data_type, dims = struct.unpack('>HBB', fid.read(4))
            shape = tuple(struct.unpack('>I', fid.read(4))[0] for d in range(dims))
            images = np.frombuffer(fid.read(), dtype=np.uint8).reshape(shape)
        
        with open(str(fname_labels), 'rb') as fid:
            magic, num = struct.unpack(">II", fid.read(8))
            labels = np.frombuffer(fid.read(), dtype=np.int8)
        
        inset_classes = {}
        for ii in range(10):
            inset_classes[ii] = images[labels==ii].copy()
        
        return inset_classes
        
    def _create_image(self, valid_classes, n_range, valid_mask = None):
        n_rois = random.randint(*n_range)
        
        
        
        img_o = np.zeros((self.out_size_x, self.out_size_y), np.float32)
        BREAK_NUM = 20 #number of trials to add an image before giving up
        
        
        
        for _ in range(n_rois):
            lab_i = random.choice(valid_classes)
            roi = random.choice(self.inset_classes[lab_i])/225.
            
            roi = self._augment(roi)
            
            roi_bw = roi>0
            for _ in range(BREAK_NUM):
            
            
                xr = random.randint(0, self.out_size_x - roi.shape[1])
                yr = random.randint(0, self.out_size_y - roi.shape[1])
                
                prev = img_o[xr:xr + roi.shape[1], yr:yr + roi.shape[0]]
                
                test_roi = (prev>0)
                if valid_mask is not None:
                    test_roi |= valid_mask[xr:xr + roi.shape[1], yr:yr + roi.shape[0]]
                
                
                pix2test = test_roi[roi_bw]
                if np.mean(pix2test) < self.max_overlap:
                    prev[:] += roi
                    break
                    
        return img_o

    def _augment(self, img):
        theta = random.uniform(-self.max_rotation, self.max_rotation)
        zoom = random.uniform(1., 1.5)
        
        is_v_flip = self.is_v_flip & (random.random() > 0.5)
        is_h_flip = self.is_h_flip & (random.random() > 0.5)
        
        int_factor = random.uniform(*self.int_range)
        
        if is_v_flip:
            img = img[::-1]
        
        if is_h_flip:
            img = img[:, ::-1]
        
        
        rows,cols = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
        img = cv2.warpAffine(img, M, (cols,rows))
        
        img = cv2.resize(img, (0,0), fx=zoom, fy=zoom) 
        
        img *= int_factor
        
        return img
    
    def __len__(self):
        return self.epoch_size
    
    def __getitem__(self, ind):
        
        if self.is_separated_output:
            img_fg1 = self._create_image(self.fg_classes, self.fg_n_range)
            img_bg1 = self._create_image(self.bg_classes, self.bg_n_range, img_fg1>0)    
            X = img_bg1 + img_fg1
            X = X[None]
            Y = np.stack((img_fg1, img_bg1))
            
            #the number of output channels change...
            return X, Y    
            
        
        if self.is_clean_output:
            img_fg1 = self._create_image(self.fg_classes, self.fg_n_range)
            img_bg1 = self._create_image(self.bg_classes, self.bg_n_range, img_fg1>0)    
            X, Y = img_bg1 + img_fg1, img_fg1
            
        elif self.is_fix_bg:
            img_fg1 = self._create_image(self.fg_classes, self.fg_n_range)    
            img_fg2 = self._create_image(self.fg_classes, self.fg_n_range)
            
            valid_mask = (img_fg1>0) | (img_fg2>0)
            img_bg1 = self._create_image(self.bg_classes, self.bg_n_range, valid_mask)
            
            X = img_bg1 + img_fg1
            Y = img_bg1 + img_fg2
        else:
           
            img_bg1 = self._create_image(self.bg_classes, self.bg_n_range)
            img_bg2 = self._create_image(self.bg_classes, self.bg_n_range)
            
            valid_mask = (img_bg1>0) | (img_bg2>0)
            img_fg1 = self._create_image(self.fg_classes, self.fg_n_range, valid_mask)
            
            X = img_bg1 + img_fg1
            Y = img_bg2 + img_fg1
        
        return X[None], Y[None]
    
    def train(self):
        self.inset_classes = self.train_insets
    
    def test(self):
        self.inset_classes = self.test_insets




#%%
if __name__ == '__main__':
    
    import matplotlib.pylab as plt
    gen = MNISTFashionFlow(is_fix_bg = False,
                           is_clean_output = True,
                           is_separated_output = False,
                           output_size = 256,
                         epoch_size = 10,
                         fg_n_range = (1, 5),
                         bg_n_range = (5, 25),
                         int_range = (0.5, 1.1),
                         max_rotation = 45,
                         is_v_flip = False
                         )
    
    for ii, (X, Y) in enumerate(gen):
        bb = X[0] -Y[0]
        dd = X[0] - np.clip(bb, 0, bb.max())
        
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(X[0], cmap='gray')
        axs[1].imshow(Y[0], cmap='gray')
        axs[2].imshow(dd, cmap='gray')
        
        if ii > 0:
            break
        #%%
        
        
        
        
        
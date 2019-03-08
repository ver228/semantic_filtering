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

def _read_insets(fname_images, fname_labels):
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


def generate_bgnd(bgnd_size = (512, 512), n_cicles = 40, max_intensity = 64):
    
    max_circ_size = bgnd_size[0]//2
    
    bgnd = np.zeros(bgnd_size,  np.float32)
    for _ in range(n_cicles):
        x = random.randint(0, bgnd_size[0]-1)
        y = random.randint(0, bgnd_size[1]-1)
        r = random.randint(5, max_circ_size)
        val = random.randint(1, max_intensity)
        cv2.circle(bgnd, (x,y), r, val, -1)
    
    sigma_ = random.uniform(0, max_intensity//2)
    noise = np.random.normal(0, sigma_, bgnd.shape)
    bgnd = cv2.blur(bgnd, (11,11)) + noise
    bgnd = np.clip(bgnd, 0, 255).astype(np.uint8)
    return bgnd
#%%
def process_data(_insets, foreground_label, set_type):
    for k, data in _insets.items():
        save_dir = save_dir_root / set_type
        if k == foreground_label:
            save_dir = save_dir / 'foreground'
        else:
            save_dir = save_dir / 'background_crops'
        save_dir.mkdir(exist_ok=True, parents=True)
        
        
        data_r = (3.*data/4.).astype(np.uint8)
        for ii, img in enumerate(data_r):
            save_name = save_dir / f'{k}_{ii}.png'
            cv2.imwrite(str(save_name), img)
    
    save_dir = save_dir_root / set_type / 'background'
    save_dir.mkdir(exist_ok=True, parents=True)
    for ii in range(500):
        bgnd = generate_bgnd()
        save_name = save_dir / f'B_{ii}.png'
        cv2.imwrite(str(save_name), bgnd)
#%%
if __name__ == '__main__':
    root_dir = Path.home() / 'workspace/datasets/MNIST_fashion'
    
    save_dir_root = Path.home() / 'workspace/denoising/data/MNIST_fashion'
    #save_dir_root = Path.home() / 'Desktop/MNIST_fashion'
    
    train_img_file = root_dir / 'raw' / 'train-images-idx3-ubyte'
    train_lbl_file = root_dir / 'raw' / 'train-labels-idx1-ubyte'
    test_img_file = root_dir / 'raw' / 't10k-images-idx3-ubyte'
    test_lbl_file = root_dir / 'raw' / 't10k-labels-idx1-ubyte'

    train_insets = _read_insets(train_img_file, train_lbl_file)
    test_insets = _read_insets(test_img_file, test_lbl_file)
    
    process_data(train_insets, foreground_label = 9, set_type = 'train')
    process_data(test_insets, foreground_label = 9, set_type = 'test')
    
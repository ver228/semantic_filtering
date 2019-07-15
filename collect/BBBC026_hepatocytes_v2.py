#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:25:34 2019

@author: avelinojaver
"""
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pylab as plt

def get_valid_labels(peaks, _labels):
        cm = cv2.connectedComponentsWithStats(peaks.astype(np.uint8))[-1][1:].astype(np.int)
        valid_labels = _labels[cm[..., 1], cm[..., 0]]
        valid_labels = valid_labels[valid_labels != 0]
        return valid_labels
    
    
def _get_mask(valid_labels, _labels, _stats, _img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    _bad_mask = np.zeros_like(_img, np.uint8)
    for lab in valid_labels:
        left, top, width, height, area = _stats[lab]
        
        
        i1, i2 = top, top + height  
        j1, j2 = left, left + width 
        
        roi_lab = _labels[i1:i2, j1:j2] == lab
        _bad_mask[i1:i2, j1:j2][roi_lab] = 1
    
    
    _bad_mask = cv2.dilate(_bad_mask, kernel, iterations = 3)
    
    return _bad_mask

def get_crops(labs2crop, _labels, _stats, _img):
    crops2save = []
    for lab in labs2crop:
        left, top, width, height, area = _stats[lab]
        i1, i2 = top, top + height  
        j1, j2 = left, left + width 
        
        if (i1 <= 0) or (j1 <= 0) or (i2 >= _labels.shape[0] - 1) or (j2 >= _labels.shape[1] - 1):
            continue
        
        roi_bad = _labels[i1:i2, j1:j2] != lab
        
        #roi_bad = cv2.erode(roi_bad.astype(np.uint8), kernel, iterations = 3)
        
        crop_img = _img[i1:i2, j1:j2].copy()
        crop_img[roi_bad==1] = 0
        crops2save.append(crop_img)
    return crops2save

def divide_image(fname):
    #%%
    img = cv2.imread(str(fname), -1)
    img_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
    
    
    bad = (img[..., 0] == 255) & (img[..., 1] <= 10) & (img[..., 2] <= 10)
    fib =  (img[..., 0] <= 10) & (img[..., 1] == 255) & (img[..., 2] <= 10)
    hep =  (img[..., 0] <= 10) & (img[..., 1] <= 10) & (img[..., 2] == 255)
    
    
    peaks2remove = bad | fib | hep
    med = cv2.medianBlur(img_g, ksize= 11) + np.random.normal(0, 2, img_g.shape).round().astype(np.int)
    img_g[peaks2remove] = med[peaks2remove]
    
    
    img_l = np.log(img_g.astype(np.float32)+1)
    bot, top = img_l.min(), img_l.max()
    img_n = (img_l - bot)/(top-bot)*255
    img_n = img_n.astype(np.uint8)
    
    blur = cv2.GaussianBlur(img_n,(11,11),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel, iterations = 3)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    nlabs, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
    
    fib_labels = get_valid_labels(fib, labels)
    hep_labels = get_valid_labels(hep, labels)
    bad_labels = get_valid_labels(bad, labels)
    
    fib_mask = _get_mask(fib_labels, labels, stats, img_g)
    hep_mask = _get_mask(hep_labels, labels, stats, img_g)
    
    bad_mask = _get_mask(bad_labels, labels, stats, img_g)
    
    bgnd_mask = (mask > 0) & ((bad_mask == 0) | (fib_mask==1) | (hep_mask ==1))
    #bgnd_mask = (fib_mask==1) | (hep_mask ==1)
    
    #fig, axs = plt.subplots(1,2)
    #axs[0].imshow(mask)
    #axs[1].imshow(bgnd_mask)
    
    
    bgnd = img_g.copy()
    bgnd[bgnd_mask] = 0
    
    
    
    fib_indeces = set(fib_labels) - set(hep_labels) - set(bad_labels)
    hep_indeces = set(hep_labels) - set(fib_labels) - set(bad_labels)
    
    fib_crops = get_crops(fib_indeces, labels, stats, img_g)
    heb_crops = get_crops(hep_indeces, labels, stats, img_g)
    
    
    #(no_hep_mask==0) | (no_hep_mask==0)
    
    return fib_crops, heb_crops, bgnd


def save_data_crops(save_dir, fgnd_crops, base_name):
    
    
    save_dir.mkdir(parents=True, exist_ok=True)
    for ii, crop in enumerate(fgnd_crops):
        f_name = save_dir / (f'{base_name}_cell{ii}.png')
        cv2.imwrite(str(f_name), crop)



if __name__ == '__main__':
    #save_dir_root = Path.home() / 'workspace/denoising/data/BBBC026_v2'
    save_dir_root = Path.home() / 'Desktop/BBBC026_all'
    
    #data_dir = Path('/Users/avelinojaver/Downloads/BBBC026_GT_images/')
    data_dir = Path.home() / 'workspace/datasets/BBBC/BBBC026/BBBC026_GT_images'
    
    
    heb_save_dir_root = save_dir_root / 'hepatocytes'
    fib_save_dir_root = save_dir_root / 'fibroblasts'
    bgnd_save_dir_root = save_dir_root / 'background'
    
    fnames = data_dir.glob('*.png')
    
    fnames = sorted(list(fnames), key = lambda x : x.name)
    
    #fnames_train = fnames[:-1]
    #fname_test = fnames[-1]
    
    for fname in fnames:
        
        fib_crops, heb_crops, bgnd = divide_image(fname)
        
        base_name = fname.stem
        b_fname = bgnd_save_dir_root / (base_name + '_bgnd.png')
        b_fname.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(b_fname), bgnd)
        
        save_data_crops(fib_save_dir_root, fib_crops, base_name)
        save_data_crops(heb_save_dir_root, heb_crops, base_name)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
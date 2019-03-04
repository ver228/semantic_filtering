#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 17:33:13 2018

@author: avelinojaver
"""
import cv2
from skimage import measure, morphology
import pandas as pd
from pathlib import Path
import tqdm
import numpy as np

from skimage.morphology import skeletonize, reconstruction
#%%
def segment_microglia(img_l, seed_threshold = 5, debug=False):
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    _, seed = cv2.threshold(img_l, seed_threshold, 255, cv2.THRESH_BINARY)
    seed = cv2.erode(seed, kernel, iterations = 3)
    
    bot = img_l.min()
    top = img_l.max()
    img_n = (img_l-bot)/(top-bot)*255
    img_n = img_n.astype(np.uint8)
    
    for _ in range(8):
        img_n = cv2.medianBlur(img_n, 3)
    
    mask_o = cv2.adaptiveThreshold(img_n, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                    cv2.THRESH_BINARY, 25, -3)
    
    
    mask_d = cv2.dilate(mask_o, kernel, iterations = 9)
    mask_d = cv2.erode(mask_d, kernel, iterations = 5)
    
    mask_skel = skeletonize(mask_d>0)
    mask_skel = cv2.dilate(mask_skel.astype(np.uint8)*255, kernel, iterations = 3)
    
    
    mask = mask_skel | mask_o>0
    mask[seed>0] = 255
    
    mask = reconstruction(seed>0, mask>0, method='dilation')
    
    if debug == True:
        import matplotlib.pylab as plt
        fig, axs = plt.subplots(1,4, figsize = (20, 5), sharex=True, sharey=True)
        axs[0].imshow(img_n)
        axs[1].imshow(mask_o)
        axs[2].imshow(mask_skel)
        axs[3].imshow(mask)
        
        
    
    return mask

#%%
def extract_intensity_stats(info_df, root_dir, new_path = None):
    dat = []
    for irow, row in tqdm.tqdm(info_df.iterrows(), total = len(info_df)):
        if new_path is not None:
            path_r = '/'.join(['stills_cleaned_20x'] + row['path'].split('/')[1:])
        else:
            path_r = row['path']
        
        fname = root_dir / path_r / row['base_name']
        
        img = cv2.imread(str(fname),-1).astype(np.float32)
        img = np.log(img + 1)
        
        
        ss = np.percentile(img, [1, 50, 99]).tolist()
        
        mad = np.median(np.abs(img-ss[0]))
        
        ss += [mad, np.mean(img)] 
        
        
        ll = [irow, ss]
        
        dat.append(ll)
    inds, vals = zip(*dat)
    df_avg = pd.DataFrame(np.array(vals), 
                          index = inds, 
                          columns = ['Iq01', 'Iq50', 'Iq99', 'Imad', 'Imean']
                          )
    df_new =  info_df.join(df_avg)
    return df_new
#%%
def _get_cell_rois(img, 
                   img_th, 
                   possible_cells_borders, 
                   cell_border_overlap, 
                   area_th = (5e3, 8e4)):
    
    se = morphology.selem.disk(3)
    mask = (img>img_th) #| possible_cells_borders
    mask = cv2.medianBlur(mask.astype(np.uint8), ksize=5)
    
    mask = cv2.dilate(mask, se, iterations=5)
    mask = cv2.erode(mask, se, iterations=1)
    mask = mask > 0
    mask = morphology.remove_small_objects(mask, 5000)
    mask = morphology.remove_small_holes(mask, 5000)
   
#    fig, axs = plt.subplots(1,3,sharex=True, sharey=True)
#    axs[0].imshow(img)
#    axs[1].imshow(mask)
#    axs[2].imshow(possible_cells_borders)
    
    L = morphology.label(mask, connectivity = 1)
    props = measure.regionprops(L, img, cache=True)
    
    valid_cells = []
    #clean_cells = np.zeros_like(mask)
    for p in props:
        bb = p.bbox
        pos_overlap = possible_cells_borders[bb[0]:bb[2], bb[1]:bb[3]][p.image].mean()
        
        is_good_area = (p.area > area_th[0]) & (p.area < area_th[1])
        
        if is_good_area and pos_overlap > cell_border_overlap:
            
            #this a more aggresive segmentation... i am not sure it will work better...
            roi = img[bb[0]:bb[2], bb[1]:bb[3]]
            
            
            refined_mask = p.image & (segment_microglia(roi) > 0)
            
            
            valid_cells.append((p.bbox, p.image, refined_mask, pos_overlap))
    #plt.figure(); plt.imshow(clean_cells)
    
    return valid_cells
#%%

def _get_bgnd_mask(img, bg_th, valid_cells, img_is_blurry, min_area_nobg = 1e4, bg_blurry_th = 5e-4):
    
    mask_bgnd = img<bg_th
    mask_bgnd = morphology.remove_small_holes(mask_bgnd)
    se = morphology.selem.disk(3)
    mask_bgnd = cv2.erode(mask_bgnd.astype(np.uint8), se, iterations=5)
    
    #do consider parts as background if they are too small or seem blurry even if the intensity is large
    L = morphology.label(~mask_bgnd, connectivity = 1)
    props = measure.regionprops(L, img, cache=True)
    for p in props:
        bb, roi = p.bbox, p.image
        blurry_ind = img_is_blurry[bb[0]:bb[2], bb[1]:bb[3]][roi].var()
        if  p.area < min_area_nobg or blurry_ind < bg_blurry_th:
            mask_bgnd[bb[0]:bb[2], bb[1]:bb[3]][roi] = 1
    
    #make sure the regions selected as valid cells are not background
    for bb, roi, _ in valid_cells:
        mask_bgnd[bb[0]:bb[2], bb[1]:bb[3]][roi] = 0
    
    mask_bgnd = cv2.erode(mask_bgnd, se, iterations=5)
    
    return mask_bgnd>0
#%%
def get_cells_bg(img, img_ori,  
                 cell_border_fg_th = 0.05, 
                 cell_border_bg_th = 0.05, 
                 cell_border_overlap = 0.8,
                 debug=False):
    
    #I calculate the laplacian as a rough estimate if a blob is a cell in or out of focus
    #the median blur is because the n2n cleaned image have 
    #specs since the training (2018.08.20) and testing (180822) conditions are different (lower laser excitation)
    #I am using 
    
    #cell_border_th = 0.05
    img_median = img.copy()
    img_median[1021] = img_median[1022]
    for _ in range(16):
        img_median = cv2.medianBlur(img_median, 5)
    img_laplacian = cv2.Laplacian(img_median, cv2.CV_32F)
    
    
    se = morphology.selem.disk(3)
    
    #possible_cells_borders = cv2.dilate((img_is_blurry>cell_border_th).astype(np.uint8), se, iterations=3)
    #possible_cells_borders = np.abs(cv2.GaussianBlur(img_laplacian,(5,5),0)) > cell_border_th
    #possible_cells_borders = morphology.remove_small_objects(possible_cells_borders, 500)
    
    img_is_blurry = img_laplacian>cell_border_fg_th
    possible_cells_borders = img_is_blurry.astype(np.uint8)
    possible_cells_borders =  cv2.morphologyEx(possible_cells_borders, cv2.MORPH_CLOSE, se, iterations=5)
    possible_cells_borders = morphology.remove_small_objects(possible_cells_borders>0, 1e3)
    possible_cells_borders = cv2.dilate(possible_cells_borders.astype(np.uint8), se, iterations=5)
    valid_cells_r = _get_cell_rois(img, th, possible_cells_borders, cell_border_overlap)
    
    
    img_is_blurry = img_laplacian>cell_border_bg_th
    possible_cells_borders = img_is_blurry.astype(np.uint8)
    possible_cells_borders =  cv2.morphologyEx(possible_cells_borders, cv2.MORPH_CLOSE, se, iterations=5)
    possible_cells_borders = morphology.remove_small_objects(possible_cells_borders>0, 1e3)
    possible_cells_borders = cv2.dilate(possible_cells_borders.astype(np.uint8), se, iterations=10)
    mask_bgnd = possible_cells_borders <= 0
    
    
    valid_cells = []
    for dat in valid_cells_r:
        bb = dat[0]
        if (bb[0] > 0) & (bb[1] > 0) & (bb[2]< img.shape[0]-1) & (bb[3]< img.shape[1]-1):
            valid_cells.append(dat)
    
    if debug:
        
        
        
        img_bg = img_ori.copy()
        noise = np.random.normal(bg_med, bg_mad, img_bg.shape)
        mm = ~mask_bgnd
        img_bg[mm] = noise[mm]
        
        img_cells = np.random.normal(bg_med, bg_mad,  img.shape)
        img_cells_score = np.zeros_like(img)
        for bb, roi, cell_score in valid_cells:
            img_cells[bb[0]:bb[2], bb[1]:bb[3]][roi] = img_ori[bb[0]:bb[2], bb[1]:bb[3]][roi]
            img_cells_score[bb[0]:bb[2], bb[1]:bb[3]][roi] = cell_score
        
        fig, axs = plt.subplots(2,3, figsize = (20, 20), sharex=True, sharey=True)
        axs[0][0].imshow(img)
        axs[0][1].imshow(img_cells)
        axs[0][2].imshow(img_bg)
        
        axs[1][0].imshow(img_ori)
        axs[1][1].imshow(img_cells_score)
        axs[1][2].imshow(mask_bgnd)
    
    return mask_bgnd, valid_cells
#%%
_debug = False
if __name__ == '__main__':
    valid_rows = ['B', 'D', 'F', 'H', 'J', 'L', 'N']
     
    #src_root_dir = Path.home() / 'OneDrive - Nexus365/microglia/data'
    #save_root_dir = Path.home() / 'OneDrive - Nexus365/microglia/data/cell_bgnd_divided'
    
    src_root_dir = Path.home() / 'workspace/Microglia/data'
    save_root_dir = Path.home() / 'workspace/denoising/data/microglia/cell_bgnd_divided'

    
    csv_file = src_root_dir / 'stills.csv'
    df = pd.read_csv(csv_file)
    
    df.loc[df['frame'].isnull(), 'frame'] = 1
    df = df[df['magnification'] == '20x']
    
    df = df[df['plate_row'].isin(valid_rows)]
    
    #df = df[df['date']=='2018.08.20']; mad_factor = 6; cell_border_fg_th = 0.12; cell_border_overlap = 0.9
    df = df[(df['date']=='180822') & ~df['path'].str.contains('_ZTest_')]; mad_factor = 2; cell_border_fg_th = 0.05; cell_border_overlap = 0.8
    
    df_int_clean = extract_intensity_stats(df, src_root_dir, new_path='stills_cleaned_20x')
    
    th = np.median(df_int_clean['Iq50'] + mad_factor*df_int_clean['Imad'])
    
    bg_med = df_int_clean['Iq50'].median()
    bg_mad = df_int_clean['Imad'].median()
    #%%
    for pos_ind, pos_df in tqdm.tqdm(df.groupby(['plate_col', 'plate_row',  'fld', 'z'])):
        pos_df = pos_df.sort_values(by='frame')
        
        pos_dat = []
        for _, row in pos_df.iterrows():
            fname = src_root_dir / row['path'] / row['base_name']
            img_ori = cv2.imread(str(fname),-1).astype(np.float32)
            img_ori = np.log(img_ori + 1)
            
            path_r = '/'.join(['stills_cleaned_20x'] + row['path'].split('/')[1:])
            fname = src_root_dir / path_r / row['base_name']
            
            img = cv2.imread(str(fname),-1).astype(np.float32)
            img = np.log(img + 1)
            
            pos_dat.append((row, img, img_ori))
        
        mask_bgnd, valid_cells = get_cells_bg(pos_dat[0][1], 
                                              pos_dat[0][2],
                                              cell_border_fg_th = cell_border_fg_th,
                                              cell_border_overlap = cell_border_overlap,
                                              debug = _debug)
        
#        if len(plt.get_fignums()) > 30:
#            break
        
        for row, img, img_ori in pos_dat:
            path_r_v1 = '/'.join(['cell_images'] + row['path'].split('/')[1:])
            path_r_v2 = '/'.join(['cell_images_dilated'] + row['path'].split('/')[1:])
            
            dd = (row['plate_row'], row['plate_col'], int(row['fld']), int(row['z']))
            save_prefix = '{}-{}_fld{}_z{}'.format(*dd)
            time_str = 't{}'.format(int(row['frame']))
            
            
            #% extract and save images
            for i_cell, (bb, mask_ori, mask_refined, _) in enumerate(valid_cells):
                for path_r, roi in ((path_r_v1, mask_refined), (path_r_v2, mask_ori)):
                
                    #cell_img = np.random.normal(bg_med, bg_mad,  roi.shape)
                    cell_img = np.zeros(roi.shape)
                    cell_img[roi] = img_ori[bb[0]:bb[2], bb[1]:bb[3]][roi]
                    cell_img_i = np.round(np.exp(cell_img)-1).astype(np.uint16)
                    
                    bn = 'cell{}_{}'.format(i_cell + 1, save_prefix)
                    ff = save_root_dir / path_r / bn / (time_str + '.tif')
                    ff.parent.mkdir(exist_ok=True, parents=True)
                    cv2.imwrite(str(ff), cell_img_i)
            
            
            
            #% save bgnd 
            img_bg = img_ori.copy()
            
            #noise = np.random.normal(bg_med, bg_mad, img_bg.shape)
            mm = ~mask_bgnd
            img_bg[mm] = 0#noise[mm]
            
            
            img_bg_i = np.round(np.exp(img_bg)-1).astype(np.uint16)
            
            
            dd = (row['plate_row'], row['plate_col'], int(row['fld']), int(row['z']), int(row['frame']))
            save_prefix = save_prefix + '_' + time_str
            
            path_r = '/'.join(['bgnd_images'] + row['path'].split('/')[1:])
            ff = save_root_dir / path_r / '{}.tif'.format(save_prefix)
            ff.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(ff), img_bg_i)
            
        
        
        

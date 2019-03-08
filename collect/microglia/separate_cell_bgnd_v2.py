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

from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize, reconstruction



#%%
def process_file(fname, min_area = 1e4, cell_border_fg_th = 0.12):
    
    img_ori = cv2.imread(str(fname),-1)
    img = img_ori.astype(np.float32)
    
    img[1021] = img[1022]
    for _ in range(16):
        img = cv2.medianBlur(img, 5)
    
    img = np.log(img + 1)
    img_laplacian = cv2.Laplacian(img, cv2.CV_32F)
    
    bot, top = img.min(), img.max()
    img_n = (img-bot)/(top-bot)*255
    img_n = img_n.astype(np.uint8)
    
    img_n[1021] = img_n[1022]
    
    th_bg = -3
    mask_bg = cv2.adaptiveThreshold(img_n, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY, 151, th_bg)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask_bg = cv2.dilate(mask_bg, kernel, iterations=10)
    _, cnts, _ = cv2.findContours(mask_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_cnts = [x for x in cnts if cv2.contourArea(x) > min_area]
    bgnd = img_ori.copy()
    cv2.drawContours(bgnd, valid_cnts, -1, 0, -1)
    
    img_is_blurry = img_laplacian>cell_border_fg_th
    possible_cells_borders = img_is_blurry.astype(np.uint8)
    possible_cells_borders =  cv2.morphologyEx(possible_cells_borders, cv2.MORPH_CLOSE, kernel, iterations=5)
    possible_cells_borders = morphology.remove_small_objects(possible_cells_borders>0, 1e3)
    possible_cells_borders = cv2.dilate(possible_cells_borders.astype(np.uint8), kernel, iterations=5)
    
    possible_cells_borders = np.abs(img_laplacian)
    
    rois_data = []
    for cnt in valid_cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        
        cnt_n = cnt.copy()
        cnt_n[..., 0] -= x
        cnt_n[..., 1] -= y
        
        bw = np.zeros((h,w), np.uint8)
        cv2.drawContours(bw,[cnt_n] , -1, 255, -1)
        
        
        
        roi = img_ori[y:y+h, x:x+w].copy()
        roi[bw == 0] = 0
        
        
        edge_score = possible_cells_borders[y:y+h, x:x+w][bw>0].mean()
        rois_data.append((roi, edge_score))
    
    
    if _debug:
        #mask_fg = np.zeros(img_ori.shape, np.uint8)
        #cv2.drawContours(mask_fg, valid_cnts, -1, 255, -1)
        
        fgnd = img_ori.copy()
        fgnd[bgnd>0] = 0
        
        import matplotlib.pylab as plt
        fig, axs = plt.subplots(1, 4, sharex=True, sharey=True, figsize= (15, 5))
        axs[0].imshow(img_n)
        axs[1].imshow(np.log(fgnd+1))
        axs[2].imshow(np.log(bgnd+1))
        axs[3].imshow(possible_cells_borders)
    
    if rois_data:
        rois, edge_avgs = zip(*rois_data)
    else:
        rois, edge_avgs = [], []
    
    return bgnd, rois, edge_avgs


#%%
def _normalize(roi):
    roi_n = roi.astype(np.float32)
    mm = roi_n>0
    roi_n[mm] -= roi_n[mm].min()
    roi_n = np.log(roi_n + 1)
    roi_n *= 255/roi_n.max()
    roi_n = roi_n.astype(np.uint8)
    return roi_n


if __name__ == '__main__':
    _debug = False
    _tight_mask = False
    
    valid_rows = ['B', 'D', 'F', 'H', 'J', 'L', 'N']
     
    src_root_dir = Path.home() / 'OneDrive - Nexus365/microglia/data'
    save_root_dir = Path.home() / 'Desktop/microglia/'
    
    #src_root_dir = Path.home() / 'workspace/Microglia/data'
    #save_root_dir = Path.home() / 'workspace/denoising/data/microglia_v2'

    if _tight_mask:
        save_root_dir = save_root_dir.parent / (save_root_dir.name + '_tight')

    save_ext = 'tif'    
    #save_ext = 'npy'
    
    if _debug:
        save_root_dir = save_root_dir / 'debug'
    
    csv_file = src_root_dir / 'stills.csv'
    df = pd.read_csv(csv_file)
    
    df.loc[df['frame'].isnull(), 'frame'] = 1
    df = df[df['magnification'] == '20x']
    df = df[df['plate_row'].isin(valid_rows)]
    
    #use only one slide and one frame per field of view
    df = df[(df['z'] == 1) & (df['frame'] == 1)]
    
    
    #df = df[df['date']=='2018.08.20'];  fg_edge_th = 0.07; bg_edge_th = 0.02
    df = df[(df['date']=='180822') & ~df['path'].str.contains('_ZTest_')]; fg_edge_th = 0.045; bg_edge_th = 0.01
    #%%
    
    all_scores_avgs = []
    df_g  = df.groupby(['plate_col', 'plate_row',  'fld', 'z'])
    for ii, (pos_ind, pos_df) in enumerate(tqdm.tqdm(df_g)):
        pos_df = pos_df.sort_values(by='frame')
    
    
        pos_dat = []
        for _, row in pos_df.iterrows():
            
            fname = src_root_dir / row['path'] / row['base_name']
        
            bgnd, rois, edge_scores = process_file(fname)
            all_scores_avgs += list(edge_scores)
            
            save_dir_fg_crops = save_root_dir / ('/'.join(['foreground'] + row['path'].split('/')[1:]))
            save_dir_bg_crops = save_root_dir / ('/'.join(['background_crops'] + row['path'].split('/')[1:]))
            save_dir_bg = save_root_dir / ('/'.join(['background'] + row['path'].split('/')[1:]))
    
            
            dd = (row['plate_row'], 
                  row['plate_col'], 
                  int(row['fld']), 
                  int(row['z']),
                  int(row['frame'])
                  )
            
            save_prefix = '{}-{}_fld{}_z{}_t{}'.format(*dd)
            ff = save_dir_bg / f'{save_prefix}.{save_ext}'
            ff.parent.mkdir(exist_ok=True, parents=True)
            
            if _debug:
                bgnd2save = _normalize(bgnd)
            else:
                bgnd2save = bgnd
            
            if save_ext == 'npy':
                np.save(str(ff), bgnd2save)
            else:
                cv2.imwrite(str(ff), bgnd2save)
            
            for i_cell, (roi, edge_score) in enumerate(zip(rois, edge_scores)):
                if edge_score >= fg_edge_th:
                    save_dir = save_dir_fg_crops
                    
                    if _tight_mask:
                    
                        th = threshold_otsu(roi[roi>0])*0.3
                        bw = (roi>th).astype(np.uint8)
                        
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
                        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=5)
                        
                        n_labs, labels, stats, centroids = cv2.connectedComponentsWithStats(bw)
                        
                        ind = np.argmax(stats[1:, -1]) + 1
                        roi[labels != ind] = 0
                    
                elif edge_score <= bg_edge_th:
                    save_dir = save_dir_bg_crops
                else:
                    continue
                
                fname = 'cell{}_{}.{}'.format(i_cell + 1, save_prefix, save_ext)
                ff = save_dir /  fname
                ff.parent.mkdir(exist_ok=True, parents=True)
                
                if _debug:
                    roi2save = _normalize(roi)
                else:
                    roi2save = roi
                
                if save_ext == 'npy':
                    np.save(str(ff), roi2save)
                else:
                    cv2.imwrite(str(ff), roi2save)
                
        if _debug and ii + 1 >=10:
            break
            
            #%%
    all_scores_avgs = np.array(all_scores_avgs)
    
    import matplotlib.pylab as plt
    plt.figure()
    plt.plot(np.sort(all_scores_avgs))
    plt.plot(plt.xlim(), (fg_edge_th,fg_edge_th), 'r:')
    plt.plot(plt.xlim(), (bg_edge_th,bg_edge_th), 'r:')
    
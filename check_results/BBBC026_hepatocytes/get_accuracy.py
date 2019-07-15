#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:07:42 2019

@author: avelinojaver
"""
import sys
from pathlib import Path
dname = Path(__file__).resolve().parents[2]
sys.path.append(str(dname))

from bgnd_removal.models import UNet

import numpy as np
import torch
import cv2
import tqdm
from skimage.measure import regionprops

from get_counts import get_labels


def calculate_performance(segmentation_labels, target_coords):
    grouped_labels = {}
    coords_by_class = {}
    results = {}
    for k_pred, pred_labels in  segmentation_labels.items():
        
        grouped_labels[k_pred] = {}
        coords_by_class[k_pred] = {}
        
        #get the labels center of mass
        props = regionprops(pred_labels)
        centroids_per_label = {x.label:x.centroid for x in props}
        
        #get all the labels and ignore label zero
        all_labs = np.unique(pred_labels)[1:]
        grouped_labels['u'] = set(all_labs)
        for k_target, coord_target in target_coords.items():
            intersected_labels = pred_labels[coord_target[...,1], coord_target[...,0]]
            intersected_labels = (intersected_labels[intersected_labels>0])
            
            #remove labels from the `missing` class
            grouped_labels['u'] = grouped_labels['u'] - set(intersected_labels)
            
            #save labels per class
            grouped_labels[k_pred][k_target] = intersected_labels
            
            #add coordinates per class
            coords_by_class[k_pred][k_target] = np.array([centroids_per_label[x] for x in intersected_labels])
        
        #add coordinates to the `missing` class
        coords_by_class[k_pred]['u'] = np.array([centroids_per_label[x] for x in grouped_labels['u']])
        
    
        TP = len(grouped_labels[k_pred][k_pred])
        tot_true = target_coords[k_pred].shape[0]
        tot_pred = len(all_labs)
        
        
        recall = TP /  tot_true if tot_true else np.nan
        precision = TP / tot_pred if tot_pred else np.nan
        
        
        N = precision + recall
        F1 = 2*recall*precision/N if N else np.nan
        
        results[k_pred] = {'R':recall, 'P':precision, 'F1':F1, 'TP':TP, 'tot_true':tot_true, 'tot_pred':tot_pred}
    return results, coords_by_class

def read_GT(fname):
    img = cv2.imread(str(fname), -1)
    img = img[..., :3]
    
    img_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
    bad = (img[..., 0] == 255) & (img[..., 1] <= 10) & (img[..., 2] <= 10)
    fib =  (img[..., 0] <= 10) & (img[..., 1] == 255) & (img[..., 2] <= 10)
    hep =  (img[..., 0] <= 10) & (img[..., 1] <= 10) & (img[..., 2] == 255)
    
    #ground truth
    target_coords = {}
    target_coords['fib'] = cv2.connectedComponentsWithStats(fib.astype(np.uint8))[-1][1:].astype(np.int)
    target_coords['hep'] = cv2.connectedComponentsWithStats(hep.astype(np.uint8))[-1][1:].astype(np.int)
    target_coords['bad'] = cv2.connectedComponentsWithStats(bad.astype(np.uint8))[-1][1:].astype(np.int)
    
    #remove the labelled pixels and conver the image into gray scale
    peaks2remove = bad | fib | hep
    med = cv2.medianBlur(img_g, ksize= 11) + np.random.normal(0, 2, img_g.shape).round().astype(np.int)
    img_g[peaks2remove] = med[peaks2remove]
    
    return img, img_g, target_coords

#%%
if __name__ == '__main__':
    #root_dir = Path.home() / 'workspace/datasets/BBBC026/BBBC026_GT_images'
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/papers/miccai2019/data/hepatocytes/BBBC026/BBBC026_GT_images'
    
    
    bn = 'BBBC026_unet-decomposition_l1smooth_20190701_204516_adam_lr0.00032_wd0.0_batch32'
    model_path = Path.home() / 'workspace/denoising/results/BBBC026' / bn / f'checkpoint.pth.tar'
    
    #bn = 'BBBC026-separated_unet_l1smooth_20190226_082017_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-hepatocytes_unet_l1smooth_20190226_082040_adam_lr0.00032_wd0.0_batch32'
    #bn = 'BBBC026-fibroblasts_unet_l1smooth_20190226_082007_adam_lr0.00032_wd0.0_batch32'
    #n_epochs = 349#299#
    #model_path = Path.home() / 'workspace/denoising/results/_old_BBBC026' / bn / f'checkpoint-{n_epochs}.pth.tar'
    
    
    
    save_dir = '/Users/avelinojaver/OneDrive - Nexus365/papers/miccai2019/data/hepatocytes/'
    save_dir = Path(save_dir)
    
    n_ch_in, n_ch_out  = 1, 1
    if '-separated' in bn:
        n_ch_out = 3
    
    cuda_id = 0
    min_area = 300
    _debug = True
    
    
    
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model = model.to(device)
    model.eval()
    
    fnames = [x for x in Path(root_dir).glob('*.png') if not x.name.startswith('.')]
    fnames = sorted(fnames, key = lambda x : x.name)
    
    all_outputs = {}
    for fname in tqdm.tqdm(fnames):
        #%%
        
        
        img, img_g, target_coords = read_GT(fname)
        
        #%%
        #calculate predictions
        x = img_g[None].astype(np.float32)
        pix_top = np.percentile(x, 99)
        xn = x/pix_top
        with torch.no_grad():
            X = torch.from_numpy(xn[None])
            X = X.to(device)
            Xhat = model(X)
    
        xhat = Xhat[0].detach().cpu().numpy()
        
        #%%
        if 'separate' in bn:
            prediction_maps = {key:xhat[ii] for ii,key in enumerate(['hep', 'bad', 'fib'])}
        elif 'fibroblast' in bn:
            prediction_maps = {'fib':xhat[0]}
        elif 'hepatocyte' in bn:
            prediction_maps = {'hep':xhat[0]}
            
        
        segmentation_labels = {}
        for key, pred in prediction_maps.items():
            th_min = 0.5 if key == 'bad' else 0.
            
            lab, _ = get_labels(pred, th_min = th_min, min_area=min_area)
            segmentation_labels[key] = lab
        
        #%%
        results, coords_by_class = calculate_performance(segmentation_labels, target_coords)
        
        all_outputs[fname.stem] = results
        
        
        
        #%%
#        
#        labels_hep, _ = get_labels(xhat[0], th_min = 0., min_area=min_area)
#        pred_hep_as_hep = labels_hep[cm_hep[...,1], cm_hep[...,0]]
#        pred_hep_as_hep = set(pred_hep_as_hep[pred_hep_as_hep>0])
#        
#        TP = len(pred_hep_as_hep)
#        tot_true = cm_hep.shape[0]
#        recall = TP /  tot_true
#        
#        all_labs = set(np.unique(labels_hep[labels_hep>0]))
#        tot_pred = len(all_labs)
#        precision = TP / tot_pred
#        
#        if xhat.shape[0] == 3:
#            #this is in demixer mode so we can obtain data from the other labels
#            labels_fib, _ = get_labels(xhat[-1], th_min = 0., min_area=min_area)
#            labels_bad, _ = get_labels(xhat[1], th_min = 0.5, min_area=min_area)
#        
#            pred_fib_as_hep = labels_hep[cm_fib[...,1], cm_fib[...,0]]
#            pred_fib_as_hep = set(pred_fib_as_hep[pred_fib_as_hep>0])
#            
#            pred_bad_as_hep = labels_hep[cm_bad[...,1], cm_bad[...,0]]
#            pred_bad_as_hep = set(pred_bad_as_hep[pred_bad_as_hep>0])
#            
#            missing_l = set(all_labs) - (set(pred_hep_as_hep) | set(pred_fib_as_hep) | set(pred_bad_as_hep))
#       
#            unmarked = len(missing_l) / tot_pred
#            wrong_fib = len(pred_fib_as_hep) / tot_pred
#            wrong_bad = len(pred_bad_as_hep) / tot_pred
#            unknown = len(missing_l) / tot_pred
#            
#            
#            #fibroblast data
#            pred_fib_as_fib = labels_fib[cm_fib[...,1], cm_fib[...,0]]
#            pred_fib_as_fib = set(pred_fib_as_fib[pred_fib_as_fib>0])
#            
#            TP = len(pred_fib_as_fib)
#            tot_true = cm_fib.shape[0]
#            fib_recall = TP /  tot_true
#            
#            all_labs = set(np.unique(labels_fib[labels_fib>0]))
#            tot_pred = len(all_labs)
#            fib_precision = TP / tot_pred
#            
#            
#        else:
#            pred_fib_as_hep = []
#            pred_bad_as_hep = []
#            
#            unmarked, wrong_fib, wrong_bad, unknown,fib_recall,fib_precision = 6*[np.nan]
#        
#        coords_by_label = {}
#        
#        props = regionprops(labels_hep)
#        props_l = {x.label:x.centroid for x in props}
#        
#        
#        coords_by_label['hep'] = np.array([props_l[x] for x in pred_hep_as_hep])
#        coords_by_label['fib'] = np.array([props_l[x] for x in pred_fib_as_hep])
#        coords_by_label['bad'] = np.array([props_l[x] for x in pred_bad_as_hep])
#        coords_by_label['u'] = np.array([props_l[x] for x in missing_l])
#    
#        #fraction of hep labeled as fibs that were also labeled as fibs
#        cc = coords_by_label['fib'].round().astype(np.int)
#        
#        if cc.size > 0:
#            labs = labels_fib[cc[..., 0], cc[..., 1]]
#            also_fibs = np.mean(labs>0)
#            
#            coords_by_label['overlaps'] = cc[labs>0]
#        else:
#            also_fibs = 0
#        
#        
#        
#        #overlap = len(hep_or_fib) / tot_pred
#        
#        _output = [f'{fname.name}',
#                   'hepatocytes:',
#                   f'recall : {recall:.4}',
#                   f'precision : {precision:.4} -> fib {wrong_fib:.4}; bad {wrong_bad:.4}; unknown {unknown:.4}',
#                   f'overlap : {also_fibs}',
#                   
#                   'fibroblasts:',
#                   f'recall : {fib_recall:.4}',
#                   f'precision : {fib_precision:.4}',
#                   ]       
#        _output = '\n'.join(_output)       
        #%%
        
        
        colors = dict(hep='r', fib='g', u='y', bad='b')
        
        if _debug:
            import matplotlib.pylab as plt
            
            
            for k_target in ['hep', 'fib']:
                if not k_target in coords_by_class:
                    continue
                
                n_figs = 1 + len(prediction_maps)
                #for key, pred in prediction_maps.items():
                fig, axs = plt.subplots(1,n_figs,sharex=True, sharey=True, figsize = (20, 10))
                
                
                
                if n_figs > 2:
                    for i_plot, k_pred in enumerate(['hep', 'fib', 'bad']):
                        pred = prediction_maps[k_pred]
                        axs[i_plot + 1].imshow(pred,  cmap = 'gray')
                else:
                    pred = prediction_maps[k_target]
                    axs[1].imshow(pred,  cmap = 'gray')
                
                
                img_rgb = img[..., 3::-1]
                
                
                if False:
                    axs[0].imshow(img_g)
                    gt_coords = target_coords[k_target]
                    
                    #axs[0].imshow(img_g,  cmap = 'gray')
                    axs[0].plot(gt_coords[..., 0], gt_coords[..., 1], 'xc')
                    for (k, coords) in coords_by_class[k_target].items():
                        if coords.size == 0:
                            continue
                        axs[0].plot(coords[..., 1], coords[..., 0], 'o', color=colors[k])
                    
                else:
                   img_rgb = img[..., 3::-1]
                   axs[0].imshow(img_rgb)
                   break
                
                
                
#%%
#            
#            if xhat.shape[0] == 3:
#                n_figs = 4
#            else:
#                n_figs = 2
#            
#            fig, axs = plt.subplots(1,n_figs,sharex=True, sharey=True, figsize = (20, 10))
#            
#            axs[1].imshow(xhat[0],  cmap = 'gray')
#            
#            if len(axs) > 2:
#                axs[2].imshow(xhat[-1],  cmap = 'gray')
#                axs[3].imshow(xhat[1],  cmap = 'gray')
#            
#            plt.suptitle(fname.name)
#            
#            
#            if False:
#                axs[0].imshow(img_g,  cmap = 'gray')
#                axs[0].plot(cm_hep[..., 0], cm_hep[..., 1], 'xm')
#                for (k, coords) in coords_by_label.items():
#                    if coords.size == 0:
#                        continue
#                    axs[0].plot(coords[..., 1], coords[..., 0], 'o', color=colors[k])
#            else:
#                img_rgb = img[..., 3::-1]
#                axs[0].imshow(img_rgb)
#        
#            for ax in axs:
#                ax.axis('off')
        #%%
#        output2save = '*****\n'.join(all_outputs)
#        
#        
#        save_dir.mkdir(parents=True, exist_ok=True)
#    
#        save_name = save_dir / f'accuracies_{n_epochs}_{bn}.txt'
#        with open(save_name, 'w') as fid:
#            fid.write(output2save)
#        a
        
            #%%
    #%%
    dat_per_class = {}
    
    
    output2save = []
    for iname, res in all_outputs.items():
        out = ['*******', f'{iname}']
        for k_class in ['hep', 'fib']:
            if not k_class in res:
                continue
            
            res_class = res[k_class]
            ll = f"{k_class} -> (P = {res_class['P']:.2f}, R = {res_class['R']:.2f}, F1 = {res_class['F1']:.2f})"
            out.append(ll)
            
            if not k_class in dat_per_class:
                dat_per_class[k_class] = {'TP':0, 'tot_pred':0, 'tot_true':0}
            
            for k, val in dat_per_class[k_class].items():
                dat_per_class[k_class][k] = val + res_class[k]
            
            
        output2save += out
    
    out = ['*******', 'all']
    for k_class in ['hep', 'fib']:
        if not k_class in res:
                continue
        dat = dat_per_class[k_class]
        R = dat['TP'] / dat['tot_true']
        P = dat['TP'] / dat['tot_pred']
        F1 = 2*R*P/(R+P)
        
        ll = f"{k_class} -> (P = {P:.2f}, R = {R:.2f}, F1 = {F1:.2f})"
        out.append(ll)
    output2save += out
        
        
    output2save = '\n'.join(output2save)
    print(output2save)
        
        
        
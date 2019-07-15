#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 23:56:23 2019

@author: avelinojaver
"""

from pathlib import Path
from fluo_merged_flow import  _test_load_BBBC42_simple, _test_load_BBBC26, _test_load_microglia
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pylab as plt

if __name__ == '__main__':
    save_dir = '/Users/avelinojaver/OneDrive - Nexus365/papers/miccai2019/MLMI/data/training_pairs/'
    save_dir = Path(save_dir)
    
    train_funcs = dict(
            #BBBC26 = _test_load_BBBC26,
            #BBBC42_simple = _test_load_BBBC42_simple,
            microglia = _test_load_microglia
            )
    #%%
    cdict = {'red':   [(0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0),
                   (1.0,  1.0, 1.0)],

         'green': [(0.0,  0.0, 0.0),
                   (1.0,  1.0, 1.0)],

         'blue':  [(0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)],
                   
         'alpha':  [(0.0,  0.0, 0.0),
                    (0.5,  1.0, 1.0),
                   (1.0,  1.0, 1.0)]
         }
    cmap = LinearSegmentedColormap('test', cdict, N=100)
    #cmap = LinearSegmentedColormap.from_list('test', [(0,0,0), (0,1,0)], N=100)
         #%%
    for set_type, func in train_funcs.items():
        gen = func(_debug = True)
        #%%
        
        for ii in range(20):
            fgnd_p1, fgnd_p2, overlap_tracker = gen.get_cell_pairs(gen.cells1_files, gen.n_cells_per_crop)
            bgnd1_p1 = gen._get_random_bgnd()
            bgnd2_p1 = gen.get_cell_pairs(gen.cells2_files, gen.n_bgnd_per_crop, overlap_tracker)[0]
            
            bgnd1_p2 = gen._get_random_bgnd()
            bgnd2_p2 = gen.get_cell_pairs(gen.cells2_files, gen.n_bgnd_per_crop, overlap_tracker)[0]
                
                
            out1 = gen._merge(fgnd_p1, bgnd1_p1, bgnd2_p1)
            out2 = gen._merge(fgnd_p2, bgnd1_p2, bgnd2_p2)
                    
                
            if gen.is_log_transform:
                def _log_normalize(x):
                    _scale_log = [np.log(x+1) for x in gen.int_scale]
                    
                    #denormalize
                    xd = x * (gen.int_scale[1]-gen.int_scale[0]) + gen.int_scale[0]
                    xd = np.log(xd + 1)
                    xd = (xd-_scale_log[0])/(_scale_log[1]-_scale_log[0])
                    return xd
                
                out1 = _log_normalize(out1)
                out2 = _log_normalize(out2)
            
                fgnd_p1 = _log_normalize(fgnd_p1)
            fig, axs = plt.subplots(1, 2, figsize = (10, 5))
            axs[0].imshow(out1[0], cmap = 'gray')
            
            fgnd_p1 = fgnd_p1/fgnd_p1.max()
            fgnd_mask = fgnd_p1[0]*0.25
            fgnd_mask[fgnd_mask>0] + 0.5
            
            alpha = 0.5 if set_type == 'microglia' else 0.5
            axs[0].imshow(fgnd_mask, cmap=cmap, alpha=alpha)
            axs[1].imshow(out2[0], cmap = 'gray')
            axs[1].imshow(fgnd_mask, cmap=cmap, alpha=alpha)
            
            
#            fig, axs = plt.subplots(2, 4, figsize = (30, 15))
#            axs[0][0].imshow(bgnd2_p1[0], cmap = 'gray')
#            axs[0][1].imshow(bgnd1_p1[0], cmap = 'gray')
#            axs[0][2].imshow(fgnd_p1[0], cmap = 'gray')
#            axs[0][3].imshow(out1[0], cmap = 'gray')
#            
#            
#            axs[1][0].imshow(bgnd2_p2[0], cmap = 'gray')
#            axs[1][1].imshow(bgnd1_p2[0], cmap = 'gray')
#            axs[1][2].imshow(fgnd_p2[0], cmap = 'gray')
#            axs[1][3].imshow(out2[0], cmap = 'gray')
#            
            for ax in axs.flatten():
                ax.axis('off')
                
            
            
            save_name = save_dir / set_type / f'{ii}.pdf'
            save_name.parent.mkdir(exist_ok = True, parents = True)
            fig.savefig(save_name)
            #plt.close()
        
        
        
        
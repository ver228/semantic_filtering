#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:12:10 2018

@author: avelinojaver
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
#%%
if __name__ == '__main__':
    save_name = Path.home() / 'workspace/RESULTS_worms-divergent-samples.csv'
    
    df = pd.read_csv(save_name, index_col=0)
    df = df[df['basename'].str.contains('l1smooth')]
    
    dd = [x.split('/')[0].partition('-samples-')[-1] for x in df['basename']]
    dd = [int(x) if x  else 279 for x in dd]
    df['n_videos'] = dd
    
    #%%
    
    
    outs = []
    for n_videos, dat in df.groupby('n_videos'):
        X = np.mean(dat)
        SD = np.std(dat) if len(dat) > 1 else None
        
        out_ = [f'{n_videos}']
        for ff in ['P', 'R', 'F1', 'mIoU']:
            m = X[ff]
            if SD is not None:
                s = SD[ff]
                out_.append(f'${m:.3f} \pm {s:.2f}$')
            else:
                out_.append(f'{m:.3f}')
        
        
        outs.append(' & '.join(out_))
    
    print(' \\\\\n'.join(outs))
    
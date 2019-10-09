#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 20:42:01 2019

@author: avelinojaver
"""

from pathlib import Path

data_root_dir = Path.home() / 'workspace/denoising/data'

data_types_basic = {
    'worms-divergent' : dict(
    flow_args = dict(
            root_dir = data_root_dir / 'c_elegans_divergent/train/',
            is_log_transform = False, 
            scale_int = (0, 255),
            samples_per_epoch = 2790,
            max_samples_per_set = None
        ),
    n_ch_in = 1,
    n_ch_out = 1
    ),    
            
    'worms-divergent-samples-5' : dict(
    flow_args = dict(
            root_dir = data_root_dir / 'c_elegans_divergent/train/',
            is_log_transform = False, 
            scale_int = (0, 255),
            samples_per_epoch = 2790,
            max_samples_per_set = 5
        ),
    n_ch_in = 1,
    n_ch_out = 1
    ),    
    
    'worms-divergent-samples-10' : dict(
    flow_args = dict(
            root_dir = data_root_dir / 'c_elegans_divergent/train/',
            is_log_transform = False, 
            scale_int = (0, 255),
            samples_per_epoch = 2790,
            max_samples_per_set = 10
        ),
    n_ch_in = 1,
    n_ch_out = 1
    ),
            
    'worms-divergent-samples-25' : dict(
    flow_args = dict(
            root_dir = data_root_dir / 'c_elegans_divergent/train/',
            is_log_transform = False, 
            scale_int = (0, 255),
            samples_per_epoch = 2790,
            max_samples_per_set = 25
        ),
    n_ch_in = 1,
    n_ch_out = 1
    ),    
            
    'worms-divergent-samples-100' : dict(
    flow_args = dict(
            root_dir = data_root_dir / 'c_elegans_divergent/train/',
            is_log_transform = False, 
            scale_int = (0, 255),
            samples_per_epoch = 2790,
            max_samples_per_set = 100
        ),
    n_ch_in = 1,
    n_ch_out = 1
    ),    
    }


data_types_synthetic = {
    'BBBC042-simple-bgnd': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_bgnd/train',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            
             crop_size = (256, 256),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = (1, 5),
             n_bgnd_per_crop = (1, 3),
             intensity_factor = (0.9, 1.1),
             fg_quantile_range = (0, 10),
             bg_quantile_range = (25, 75),
             
             frac_crop_valid = 0.1,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             max_overlap = 0.5
        ),
    n_ch_in = 1,
    n_ch_out = 1
    ),    
    'BBBC042-simple-bgnd-S5': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_bgnd/train_S5',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            
             crop_size = (256, 256),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = (1, 5),
             n_bgnd_per_crop = (1, 3),
             intensity_factor = (0.9, 1.1),
             fg_quantile_range = (0, 10),
             bg_quantile_range = (25, 75),
             
             frac_crop_valid = 0.1,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             max_overlap = 0.5
        ),
    n_ch_in = 1,
    n_ch_out = 1
    ),    
    'BBBC042-simple-bgnd-S10': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_bgnd/train_S10',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            
             crop_size = (256, 256),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = (1, 5),
             n_bgnd_per_crop = (1, 3),
             intensity_factor = (0.9, 1.1),
             fg_quantile_range = (0, 10),
             bg_quantile_range = (25, 75),
             
             frac_crop_valid = 0.1,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             max_overlap = 0.5
        ),
    n_ch_in = 1,
    n_ch_out = 1
    ),    
    'BBBC042-simple-bgnd-S25': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_bgnd/train_S25',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            
             crop_size = (256, 256),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = (1, 5),
             n_bgnd_per_crop = (1, 3),
             intensity_factor = (0.9, 1.1),
             fg_quantile_range = (0, 10),
             bg_quantile_range = (25, 75),
             
             frac_crop_valid = 0.1,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             max_overlap = 0.5
        ),
    n_ch_in = 1,
    n_ch_out = 1
    ),   
    'BBBC042-simple-bgnd-S100': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_bgnd/train_S100',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            
             crop_size = (256, 256),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = (1, 5),
             n_bgnd_per_crop = (1, 3),
             intensity_factor = (0.9, 1.1),
             fg_quantile_range = (0, 10),
             bg_quantile_range = (25, 75),
             
             frac_crop_valid = 0.1,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             max_overlap = 0.5
        ),
    n_ch_in = 1,
    n_ch_out = 1
    ),    
    'BBBC042-simple': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_v2/train',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            
             crop_size = (256, 256),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = (1, 5),
             n_bgnd_per_crop = (1, 3),
             intensity_factor = (0.9, 1.1),
             fg_quantile_range = (0, 10),
             bg_quantile_range = (25, 75),
             
             frac_crop_valid = 0.1,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             max_overlap = 0.5
        ),
    n_ch_in = 1,
    n_ch_out = 1
    ),    
    'BBBC042-simple-S5': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_v2/train_S5',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            
             crop_size = (256, 256),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = (1, 5),
             n_bgnd_per_crop = (1, 3),
             intensity_factor = (0.9, 1.1),
             fg_quantile_range = (0, 10),
             bg_quantile_range = (25, 75),
             
             frac_crop_valid = 0.1,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             max_overlap = 0.5
        ),
    n_ch_in = 1,
    n_ch_out = 1
    ),    
    'BBBC042-simple-S10': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_v2/train_S10',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            
             crop_size = (256, 256),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = (1, 5),
             n_bgnd_per_crop = (1, 3),
             intensity_factor = (0.9, 1.1),
             fg_quantile_range = (0, 10),
             bg_quantile_range = (25, 75),
             
             frac_crop_valid = 0.1,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             max_overlap = 0.5
        ),
    n_ch_in = 1,
    n_ch_out = 1
    ),
    'BBBC042-simple-S25': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_v2/train_S25',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            
             crop_size = (256, 256),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = (1, 5),
             n_bgnd_per_crop = (1, 3),
             intensity_factor = (0.9, 1.1),
             fg_quantile_range = (0, 10),
             bg_quantile_range = (25, 75),
             
             frac_crop_valid = 0.1,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             max_overlap = 0.5
        ),
    n_ch_in = 1,
    n_ch_out = 1
    ),
    'BBBC042-simple-S100': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_v2/train_S100',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            
             crop_size = (256, 256),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = (1, 5),
             n_bgnd_per_crop = (1, 3),
             intensity_factor = (0.9, 1.1),
             fg_quantile_range = (0, 10),
             bg_quantile_range = (25, 75),
             
             frac_crop_valid = 0.1,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             max_overlap = 0.5
        ),
    n_ch_in = 1,
    n_ch_out = 1
    ),    
     'BBBC042-colour-bgnd': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_colour_more_bgnd/train',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            crop_size = (256, 256, 3),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = (1, 5),
             n_bgnd_per_crop = (0, 3),
             
             intensity_factor = (0.9, 1.1),
             fg_quantile_range = (90, 100),
             bg_quantile_range = (25, 75),
             
             frac_crop_valid = 0.1,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             
             max_overlap = 0.5,
             null_value = 1.,
             merge_by_prod = True,
        ),
    n_ch_in = 3,
    n_ch_out = 3
    ),
            
    'BBBC042-colour-bgnd-S5': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_colour_more_bgnd/train_S5',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            crop_size = (256, 256, 3),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = (1, 5),
             n_bgnd_per_crop = (0, 3),
             
             intensity_factor = (0.9, 1.1),
             fg_quantile_range = (90, 100),
             bg_quantile_range = (25, 75),
             
             frac_crop_valid = 0.1,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             
             max_overlap = 0.5,
             null_value = 1.,
             merge_by_prod = True,
        ),
    n_ch_in = 3,
    n_ch_out = 3
    ),
    'BBBC042-colour-bgnd-S10': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_colour_more_bgnd/train_S10',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            crop_size = (256, 256, 3),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = (1, 5),
             n_bgnd_per_crop = (0, 3),
             
             intensity_factor = (0.9, 1.1),
             fg_quantile_range = (90, 100),
             bg_quantile_range = (25, 75),
             
             frac_crop_valid = 0.1,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             
             max_overlap = 0.5,
             null_value = 1.,
             merge_by_prod = True,
        ),
    n_ch_in = 3,
    n_ch_out = 3
    ),
    'BBBC042-colour-bgnd-S25': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_colour_more_bgnd/train_S25',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            crop_size = (256, 256, 3),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = (1, 5),
             n_bgnd_per_crop = (0, 3),
             
             intensity_factor = (0.9, 1.1),
             fg_quantile_range = (90, 100),
             bg_quantile_range = (25, 75),
             
             frac_crop_valid = 0.1,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             
             max_overlap = 0.5,
             null_value = 1.,
             merge_by_prod = True,
        ),
    n_ch_in = 3,
    n_ch_out = 3
    ),        
            
    'BBBC042-colour-bgnd-S100': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_colour_more_bgnd/train_S100',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            crop_size = (256, 256, 3),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = (1, 5),
             n_bgnd_per_crop = (0, 3),
             
             intensity_factor = (0.9, 1.1),
             fg_quantile_range = (90, 100),
             bg_quantile_range = (25, 75),
             
             frac_crop_valid = 0.1,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             
             max_overlap = 0.5,
             null_value = 1.,
             merge_by_prod = True,
        ),
    n_ch_in = 3,
    n_ch_out = 3
    ),
            
    'BBBC042-colour-v4': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_colour/train',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            crop_size = (256, 256, 3),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = (2, 5),
             n_bgnd_per_crop = (1, 3),
             intensity_factor = (0.95, 1.05),
             fg_quantile_range = (90, 100),
             bg_quantile_range = (25, 75),
             frac_crop_valid = 0.1,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             max_overlap = 0.25,
             null_value = 1.,
             merge_by_prod = True
             
        ),
    n_ch_in = 3,
    n_ch_out = 3
    ),       
    
    'BBBC042-colour-v4-S100': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_colour/train_S100',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            crop_size = (256, 256, 3),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = (2, 5),
             n_bgnd_per_crop = (1, 3),
             intensity_factor = (0.95, 1.05),
             fg_quantile_range = (90, 100),
             bg_quantile_range = (25, 75),
             frac_crop_valid = 0.1,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             max_overlap = 0.25,
             null_value = 1.,
             merge_by_prod = True
             
        ),
    n_ch_in = 3,
    n_ch_out = 3
    ),
    'BBBC042-colour-v4-S25': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_colour/train_S25',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            crop_size = (256, 256, 3),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = (2, 5),
             n_bgnd_per_crop = (1, 3),
             intensity_factor = (0.95, 1.05),
             fg_quantile_range = (90, 100),
             bg_quantile_range = (25, 75),
             frac_crop_valid = 0.1,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             max_overlap = 0.25,
             null_value = 1.,
             merge_by_prod = True
             
        ),
    n_ch_in = 3,
    n_ch_out = 3
    ),
            
    'BBBC042-colour-v4-S10': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_colour/train_S10',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            crop_size = (256, 256, 3),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = (2, 5),
             n_bgnd_per_crop = (1, 3),
             intensity_factor = (0.95, 1.05),
             fg_quantile_range = (90, 100),
             bg_quantile_range = (25, 75),
             frac_crop_valid = 0.1,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             max_overlap = 0.25,
             null_value = 1.,
             merge_by_prod = True
             
        ),
    n_ch_in = 3,
    n_ch_out = 3
    ),
            
    'BBBC042-colour-v4-S5': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_colour/train_S5',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            crop_size = (256, 256, 3),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = (2, 5),
             n_bgnd_per_crop = (1, 3),
             intensity_factor = (0.95, 1.05),
             fg_quantile_range = (90, 100),
             bg_quantile_range = (25, 75),
             frac_crop_valid = 0.1,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             max_overlap = 0.25,
             null_value = 1.,
             merge_by_prod = True
             
        ),
    n_ch_in = 3,
    n_ch_out = 3
    ),
            
    'BBBC026-fold1':dict(
            flow_args = dict(
                        root_dir =  data_root_dir / 'BBBC026/fold1',
                            bgnd_prefix = 'background',
                            cells1_prefix = 'hepatocytes',
                            cells2_prefix = 'fibroblasts',
                           crop_size = (256, 256),
                             is_log_transform = False,
                             int_scale = (0, 255),
                             img_ext = '*.png',
                             is_timeseries_dir = False,
                             n_cells_per_crop = (3, 8),
                             intensity_factor = (0.5, 1.5),
                             frac_crop_valid = 0.5,
                             zoom_range = (0.75, 1.3),
                             rotate_range = (0, 90),
                             max_overlap = 0.9,
                             fg_quantile_range = (0, 10),
                             bg_quantile_range = (25, 75),
                             ),
    n_ch_in = 1,
    n_ch_out = 1
    ),
    'BBBC026-fold2':dict(
            flow_args = dict(
                        root_dir =  data_root_dir / 'BBBC026/fold2',
                            bgnd_prefix = 'background',
                            cells1_prefix = 'hepatocytes',
                            cells2_prefix = 'fibroblasts',
                           crop_size = (256, 256),
                             is_log_transform = False,
                             int_scale = (0, 255),
                             img_ext = '*.png',
                             is_timeseries_dir = False,
                             n_cells_per_crop = (3, 8),
                             intensity_factor = (0.5, 1.5),
                             frac_crop_valid = 0.5,
                             zoom_range = (0.75, 1.3),
                             rotate_range = (0, 90),
                             max_overlap = 0.9,
                             fg_quantile_range = (0, 10),
                             bg_quantile_range = (25, 75),
                             ),
    n_ch_in = 1,
    n_ch_out = 1
    ),
                    
    'BBBC026-fold3':dict(
            flow_args = dict(
                        root_dir =  data_root_dir / 'BBBC026/fold3',
                            bgnd_prefix = 'background',
                            cells1_prefix = 'hepatocytes',
                            cells2_prefix = 'fibroblasts',
                           crop_size = (256, 256),
                             is_log_transform = False,
                             int_scale = (0, 255),
                             img_ext = '*.png',
                             is_timeseries_dir = False,
                             n_cells_per_crop = (3, 8),
                             intensity_factor = (0.5, 1.5),
                             frac_crop_valid = 0.5,
                             zoom_range = (0.75, 1.3),
                             rotate_range = (0, 90),
                             max_overlap = 0.9,
                             fg_quantile_range = (0, 10),
                             bg_quantile_range = (25, 75),
                             ),
    n_ch_in = 1,
    n_ch_out = 1
    ),
                    
    'BBBC026-fold4':dict(
            flow_args = dict(
                        root_dir =  data_root_dir / 'BBBC026/fold4',
                            bgnd_prefix = 'background',
                            cells1_prefix = 'hepatocytes',
                            cells2_prefix = 'fibroblasts',
                           crop_size = (256, 256),
                             is_log_transform = False,
                             int_scale = (0, 255),
                             img_ext = '*.png',
                             is_timeseries_dir = False,
                             n_cells_per_crop = (3, 8),
                             intensity_factor = (0.5, 1.5),
                             frac_crop_valid = 0.5,
                             zoom_range = (0.75, 1.3),
                             rotate_range = (0, 90),
                             max_overlap = 0.9,
                             fg_quantile_range = (0, 10),
                             bg_quantile_range = (25, 75),
                             ),
    n_ch_in = 1,
    n_ch_out = 1
    ),
    'BBBC026-fold5':dict(
            flow_args = dict(
                        root_dir =  data_root_dir / 'BBBC026/fold5',
                            bgnd_prefix = 'background',
                            cells1_prefix = 'hepatocytes',
                            cells2_prefix = 'fibroblasts',
                           crop_size = (256, 256),
                             is_log_transform = False,
                             int_scale = (0, 255),
                             img_ext = '*.png',
                             is_timeseries_dir = False,
                             n_cells_per_crop = (3, 8),
                             intensity_factor = (0.5, 1.5),
                             frac_crop_valid = 0.5,
                             zoom_range = (0.75, 1.3),
                             rotate_range = (0, 90),
                             max_overlap = 0.9,
                             fg_quantile_range = (0, 10),
                             bg_quantile_range = (25, 75),
                             ),
    n_ch_in = 1,
    n_ch_out = 1
    ),
    'BBBC026-fibroblast-fold1':dict(
            flow_args = dict(
                        root_dir =  data_root_dir / 'BBBC026/fold1',
                            bgnd_prefix = 'background',
                            cells1_prefix = 'fibroblasts',
                            cells2_prefix = 'hepatocytes',
                           crop_size = (256, 256),
                             is_log_transform = False,
                             int_scale = (0, 255),
                             img_ext = '*.png',
                             is_timeseries_dir = False,
                             n_cells_per_crop = (3, 8),
                             intensity_factor = (0.5, 1.5),
                             frac_crop_valid = 0.5,
                             zoom_range = (0.75, 1.3),
                             rotate_range = (0, 90),
                             max_overlap = 0.9,
                             fg_quantile_range = (0, 10),
                             bg_quantile_range = (25, 75),
                             ),
    n_ch_in = 1,
    n_ch_out = 1
    ),
    'BBBC026-fibroblast-fold2':dict(
            flow_args = dict(
                        root_dir =  data_root_dir / 'BBBC026/fold2',
                            bgnd_prefix = 'background',
                            cells1_prefix = 'fibroblasts',
                            cells2_prefix = 'hepatocytes',
                           crop_size = (256, 256),
                             is_log_transform = False,
                             int_scale = (0, 255),
                             img_ext = '*.png',
                             is_timeseries_dir = False,
                             n_cells_per_crop = (3, 8),
                             intensity_factor = (0.5, 1.5),
                             frac_crop_valid = 0.5,
                             zoom_range = (0.75, 1.3),
                             rotate_range = (0, 90),
                             max_overlap = 0.9,
                             fg_quantile_range = (0, 10),
                             bg_quantile_range = (25, 75),
                             ),
    n_ch_in = 1,
    n_ch_out = 1
    ),
    'BBBC026-fibroblast-fold3':dict(
            flow_args = dict(
                        root_dir =  data_root_dir / 'BBBC026/fold3',
                            bgnd_prefix = 'background',
                            cells1_prefix = 'fibroblasts',
                            cells2_prefix = 'hepatocytes',
                           crop_size = (256, 256),
                             is_log_transform = False,
                             int_scale = (0, 255),
                             img_ext = '*.png',
                             is_timeseries_dir = False,
                             n_cells_per_crop = (3, 8),
                             intensity_factor = (0.5, 1.5),
                             frac_crop_valid = 0.5,
                             zoom_range = (0.75, 1.3),
                             rotate_range = (0, 90),
                             max_overlap = 0.9,
                             fg_quantile_range = (0, 10),
                             bg_quantile_range = (25, 75),
                             ),
    n_ch_in = 1,
    n_ch_out = 1
    ),                
    'BBBC026-fibroblast-fold4':dict(
            flow_args = dict(
                        root_dir =  data_root_dir / 'BBBC026/fold4',
                            bgnd_prefix = 'background',
                            cells1_prefix = 'fibroblasts',
                            cells2_prefix = 'hepatocytes',
                           crop_size = (256, 256),
                             is_log_transform = False,
                             int_scale = (0, 255),
                             img_ext = '*.png',
                             is_timeseries_dir = False,
                             n_cells_per_crop = (3, 8),
                             intensity_factor = (0.5, 1.5),
                             frac_crop_valid = 0.5,
                             zoom_range = (0.75, 1.3),
                             rotate_range = (0, 90),
                             max_overlap = 0.9,
                             fg_quantile_range = (0, 10),
                             bg_quantile_range = (25, 75),
                             ),
    n_ch_in = 1,
    n_ch_out = 1
    ),
    'BBBC026-fibroblast-fold5':dict(
            flow_args = dict(
                        root_dir =  data_root_dir / 'BBBC026/fold5',
                            bgnd_prefix = 'background',
                            cells1_prefix = 'fibroblasts',
                            cells2_prefix = 'hepatocytes',
                           crop_size = (256, 256),
                             is_log_transform = False,
                             int_scale = (0, 255),
                             img_ext = '*.png',
                             is_timeseries_dir = False,
                             n_cells_per_crop = (3, 8),
                             intensity_factor = (0.5, 1.5),
                             frac_crop_valid = 0.5,
                             zoom_range = (0.75, 1.3),
                             rotate_range = (0, 90),
                             max_overlap = 0.9,
                             fg_quantile_range = (0, 10),
                             bg_quantile_range = (25, 75),
                             ),
    n_ch_in = 1,
    n_ch_out = 1
    ),            
    'microglia':dict(
            flow_args = dict(
                        root_dir =  data_root_dir / 'microglia_v2',
                            bgnd_prefix = 'background',
                            cells1_prefix = 'foreground',
                            cells2_prefix = 'background_crops',
                            crop_size = (512, 512),
                             is_log_transform = True,
                             int_scale = (5, 40000),
                             img_ext = '*.tif',
                             is_timeseries_dir = False,
                             n_cells_per_crop = (1, 5),
                             n_bgnd_per_crop = (3, 6),
                             intensity_factor = (0.05, 10),
                             frac_crop_valid = 0.5,
                             zoom_range = (0.9, 1.1),
                             rotate_range = (0, 90),
                             max_overlap = 1.,
                             fg_quantile_range = (0, 10),
                             bg_quantile_range = (25, 75),
                             ),
    n_ch_in = 1,
    n_ch_out = 1
    ),

    
            
    }

__deprecated__ = {
        
        
        
    'BBBC042-colour': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_colour/train',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            crop_size = (256, 256, 3),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = 3,
             n_bgnd_per_crop = 10,
             intensity_factor = (0.97, 1.03),
             base_quantile = 98,
             frac_crop_valid = 0.2,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             max_overlap = 0.25,
             null_value = 1.,
             merge_by_prod = True
             
        ),
    n_ch_in = 3,
    n_ch_out = 3
    ),
            
    'BBBC042-colour-v2': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_colour_v2/train',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            crop_size = (256, 256, 3),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = 5,
             n_bgnd_per_crop = 5,
             intensity_factor = (0.95, 1.05),
             base_quantile = 98,
             frac_crop_valid = 0.2,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             max_overlap = 0.5,
             null_value = 1.,
             merge_by_prod = True
             
        ),
    n_ch_in = 3,
    n_ch_out = 3
    ),
    'BBBC042-colour-v3': dict(
    flow_args = dict(
            root_dir = data_root_dir / 'BBBC042_colour_v2/train',
            cells1_prefix = 'foreground',
            cells2_prefix = 'background_crops',
            bgnd_prefix = 'background',
            crop_size = (256, 256, 3),
             is_log_transform = False,
             int_scale = (0, 255),
             img_ext = '*.tif',
             is_timeseries_dir = False,
             n_cells_per_crop = (2, 5),
             n_bgnd_per_crop = (1, 3),
             intensity_factor = (0.95, 1.05),
             fg_quantile_range = (90, 100),
             bg_quantile_range = (25, 75),
             frac_crop_valid = 0.1,
             zoom_range = (0.9, 1.1),
             rotate_range = (0, 90),
             max_overlap = 0.25,
             null_value = 1.,
             merge_by_prod = True
             
        ),
    n_ch_in = 3,
    n_ch_out = 3
    ),
    
        }
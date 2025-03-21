#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 11:33:22 2025

@author: mlurig@ad.ufl.edu
"""

#%% imports

import os 
import pandas as pd

import torch

#%% setup

os.chdir(r"/home/mlurig/git-repos/unicom")
from unicom import unicom, dataset
from unicom.retrieval import extract_feat, get_metric

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)


#%% inference

checkpoint_path = r"unicom/checkpoints/butterflies_220k_shuff/ViT_L_14@336px_17_test_92.875.pt"
    
model, transform_clip = unicom.load(name="ViT-L/14@336px")
checkpoint = torch.load(checkpoint_path, map_location="cuda")
if any(key.startswith("model.") for key in checkpoint.keys()):
    checkpoint = {key.replace("model.", ""): value for key, value in checkpoint.items()}
model.load_state_dict(checkpoint)

model = model.cuda()
model.eval()

testset = dataset.butterflies.ButterfliesDataset(
    root="/home/mlurig/Dropbox/projects/2024_nymphalidae/data_raw/segmentation_masks_clean/nymphalidae_whole_specimen-v240606/", 
    mode="eval", 
    class_file="/home/mlurig/Dropbox/projects/2024_nymphalidae/data/clusters_assignments_mod.csv",
    transform=transform_clip,
    )

with torch.no_grad():
    x, y = extract_feat(model, testset, 16, 4)
    metric = get_metric(x, y)


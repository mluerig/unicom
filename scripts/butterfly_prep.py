#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 17:44:10 2025

@author: mlurig@ad.ufl.edu
"""

#%% imports

import os 
import pandas as pd


#%% setup

os.chdir(r"/home/mlurig/git-repos/unicom")

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)

#%%  clusters 

data_clust = pd.read_csv(r"/home/mlurig/Dropbox/projects/2024_nymphalidae/data/clusters_assignments_arthur.csv")

mask_root = r"/home/mlurig/Dropbox/projects/2024_nymphalidae/data_raw/segmentation_masks_clean/nymphalidae_whole_specimen-v240606/"

data_clust = data_clust[data_clust["cluster_id"]!=-2]
data_clust = data_clust.sort_values(by=["species", "cluster_id"])
data_clust["class_str"] = data_clust["species"] + "_" + data_clust["cluster_id"].astype(str)
data_clust["class"] = data_clust["class_str"].astype('category').cat.codes
data_clust["rel_path"] = data_clust.apply(lambda row: os.path.join(row["species"], row["mask_name"]), axis=1)
data_clust.reset_index(drop=True, inplace=True)

data_clust.to_csv(r"/home/mlurig/Dropbox/projects/2024_nymphalidae/data/clusters_assignments_mod.csv", index=False)

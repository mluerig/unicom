#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 17:24:49 2025

@author: mlurig@ad.ufl.edu
"""

import os
import pandas as pd
from .base import BaseDataset

class ButterfliesDataset(BaseDataset):
    def __init__(self, root, mode, class_file, transform=None, train_ratio=0.8):
        # Initialize the base dataset first
        BaseDataset.__init__(self, root, mode, transform)
        
        # Now load and process the CSV file
        csv_path = os.path.join(root, class_file)
        class_df = pd.read_csv(csv_path)
        
        # Convert unique class values to int and sort them
        all_classes = sorted({int(str(c).strip()) for c in class_df["class"].unique()})
        num_classes = len(all_classes)
        split_index = int(num_classes * train_ratio)
        
        if mode == "train":
            desired_classes = set(all_classes[:split_index])
        elif mode == "eval":
            desired_classes = set(all_classes[split_index:])
        else:
            raise ValueError("Mode must be 'train' or 'eval'.")
        
        index = 0
        for _, row in class_df.iterrows():
            label = int(str(row["class"]).strip())
            if label in desired_classes:
                img_path = os.path.join(root, row["rel_path"])
                self.im_paths.append(img_path)
                self.ys.append(label)
                self.I.append(index)
                index += 1
        
        # Check that the labels collected match the desired classes
        if set(self.ys) != desired_classes:
            raise ValueError(f"Mismatch: CSV yielded labels {set(self.ys)} but expected {desired_classes}.")
        
        # Set the classes attribute
        self.classes = sorted(desired_classes)



# -*- coding: utf-8 -*-
import numpy as np


def rm_outliers(x):
    jet0, jet1, jet2, jet3 = data_processing(x)
    
    jets = np.array([jet0,jet1,jet2,jet3])
    
    for jet in jets:
        
        means = np.mean(jet,axis=0)
        stand_dev = np.std(jet,axis=0)
    
    


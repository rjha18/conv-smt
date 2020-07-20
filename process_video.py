

import os
import sys

from data_source import data_source
import numpy as np;


data_source = data_source(kernel_size=15,sigma=1.5);


input_dir = './Data/eagle/'
npy_fnm = './Data/e1.npy';
processed_npy_fnm = './Data/e-processed.npy';


# data_source.frames_to_npy(input_dir,npy_fnm,start=911,subsample=1,pre_h=0,suff_h=0,pre_w=0,suff_w=0,frames_threshold=10000)
data_source.process_npy(npy_fnm,processed_npy_fnm);

#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: Apr 12, 2018



import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, 'source_code/')
from grdecl_read_plot import *

def combine_mc_samples(sample_name, sample_size, i_dim, j_dim, k_dim):
    '''
    This is the function for combining the monte carlo model samples (in GRDECL format) into a single numpy ndarray
    arg:
        sample_name: the unique prefix (including directory) of the model samples
        samples_size: the total number of samples
        i_dim, j_dim, k_dim: the i, j, k dimensions of the model. 
    '''
    combined_smpls = []
    for i in tqdm(range(1, sample_size+1)):
        combined_smpls.append(grdecl_read(sample_name+str(i)+'.GRDECL', i_dim, j_dim, k_dim).flatten())
    combined_smpls = np.asarray(combined_smpls)
    return combined_smpls
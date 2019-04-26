#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: Apr 12, 2018



import numpy as np

from tqdm import tqdm

import sys
sys.path.insert(0, 'source_code/')
from grdecl_read_plot import *

def combine_m_sampls(file_pre, samples_size, i_dim, j_dim, k_dim):    
    '''
    This is the function for combining the prior model samples (in GRDECL format) into a single numpy ndarray
    arg:
        file_pre: the prefix (including directory) of the model samples
        samples_size: the total number of samples
        i_dim, j_dim, k_dim: the i, j, k dimensions of the model. 
    '''
    mat_models=[]
    for realnum in tqdm(range(samples_size)):
        grid_data = grdecl_read(file_pre + str(realnum+1) + '.GRDECL',i_dim,j_dim,k_dim)[0]
        mat_models.append(grid_data.flatten())
    mat_models=np.asarray(mat_models)
    return mat_models
#Signed Distance Functions (SDF)
## Author: David Zhen Yin   
## Contact: yinzhen@stanford.edu
## Date: September 11, 2018

import numpy as np
import skfmm 
#SDF calculation lib

import sys
sys.path.insert(0, 'source_code/')
from grdecl_read_plot import *

## Gobal SDF: glb_sdf
## fac_file: the input facies file name
## fac_num: the facies type that is used for the signed distance calculation

def glb_sdf(fac_file, fac_num):
    fac_mtr=grdecl_read(fac_file, 200, 100, 75)
    fac_mtr[fac_mtr != fac_num] = -1
    fac_mtr[fac_mtr == fac_num] = 1    
    fac_sd = skfmm.distance(fac_mtr)
            
    return fac_sd


def fac_samples_rbnsd(facies_ndarray_file, samples_size, fac_nums, norm_thresh, i_dim, j_dim, k_dim):
    '''
    This is the function to calcualte signed distance for monte carlo facies samples，
    The results are Radial Based Normalization of SD (RBN-SD)
    ## facies_ndarray_file: the file location of facies samples，facies ndarray format: n_samples x grid_dim
    ## fac_nums: the 1d array of facies type numbers that is used for the signed distance calculation
    ## norm_thresh: the starting point of the Radial Based Normalization
    ## i_dim, j_dim, k_dim: x, y, z dimensions of grid model
    '''
    m_dim = i_dim*j_dim*k_dim
    
    fac_rbnsd_all = np.zeros((samples_size, len(fac_nums)*i_dim*j_dim*k_dim))

    for facindex in range(len(fac_nums)):         
        facies_array = np.load(facies_ndarray_file)
        facies_array[facies_array != fac_nums[facindex]] = -1
        facies_array[facies_array == fac_nums[facindex]] = 1   
        
        for i in tqdm(range(samples_size)):
            sdf_val = skfmm.distance(facies_array[i].reshape(k_dim, j_dim, i_dim))
            sdf_val[sdf_val<=-norm_thresh] =\
                -norm_thresh - sdf_val[sdf_val<=-norm_thresh]/min(sdf_val[sdf_val<=-norm_thresh])
            sdf_val[sdf_val>norm_thresh] = \
                norm_thresh + sdf_val[sdf_val>=norm_thresh]/max(sdf_val[sdf_val>=norm_thresh])
            fac_rbnsd_all[i, m_dim*(facindex):m_dim*(facindex+1)] = sdf_val.flatten()
    
    return fac_rbnsd_all 
	

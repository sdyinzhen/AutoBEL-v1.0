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


## Radial Based Normalization of SDF (RBN-SD): rbn_glb_sdf
## fac_file: the input facies file name， format: num_samples x grid_num
## fac_typenum: the facies type number that is used for the signed distance calculation
## Threshold: the starting point of the Radial Based Normalization
## real_nums: total realizaton number
	
def rbn_glb_sdf(fac_file, real_nums, i_max, j_max, k_max, fac_typenum, threshold):
    
    fac_reals = np.loadtxt(fac_file)
    sd_per_fac = []
    for i in tqdm(range(real_nums)):
        fac_mtr=fac_reals[i].reshape(i_max, j_max, k_max)

        fac_mtr[fac_mtr != fac_typenum] = -1
        fac_mtr[fac_mtr == fac_typenum] = 1  

        sdf_val = skfmm.distance(fac_mtr)
        if sdf_val.min() < -threshold:
            sdf_val[sdf_val<=-threshold] =\
                -threshold - sdf_val[sdf_val<=-threshold]/min(sdf_val[sdf_val<=-threshold])
        if sdf_val.max() > threshold:
            sdf_val[sdf_val>threshold] = \
                threshold + sdf_val[sdf_val>=threshold]/max(sdf_val[sdf_val>=threshold])
        sdf_val = sdf_val.flatten()
        sd_per_fac.append(sdf_val)
    sd_per_fac = np.asarray(sd_per_fac)
    return sd_per_fac	
	

## the signed distance is calculated vertically, in well log direction. 
def sdf_facies_verti(fac_file, fac_num):
    fac_mtr=grdecl_read(fac_file, 200, 100, 75)
    fac_mtr[fac_mtr != fac_num] = -1
    fac_mtr[fac_mtr == fac_num] = 1    
    #fac_sdf = skfmm.distance(fac_mtr)
 
    for i in range(100):
        for j in range(200):
            
            if np.var(fac_mtr[:,i,j]) == 0:
                
                if np.mean(fac_mtr[:,i,j]) == fac_num:
                    fac_mtr[:,i,j] = np.arange(0.5,75.5)
                else:
                    fac_mtr[:,i,j] = -np.arange(0.5,75.5)
            else:
                fac_mtr[:,i,j] = skfmm.distance(fac_mtr[:,i,j])  
            
    return fac_mtr


# errorCorrection_FaciesSmooth_3D(Models_mat, i_dim, j_dim, k_dim)
# Authors: Chen Zuo, David Zhen Yin
# Contact: yinzhen@stanford.edu
# Date: Oct 22, 2018

# This is the function to smooth the noise of facies posterior 3D models. 
# This function is for use after the BEL facies posterior models are corrected by 'errorCorrection_PatchDS_3D' function
# This function returns the smoothered facies posterior 3D models. 
# e.g. smoothed_posterior = errorCorrection_FaciesSmooth_3D(Models_mat, i_dim, j_dim, k_dim)

# Models_mat: the 2D array matrix that constains all the posterior facies realizations(each realization is vectorized) that need to correct, 
#            Models_mat = realization_number x total_grid_number_per_model(i_dim x j_dim x k_dim)
# i_dim, j_dim, k_dim: the i, j, k dimension of the model

import numpy as np
import random
from tqdm import tqdm
def errorCorrection_FaciesSmooth_3D(Models_mat, i_dim, j_dim, k_dim):
    
    #set up some constant variables
    realnums = len(Models_mat)
    
    Realization_Layer = k_dim
    Realization_Height = j_dim
    Realization_Width = i_dim
    
    smthd_realization = []
    
    for real_num in tqdm(range(realnums)):
        # use the statistical filter to reduce noise (isolated points)
        realization = Models_mat[real_num].reshape(Realization_Layer,Realization_Height,Realization_Width)
        realization_temporary = np.zeros((Realization_Layer,Realization_Height,Realization_Width))
        stride_x = 1
        stride_y = 1
        noisedPattern = []
        for realization_z in range(Realization_Layer):
            for realization_y in range(Realization_Height):
                for realization_x in range(Realization_Width):
                    bottom_y = max(realization_y-stride_y,0)
                    up_y = min(realization_y+stride_y+1,Realization_Height)
                    bottom_x = max(realization_x-stride_x,0)
                    up_x = min(realization_x+stride_x+1,Realization_Width)
                    noisedPattern = []
                    for sample_y in range(bottom_y,up_y):
                        for sample_x in range(bottom_x,up_x):
                            noisedPattern.append(realization[realization_z][sample_y][sample_x])
                    noisedPattern.sort();
                    sample_value = noisedPattern[int(len(noisedPattern)/2)];
                    realization_temporary[realization_z][realization_y][realization_x] = sample_value
                    
        smthd_realization.append(realization_temporary)
    smthd_realization = np.asarray(smthd_realization)
    
    return smthd_realization


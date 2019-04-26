#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: Oct 13, 2018

# This function is for contruct the data matrix for M-PCA, by concatenating the normalized d_i into a single matrix. 
# data_mat is the original data matrix that contains all the types of data variablbes (e.g. poro, perm, and etc)\
# : data_mat  = realizatons x data_samples x data_types.
# d_obs_mat is the correponding observation data matrix: d_obs_mat = obs_data_sampes x data_types
# scalar is a constant (e.g. 100, or 1000) that scales the singular value normalied  d_i
# e.g. dmat_4mixpca(d_var, 1000)
import numpy as np
def first_eigval(X):
    X = X - X.mean(axis=0)
    eig_val, eig_vec = np.linalg.eig(X.dot(X.transpose()))
    return eig_val[0].real
	
def dmat_4mixpca(data_mat, d_obs_mat, scalar):
    s =np.sqrt(first_eigval(data_mat[:,:, 0]))
    dmat_4mpca= data_mat[:,:, 0]/s
    dobsmat_4mpca = d_obs_mat[:, 0]/s
    
    for i in range(len(data_mat[0,0,:])-1):
        s = np.sqrt(first_eigval(data_mat[:,:, i+1]))
        
        dmat_4mpca = np.concatenate((dmat_4mpca, (data_mat[:,:, i+1]/s)), axis = 1)
        dobsmat_4mpca = np.concatenate((dobsmat_4mpca, (d_obs_mat[:,i+1]/s)), axis=0)
    dobsmat_4mpca = dobsmat_4mpca.reshape(-1, len(dobsmat_4mpca))
    return dmat_4mpca*scalar, dobsmat_4mpca*scalar


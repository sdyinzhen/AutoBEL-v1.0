
# coding: utf-8

# In[1]:


#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: September 12, 2018

### This function estimates and returns the sampled posterior h_c and its distribution. 
import numpy as np
## This function calculates the c_dd^* in canonical space of after lienar (PCA + CCA) tranformation
## pca_compnts: is the full data pca components
## cca_ad: is the ad matrix of cca, calculated using the above function
## dat_err_levl: the percentage of data error
## d_obs: is the original observation data, which is used as input for pca, 1xn array.
def cal_c_dd_star_pca_cca(pca_compnts, cca_ad, dat_err_levl, d_obs):
    c_dd = np.zeros((len(d_obs[0,:]), len(d_obs[0,:])), float)
    np.fill_diagonal(c_dd, (d_obs*dat_err_levl)**2)
    cdd_pca = np.diag(np.diag(np.matmul(np.matmul(pca_compnts, c_dd), pca_compnts.T)))
    cdd_star = np.diag(np.diag(np.matmul(np.matmul(cca_ad.T, cdd_pca), cca_ad)))
    #cdd_star= np.matmul(cca_ad[:,compnum-1], cca_ad[:,compnum-1].reshape(len(d_obs[0,:]), -1))*cdd_pca
    return cdd_star
###  This function calculates the c_dd^* in canonical space only with cca tranform (without pca tranform). 
def cal_c_dd_star_cca(cca_ad, dat_err_levl, d_obs):
    c_dd = np.zeros((len(d_obs[0,:]), len(d_obs[0,:])), float)
    np.fill_diagonal(c_dd, (d_obs*dat_err_levl)**2)
    ##cdd_pca = np.diag(np.diag(np.matmul(np.matmul(pca_compnts, c_dd), pca_compnts.T)))
    cdd_star = np.diag(np.diag(np.matmul(np.matmul(cca_ad.T, c_dd), cca_ad)))
    #cdd_star= np.matmul(cca_ad[:,compnum-1], cca_ad[:,compnum-1].reshape(len(d_obs[0,:]), -1))*cdd_pca
    return cdd_star


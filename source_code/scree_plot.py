#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: September 11, 2018
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[1]:


## This is the PCA scree plot function, 
## This function also report the number of PCs that preserves the assigned amount of information of the input_data. 
## input_data: orignial input matrix for PCA analys; pc_num: number of pc components. 
## keep_info_prcnt: the amount of infomation (cumulative variance ratio) to preserve after PCA. 
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
def scree_plot(input_data, keep_info_prcnt):
    X = input_data-input_data.mean(axis=0)
    eig_val, eig_vec = np.linalg.eig(X.dot(X.transpose()))
    eigval_sum = np.sum(eig_val)
    plt.figure(figsize=(5, 4))
    plt.plot(np.arange(1,len(eig_val)+1), np.cumsum(eig_val)/eigval_sum, \
             marker='o', markersize=5, linestyle = 'dashed', color='blue')
    plt.xticks(fontsize = 14)    
    plt.yticks(np.arange(0,1.01,0.1),fontsize = 14)
    plt.xlabel('Number of PCs', fontsize = 14, weight='bold')
    plt.ylabel('Cumulative_variance', fontsize = 14, weight='bold')
    plt.title('Dimension reduction of model and data', fontsize=18, loc='left', weight='bold')
    plt.grid(linestyle='dashed')
    infor_list = np.cumsum(eig_val)/eigval_sum
    infor_list = np.array(np.where(infor_list<=keep_info_prcnt/100))[0]
    return infor_list[-1]+1
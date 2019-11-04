#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: September 11, 2018

import numpy as np
import matplotlib.pyplot as plt
def scree_plot(input_data, data_name, keep_info_prcnt, plotORnot):
    '''This is the PCA scree plot function, 
       This function also report the number of PCs that preserves the assigned amount of information of the input_data. 
       input_data: orignial input matrix for PCA analys; pc_num: number of pc components. 
       data_name: name of the input data, e.g. 'model', 'data'
       keep_info_prcnt: the amount of infomation (cumulative variance ratio) to preserve after PCA. 
       plotORnot: 'plot' - will produce the screet plot, 'not' - no plot, only return the pcnum for the required prcnt variance.
       '''
    X = input_data-input_data.mean(axis=0)
    eig_val, eig_vec = np.linalg.eig(X.dot(X.transpose()))
    eigval_sum = np.sum(eig_val)
    infor_list = np.cumsum(eig_val)/eigval_sum
    infor_list = np.array(np.where(infor_list<=keep_info_prcnt/100))[0]
    if plotORnot == 'plot':
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1,len(eig_val)+1), np.cumsum(eig_val)/eigval_sum, \
                 marker='o', markersize=5, linestyle = 'dashed', color='blue')
        plt.xticks(fontsize = 14)    
        plt.yticks(np.arange(0,1.01,0.1),fontsize = 14)
        plt.xlabel('number of PCs', fontsize = 12, weight='bold')
        plt.ylabel('cumulative variance ratio', fontsize = 12, weight='bold')
        plt.title('Dimension reduction of ' + data_name +' - PCA scree plot', fontsize=18, loc='left', style='italic')
        plt.grid(linestyle='dashed')
        plt.axhline(y=keep_info_prcnt/100, linewidth=2, color='red', linestyle='--')

        plt.axvline(x=infor_list[-1]+1, linewidth=2, color='red', linestyle='--')
        plt.text(infor_list[-1]-2, -0.06, str(infor_list[-1]+1), fontsize=14, weight='bold', color='red')

    return infor_list[-1]+1
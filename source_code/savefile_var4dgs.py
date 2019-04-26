
# coding: utf-8

# In[ ]:


# Export the PCs of the d or h variables as DGSA ParameterFile
## Author: David Zhen Yin   
## Contact: yinzhen@stanford.edu
## Date: September 11, 2018

import numpy as np

# e.g. savefile_var4dgsa(d_pc_scores, 9, 'DGSA/for_redistribution_files_only/Y_data_poro.txt')
## pc_score: the whole array of the PCA scores
## pc_num: the total number of the PC componets to export
## savefilename: the directory and name of the exported files
def savefile_var4dgsa(pc_score, pc_num, savefilename):
    headers = []
    for i in range(pc_num):
        headers.append('PC_'+str(i+1))
    headers.append('#')
    headers = ' '.join(headers)    
    np.savetxt(savefilename, pc_score[:,:pc_num], fmt='%f',               comments='', header=headers)


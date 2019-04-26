
# coding: utf-8

# In[3]:


#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: September 11, 2018

## this function filters out the sensitive h(prediction) component numbers
##sa_coefilename: the directory and name of the sa coefficient file
##totl_paranum: the total number of the parameters in the SA
##thresh_values: Threshold value for the sensitive parameters

import numpy as np
import pandas as pd
def dgsa_sensitive_h_pcs(sa_coefilename, totl_paranum, thresh_values):
    headers = []
    for i in range(totl_paranum):
        headers.append(i+1)
    ave_sa = np.loadtxt(sa_coefilename)
    ave_sa = pd.DataFrame({'SA_coeff':ave_sa[:totl_paranum]/thresh_values,'sensitive_pcnum': headers})
    ave_sa = ave_sa.sort_values('SA_coeff',ascending=False)
    ave_sa = ave_sa.reset_index(drop=True)
    sensitive_pcs = ave_sa[ave_sa["SA_coeff"] >1] 
    return sensitive_pcs["sensitive_pcnum"].values


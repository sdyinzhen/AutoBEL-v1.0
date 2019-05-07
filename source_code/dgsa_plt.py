
# coding: utf-8

# In[2]:


#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: September 11, 2018

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def dgsa_plt(SA_dataframe, plt_paranum):
    
    '''
    Distance based Generalized Sensitivity Analysis method 
    Args:
        SA_dataframe: (pd.DataFrame) StandardizedSensitivity data frame from DGSA
        plt_paranum: (int) Numbers of bar plot, from the most sensitivity parameter
    Output:
        Pareto Plot for SA
    '''

    SA_dataframe = SA_dataframe.sort_values(by=0, ascending=False)

    #############Define the colour bar #########
    pltdata = SA_dataframe.values[:plt_paranum,0]
    
    mask= pltdata <= 1
    ### red
    colors = np.asarray([[1, 0, 0, 1.0]]*plt_paranum)
    ###blue
    colors[mask] = [0, 0, 1, 1.0]
    #### red
    colors[:,3]= (abs(pltdata-max(pltdata[mask]))                    /abs(max(pltdata)-max(pltdata[mask])))**0.3
    #### blue
    colors[:,3][mask]= (np.sqrt(colors[mask][:,3]/(max(colors[mask][:,3]))))**2

    plt.figure(figsize=(12,5))
    plt.bar(np.arange(plt_paranum), pltdata, width=0.85, color = colors,             edgecolor='k', linewidth=0.5,)
    plt.xlim([-0.6,plt_paranum])
    if max(pltdata[:plt_paranum])>3:
        plt.ylim([min(pltdata[:plt_paranum]-0.2), 3])
    else:
        plt.ylim([min(pltdata[:plt_paranum]-0.2), max(pltdata[:plt_paranum]+0.05)])
    plt.xticks(np.arange(plt_paranum), SA_dataframe.index[:plt_paranum], fontsize=13, rotation =70)
    plt.yticks(fontsize=16)
    plt.axhline(y=1, linestyle = '--', c ='k')
    plt.ylabel('sensitivity to data', fontsize=16)
    plt.title('Global Sensitivity of model to data (calculated by DGSA)', fontsize=18, loc='left', style='italic')
    plt.xlabel('model parameters', fontsize=16)
	
	

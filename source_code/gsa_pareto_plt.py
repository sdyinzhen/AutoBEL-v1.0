#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: August 17, 2019

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def gsa_pareto_plt(GSA_dataframe, response_name):
    
    '''
    Pareto plot to display sensitivity analysis results. 
    Args:
        GSA_dataframe: main sensitivity of parameters measured from DGSA, (pd.DataFrame)data frame. 
        response_name: the name of reponses attribute, type: string
    Output:
        Pareto Plot for the GSA
    '''
    
    GSA_dataframe = GSA_dataframe.sort_values(by=0, ascending=False)

    #############Define the colour bar #########
    n_parameter = len(GSA_dataframe)
    n_sensitive = len(GSA_dataframe.values[GSA_dataframe.values >=1])
    if n_parameter>=20:
        if n_parameter - n_sensitive > 6: 
            n_plt_para = len(GSA_dataframe.values[GSA_dataframe.values >=1])+6
        else:
            n_plt_para = n_parameter
    else:
        n_plt_para = n_parameter
        
    pltdata = GSA_dataframe.values[:n_plt_para,0]
    
    mask= pltdata <= 1
    ### red
    colors = np.asarray([[1, 0, 0, 1.0]]*n_plt_para)
    ###blue
    colors[mask] = [0, 0, 1, 1.0]
    #### red
    colors[:,3]= (abs(pltdata-max(pltdata[mask]))/abs(max(pltdata)-max(pltdata[mask])))**0.3
    #### blue
    colors[:,3][mask]= (np.sqrt(colors[mask][:,3]/(max(colors[mask][:,3]))))**2

    plt.figure(figsize=(12,5))
    plt.bar(np.arange(n_plt_para), pltdata, width=0.85, color = colors,edgecolor='k', linewidth=0.5,)
    plt.xlim([-0.6,n_plt_para])
    if max(pltdata[:n_plt_para])>3:
        plt.ylim([min(pltdata[:n_plt_para]-0.2), 3])
    else:
        plt.ylim([min(pltdata[:n_plt_para]-0.2), max(pltdata[:n_plt_para]+0.05)])
    plt.xticks(np.arange(n_plt_para), GSA_dataframe.index[:n_plt_para], fontsize=15, rotation =70)
    plt.yticks(fontsize=16)
    plt.axhline(y=1, linestyle = '--', c ='k')
    plt.ylabel('Sensitivity measurements', fontsize=16)
    plt.title('Global sensitivity of parameters to "' + response_name+ '" (by DGSA)', fontsize=18, loc='left', style='italic')
    plt.xlabel('Parameters', fontsize=17)
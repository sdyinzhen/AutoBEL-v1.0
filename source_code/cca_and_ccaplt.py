#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: September 11, 2018
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[14]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cross_decomposition import CCA

## This function is for the CCA of data variable d_var and prediction variable h_var. 
## This function returns the d_c, h_c, which are d_var and h_var in the canonical space. and also returns the coressponding Ah, Ad. 
## h_var: the array of prediction variable
## d_var: the array of data variable
## components_num: the number of canonical components. 
def cca_d_h(d_var, h_var, components_num):
	cca=CCA(n_components=components_num, scale=True, max_iter=2000)
	cca.fit(d_var, h_var)
	d_c,h_c=cca.transform(d_var, h_var)  
	ah = np.linalg.inv((h_var.T).dot(h_var)).dot(h_var.T).dot(h_c)
	ad = np.linalg.inv((d_var.T).dot(d_var)).dot(d_var.T).dot(d_c)

	return d_c, h_c, ad, ah

def cca_plt(d_cca, h_cca, d_obs_cca, com_num):
    '''
    This function produces a single regression plot between d_c and h_c,with the d_obs_c moarked
        d_cca: the data variable array in canonical space
        h_cca: the prediction variable array in canonical space
        d_obs_cca: the d_obs variable array in canonical space
        com_num: provide the number of the cca component for plot. 
    
    '''
    plotdata = np.column_stack([d_cca[:,com_num-1],h_cca[:,com_num-1]])
    plotdata = pd.DataFrame(plotdata, columns=['d (CCA component '+str(com_num)+')',\
                                               'm (CCA component '+str(com_num)+')'])
    sns.set(style="white",palette='deep', font_scale=1.5)
    sns.set_style("ticks", {"xtick.major.size": 4, "ytick.major.size": 4})
    fig=sns.jointplot(x='d (CCA component '+str(com_num)+')', y='m (CCA component '+str(com_num)+')', \
                      data = plotdata, kind="reg",\
                      scatter_kws={"s": 45, "linewidths":0.5, "edgecolor":'k', "color":'royalblue','alpha':1},    
                      line_kws={"linewidth": 1.2, "color": "k",},\
                      annot_kws={'stat':"r"},\
                      marginal_kws={'hist':None,'kde':None})
    fig.ax_marg_x.set_axis_off()
    fig.ax_marg_y.set_axis_off()
    fig.fig.set_size_inches(9,5)
    plt.axvline(x=d_obs_cca[:,com_num-1], linewidth=3, linestyle= '--',alpha=0.7, c='red')
    plt.title('Canonical Corrleation Analysis betweeen model and data', fontsize=18, loc='left', style='italic')

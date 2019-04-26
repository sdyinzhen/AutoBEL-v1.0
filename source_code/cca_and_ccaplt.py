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
	cca=CCA(n_components=components_num, scale=True)
	cca.fit(d_var, h_var)
	d_c,h_c=cca.transform(d_var, h_var)  
	ah = np.linalg.inv((h_var.T).dot(h_var)).dot(h_var.T).dot(h_c)
	ad = np.linalg.inv((d_var.T).dot(d_var)).dot(d_var.T).dot(d_c)

	return d_c, h_c, ad, ah

## This function produces a single regression plot between d_c and h_c,with the d_obs_c moarked
## d_cca: the data variable array in canonical space
## h_cca: the prediction variable array in canonical space
## d_obs_cca: the d_obs variable array in canonical space
## com_num: provide the number of the cca component for plot. 
def cca_plt(d_cca, h_cca, d_obs_cca, com_num):
    ### com stands for number cca component number
    plotdata = np.column_stack([d_cca[:,com_num-1],h_cca[:,com_num-1]])
    plotdata = pd.DataFrame(plotdata, columns=['d (CCA component_'+str(com_num)+' score)',                                            'h (CCA component_'+str(com_num)+' score)'])
    sns.set(style="white",palette='deep', font_scale=1.7)
    sns.set_style("ticks", {"xtick.major.size": 4, "ytick.major.size": 4})
    fig=sns.jointplot(x='d (CCA component_'+str(com_num)+' score)', y='h (CCA component_'+str(com_num)+' score)',                      data = plotdata, kind="reg",                       scatter_kws={"s": 45, "linewidths":0.5, "edgecolor":'k', "color":'royalblue','alpha':1},    
                      line_kws={"linewidth": 1.2, "color": "k",},\
                      annot_kws={'stat':"r"},\
                      marginal_kws={'hist':None,'kde':None})
    fig.ax_marg_x.set_axis_off()
    fig.ax_marg_y.set_axis_off()
    fig.fig.set_size_inches(9,5)
    plt.axvline(x=d_obs_cca[:,com_num-1], linewidth=3, linestyle= '--',alpha=0.7, c='red')
    plt.title('Canonical Corrleation Analysis', fontsize=18, loc='left', weight='bold')


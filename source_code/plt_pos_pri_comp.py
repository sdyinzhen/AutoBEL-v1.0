
# coding: utf-8

# In[4]:


#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: September 12, 2018

# this function return the scatter plot of prior and posterior components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plt_pos_pri_comp(x_compnum, y_compnum, h_pri, h_post):
    '''
    this function return the scatter plot of prior and posterior components
    x_compnum, y_compnum: component number of the prior/posterior in x and y axis. 
    h_pri: the whole prior array 
    h_post: the whole posterior array
    '''
    plotdata_pri = np.column_stack([h_pri[:,x_compnum-1], h_pri[:,y_compnum-1]])
    plotdata_pri = pd.DataFrame(plotdata_pri, columns=['PC'+str(x_compnum), 'PC'+str(y_compnum)])
    plotdata_pos = np.column_stack([h_post[:,x_compnum-1], h_post[:,y_compnum-1]])
    plotdata_pos = pd.DataFrame(plotdata_pos, columns=['PC'+str(x_compnum), 'PC'+str(y_compnum)])

    sns.set(style="white",palette='deep', font_scale=1.6)
    sns.set_style("ticks", {"xtick.major.size": 4, "ytick.major.size": 4})
    fig=sns.jointplot(x='PC'+str(x_compnum), y='PC'+str(y_compnum), data = plotdata_pri,\
                      kind="kde", stat_func=None, color='lightskyblue')
    fig.ax_marg_x.set_axis_off()
    fig.ax_marg_y.set_axis_off()
    fig.plot_joint(plt.scatter,color='royalblue',s=65, linewidths=0.5,edgecolor='k',alpha=0.95)

    plt.scatter(h_post[:,x_compnum-1],h_post[:,y_compnum-1],c='red', s=60, linewidths=0.5,edgecolor='k')
    plt.title('Plot of prior and posterior model PC scores',  fontsize=18, loc='left', style='italic')
    fig.fig.set_size_inches(8,5)
    #sns.plt.title('Transform back to PCA space', fontsize=18, loc='left', weight='bold')    
    #fig.ax_joint.set_xlim(np.min(h[:,x_compnum-1])*1.3,np.max(h[:,x_compnum-1])*1.2)
    #fig.ax_joint.set_ylim(np.min(h[:,y_compnum-1])*1.3,np.max(h[:,y_compnum-1])*1.2)

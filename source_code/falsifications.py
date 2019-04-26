#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: September 11, 2018


## This is the PCA scree plot function
## input_data: orignial input matrix for PCA analys; pc_num: number of pc components. 
## fig_w, fig_h: width & height of the plot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
def fals_1Dplt(d_var_pcsc, d_obs_pcsc, pc_num):
    pc_name=[]
    for i in range(pc_num):
        pc_name.append('PC'+str(i+1))
    
    d_scores = pd.DataFrame(d_var_pcsc[:,:pc_num], columns=pc_name)        
    dob_scores = pd.DataFrame(d_obs_pcsc[:,:pc_num],columns=pc_name)
    plt.figure(figsize=(8,4))
    
    sns.set(font_scale=1.4)
    sns.violinplot(data=d_scores, inner=None, color=".8", width=1)
    sns.stripplot(data=d_scores, jitter=True, size=4, linewidth=1)
    sns.stripplot(data=dob_scores, color='red',size=9, linewidth=1, marker="D", edgecolor='y')
    sns.reset_orig()


def fals_2Dplt(d_var_pcsc, d_obs_pcsc, pcx, pcy):
    
    plotdata = np.column_stack([d_var_pcsc[:,pcx-1],d_var_pcsc[:,pcy-1]])
    plotdata = pd.DataFrame(plotdata, columns=['PC'+str(pcx), 'PC'+str(pcy)])
    sns.set(style="white",palette='deep', font_scale=1.4)
    sns.set_style("ticks", {"xtick.major.size": 4, "ytick.major.size": 4})
    
    fig=sns.jointplot(x='PC'+str(pcx), y='PC'+str(pcy),data = plotdata, kind="kde", stat_func=None, color='lightskyblue')
    fig.plot_joint(plt.scatter,color='royalblue',s=35, linewidths=0.5,edgecolor='k')
    fig.fig.set_size_inches(5,4)
    
    fig.ax_joint.set_xlim(np.min(d_var_pcsc[:,pcx-1])*1.2,np.max(d_var_pcsc[:,pcx-1])*1.1)
    fig.ax_joint.set_ylim(np.min(d_var_pcsc[:,pcy-1])*1.3,np.max(d_var_pcsc[:,pcy-1])*1.1)
    
    plt.scatter(d_obs_pcsc[:,pcx-1],d_obs_pcsc[:,pcy-1], color='red',s=80, marker='D', linewidths=1,edgecolor='yellow')
    sns.reset_orig()


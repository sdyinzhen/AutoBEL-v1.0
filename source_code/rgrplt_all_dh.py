#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: September 11, 2018
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[14]:


## This function is for the multiple regression plot between data and h
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or             isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


## x_data: the data arrays as x-axis
## y_predict" the prediction arrays as y-axis
## d_pc_num: the total number of data PC, it will also be the row number of the plots
## h_pc_num: the total number of predction h PC, it will also be the column number of the plots
def rgrplt_all_dh(x_data, y_predict, d_obs, d_pc_num, h_pc_num):
    #import matplotlib.gridspec as gridspec
    sfg=SeabornFig2Grid
    ax=plt.figure(figsize=( 5.5*len(h_pc_num), 4.8*len(d_pc_num)))
    gs = gridspec.GridSpec(len(d_pc_num), len(h_pc_num))
          
    count=0
    
    for pcx in tqdm(d_pc_num):
        for pcy in h_pc_num:
            plotdata = np.column_stack([x_data[:,pcx-1],y_predict[:,pcy-1]])
            plotdata = pd.DataFrame(plotdata, columns=['d PC'+str(pcx), 'h PC'+str(pcy)])
            
            #ax.add_subplot(len(d_pc_num), len(h_pc_num), count)
            
            sns.set(style="white",palette='deep', font_scale=1.4)
            sns.set_style("ticks", {"xtick.major.size": 4, "ytick.major.size": 4})
            fig=sns.jointplot(x='d PC'+str(pcx), y='h PC'+str(pcy),data = plotdata,                               kind="kde", stat_func=None, color='lightskyblue')
            fig.plot_joint(plt.scatter,color='royalblue',s=35, linewidths=0.5,edgecolor='k')
            
            fig.fig.set_size_inches(3.8, 3.3)
            
            #plt.axvline(d_obs[:,pcx-1]*0.9, d_obs[:,pcx-1]*1.1, alpha=0.7,facecolor='r', edgecolor='y')
            plt.axvline(x=d_obs[:,pcx-1],color='red',linewidth=2)
            
            fig.ax_joint.set_xlim(np.min(x_data[:,pcx-1])-abs(np.min(x_data[:,pcx-1]))*0.1,np.max(x_data[:,pcx-1])+abs(np.max(x_data[:,pcx-1]))*0.1)
            fig.ax_joint.set_ylim(np.min(y_predict[:,pcy-1])*1.1,np.max(y_predict[:,pcy-1])*1.1)
            
            sfg(fig, ax, gs[count])
            
            count=count+1
    
    plt.tight_layout(w_pad=3, h_pad=2.5, rect=[0.02, 0, 1.2, 1.2])


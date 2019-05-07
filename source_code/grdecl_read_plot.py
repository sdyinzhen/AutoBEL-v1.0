#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: September 11, 2018
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[1]:


# grdecl_read_plot defines the function to read & plot multiple realizations of GRDECL exports (2D layer).
# real_num: the number of realizations. 
# file_name: the petrel exporeted grdecl file name, e.g.: Facies_1.GRDECL
# i_dim, j_dim: i and j dimensions of the exported grdecl. 
# plot_num: the number of realizations to plot, currently the plot_num must < 16.
# GRDECL export setting in Petrel: 
# Local coord system, without Mapaxes, User define cell origin, cell origin at(I=0, J=max J,K), Traverse first along I, then along J

# this returns grid_data in grid_data[k_dim, j_dim, i_dim] formate
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def grdecl_read(file_name, i_dim, j_dim, k_dim):
       
    ####### MAIN FUNCTION ######
    
    ## import the GRDECl files by ignore the first 15 header lines, and then unwrap the unified grid values, then plot them
    new_data=[]
    f = open(file_name, 'r')
    data= f.readlines()[15:]
    f.close()
    ## convert the original data into to a 3D array cube with i rows, j columns, and k layers. 
    for row in data:
        for a in row.split():
            index = a.find('*')        
            if index >=0:
                cell_num = int(a[:index])
                for i in range(cell_num):
                    val=float(a[index+1:])
                    new_data.append(val)
            elif a !='/':
                val=float(a)
                new_data.append(val)
    new_data = np.asarray(new_data)
    grid_data =new_data.reshape(k_dim,j_dim,i_dim)
    
    # The grid_data will be grid_data[k_dim][j_dim][i_dim]
    return grid_data
	
	
def grdecl_plot(file_pre, i_dim, j_dim, k_dim, layernum, pro_type):
    fig=plt.figure(figsize=(15,14))
    count = 1
    if pro_type == 'facies':
        
        for realnum in tqdm(range(12)):
            grid_data = grdecl_read(file_pre + str(realnum+1) + '.GRDECL',200,100,75)        
            plot=fig.add_subplot(3, 4, count)
            count = count+1
            prop_mean = format(np.mean(grid_data),'.4f')
            plt.imshow(grid_data[:][:][layernum],cmap='viridis_r') # for poro
            plt.xticks(fontsize = 13)
            plt.yticks(fontsize = 13)
            plt.title('prior model sample #'+str(count-1), fontsize=14, style='italic')
            plot.set_xlabel('Realization No. ' + str(realnum), fontsize = 14)
       
    else:
        for realnum in tqdm(range(12)):
            grid_data = grdecl_read(file_pre + str(realnum+1) + '.GRDECL',200,100,75)         
            plot=fig.add_subplot(3, 4, count)
            count = count+1

            prop_mean = format(np.mean(grid_data),'.4f')
            plot.set_xlabel('whole field "' + pro_type + '" = ' + str(prop_mean), fontsize = 14)
            c_max = np.max(grid_data[layernum])*1.05
            c_min = np.min(grid_data[layernum])

            if pro_type == 'Sw':
                plt.imshow(grid_data[layernum],cmap='jet_r', \
                           vmin=c_min,vmax=c_max )
            else:
                plt.imshow(grid_data[layernum],cmap='jet', \
                           vmin=c_min,vmax=c_max*1.05)                
            plt.xticks(fontsize = 13)
            plt.yticks(fontsize = 13)
            plt.title('prior model sample #'+str(count-1), fontsize=14, style='italic')

#             plt.colorbar(fraction = 0.02)
            plt.colorbar(fraction = 0.02, ticks=np.around([c_min*1.1, c_max], decimals=1))
    plt.subplots_adjust(top=0.55, bottom=0.08, left=0.10, right=0.95, hspace=0.15,
                    wspace=0.35)

	
    #t = ("Prior model samples")
	#plt.figure(figsize=(3, 0.1))
    #plt.text(0, 0, t, style='normal', ha='center', fontsize=16, weight = 'bold')
    #plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
    #plt.show()
	

def grdecl_col_plot(file_pre, first_realnum, last_realnum, pstep, colnum, pro_type):
    plot_num = int((last_realnum-first_realnum+1)/pstep)    
    fig_row = int((plot_num+3)/4)
    fig=plt.figure(figsize=(fig_row*8,20))
    count = 1
    if pro_type == 'f':
        
        for realnum in tqdm(range(first_realnum, last_realnum, pstep)):
            grid_data = grdecl_read(file_pre + str(realnum+1) + '.GRDECL',200,100,75)        
            plot=fig.add_subplot(fig_row, 4, count)
            count = count+1
            #plot=fig.add_subplot(3,plot_num/3,count)
            #prop_mean = format((1-np.mean(grid_data)/3-0.01)*100,'.2f')
            #plot.set_xlabel('Sand Fraction = ' + str(prop_mean)+'%', fontsize = 14)
            #plt.imshow(grid_data,cmap='viridis_r',extent=[0,2500,0,2500]) # for facies
            prop_mean = format(np.mean(grid_data),'.4f')
            plt.imshow(grid_data[:,colnum,:],cmap='viridis_r') # for poro
            plt.xticks(fontsize = 13)
            plt.yticks(fontsize = 13)
            #plot.set_xlabel('whole field fac mean= ' + str(prop_mean), fontsize = 14)
            plot.set_xlabel('Realization No. ' + str(realnum), fontsize = 14)

            #plt.axvline(x=77, linewidth=1.2, c='r')
            #plt.axvline(x=143, linewidth=1.2, c='r')

       
    else:
        for realnum in tqdm(range(first_realnum, last_realnum, pstep)):
            grid_data = grdecl_read(file_pre + str(realnum+1) + '.GRDECL',200,100,75)         
            plot=fig.add_subplot(fig_row, 4, count)
            count = count+1
            #plot=fig.add_subplot(3,plot_num/3,count)
            prop_mean = format(np.mean(grid_data),'.4f')
            plot.set_xlabel('whole field "thc_coeff" = ' + str(prop_mean), fontsize = 14)
            #plt.imshow(grid_data,cmap='viridis_r',extent=[0,2500,0,2500]) # for facies
            plt.imshow(grid_data[:,colnum,:],cmap='jet',extent=[0,50000,0,25000], vmin=np.min(grid_data[:,colnum,:]),vmax=np.max(grid_data[:,colnum,:])*1.05) # for poro            
            plt.xticks(fontsize = 13)
            plt.yticks(fontsize = 13)
            #print(realnum)
    
    #plt.colorbar()     
    plt.subplots_adjust(0.04, 0.04, 0.96, 0.44, 0.2, 0.30)


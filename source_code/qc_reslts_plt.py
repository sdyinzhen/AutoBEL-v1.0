#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: Apri 03, 2019


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def m_ensampl_plt(m_smpls_pos, m_smpls_pri, layernum, i_dim, j_dim, k_dim, dobs):
    '''    
    Plot the the ensemble mean and variance of prior and posterior
    Args:
        m_smpls_pos: the name of posterior model sample matrix, N_realizations x Grid_dims
        m_smpls_pri: the name of prior model sample matrix, N_realizations x Grid_dims
        layernum: which layer to plot
        dobs: the well data observation: 4 x well_number: row1 = x, row2=j, row3 = k, row4 = value
        
    Output:
        Pareto Plot for SA
    '''

    plt.figure(figsize=(12.5,6.2))
    
    plt.subplot(221)
    m_ens_mean = np.mean(m_smpls_pos[:,j_dim*i_dim*(layernum-1):j_dim*i_dim*layernum], axis=0)
    
    plt.imshow(m_ens_mean.reshape(j_dim,i_dim), \
               extent=[0,50000,25000,0], cmap='jet', \
               vmax=np.max(m_ens_mean), vmin=np.min(m_ens_mean))
    
    plt.colorbar(fraction = 0.02)
    plt.scatter(dobs[0, len(dobs[0,:])*(layernum-1):len(dobs[0,:])*(layernum)]*250, \
                dobs[1, len(dobs[0,:])*(layernum-1):len(dobs[0,:])*(layernum)]*250, \
                c= dobs[3, len(dobs[0,:])*(layernum-1):len(dobs[0,:])*(layernum)], \
                cmap='jet', edgecolors=(0, 0, 0), linewidth =0.9, s=80, vmax=np.max(m_ens_mean), vmin=np.min(m_ens_mean))
    plt.tick_params(labelsize=13)
    plt.title('Ensemble Mean of Posterior', fontsize = 15, style='italic')

    plt.subplot(222)
    m_ens_mean_pri = np.mean(m_smpls_pri[:,j_dim*i_dim*(layernum-1):j_dim*i_dim*layernum], axis=0)
    
    plt.imshow(m_ens_mean_pri.reshape(j_dim,i_dim), \
               extent=[0,50000,25000,0], cmap='jet', \
               vmax=np.max(m_ens_mean), vmin=np.min(m_ens_mean))
    
    plt.colorbar(fraction = 0.02)
    
    plt.scatter(dobs[0, len(dobs[0,:])*(layernum-1):len(dobs[0,:])*(layernum)]*250, \
                dobs[1, len(dobs[0,:])*(layernum-1):len(dobs[0,:])*(layernum)]*250, \
                c= dobs[3, len(dobs[0,:])*(layernum-1):len(dobs[0,:])*(layernum)], \
                cmap='jet', edgecolors=(0, 0, 0), linewidth =0.9, s=80, \
                vmax=np.max(m_ens_mean), vmin=np.min(m_ens_mean))
    plt.tick_params(labelsize=13)
    plt.title('Ensemble Mean of Prior', fontsize = 15, style='italic')
  
    
    plt.subplot(223)
    m_ens_var =  np.var(m_smpls_pos[:,j_dim*i_dim*(layernum-1):j_dim*i_dim*layernum], axis=0)
    plt.imshow(m_ens_var.reshape(j_dim,i_dim), extent=[0,50000,25000,0], cmap='bwr', \
               vmin = np.min(m_ens_var), vmax=np.max(m_ens_var))
    plt.colorbar(fraction = 0.02)
    plt.scatter(dobs[0, len(dobs[0,:])*(layernum-1):len(dobs[0,:])*(layernum)]*250, \
                dobs[1, len(dobs[0,:])*(layernum-1):len(dobs[0,:])*(layernum)]*250, \
                color='white', edgecolors=(0, 0, 0), linewidth =1.2, s=80, alpha=0.7)
    plt.tick_params(labelsize=13)
    plt.title('Ensemble Variance of Posterior', fontsize = 15, style='italic')
    

    plt.subplot(224)
    m_ens_var_pri =  np.var(m_smpls_pri[:,j_dim*i_dim*(layernum-1):j_dim*i_dim*layernum], axis=0)
    plt.imshow(m_ens_var_pri.reshape(j_dim,i_dim), extent=[0,50000,25000,0], cmap='bwr', \
               vmin = np.min(m_ens_var), vmax=np.max(m_ens_var))
    plt.colorbar(fraction = 0.02)
    plt.scatter(dobs[0, len(dobs[0,:])*(layernum-1):len(dobs[0,:])*(layernum)]*250, \
                dobs[1, len(dobs[0,:])*(layernum-1):len(dobs[0,:])*(layernum)]*250, \
                color='white', edgecolors=(0, 0, 0), linewidth =1.2, s=80, alpha=0.7)
    plt.tick_params(labelsize=13)
    plt.title('Ensemble Variance of Prior', fontsize = 15, style='italic')    
    
    plt.tight_layout()
    
    return 


def m_sampls_plt(m_smpls,m_type, i_dim, j_dim, k_dim):
    '''
    Plot the 
    Args:
        m_smpls: (str) the name of model sample matrix, N_realizations x Grid_dims
        first_realnum: the first realization number of the model
        last_realnum: the last realization number of the model
        pstep: plot per every pstep.
        layernum: which layer to plot
        m_type: type of the model, e.g: facies, poro, perm
        
    Output:
        Pareto Plot for SA
    '''
    plot_num = 12    
    fig_row = 3
    layernum = 1
    fig=plt.figure(figsize=(15,14))
    count = 1
    if m_type == 'facies':
        
        for realnum in tqdm(range(12)):
            grid_data = plt.imshow(m_smpls[realnum, :].reshape(k_dim,j_dim,i_dim))
            plot=fig.add_subplot(fig_row, 4, count)
            count = count+1
            prop_mean = format(np.mean(grid_data),'.4f')
            plt.imshow(grid_data[layernum-1],cmap='viridis_r',extent=[0,50000,0,25000]) # for poro
            plt.xticks(fontsize = 13)
            plt.yticks(fontsize = 13)
            plt.title('posterior model sample #'+str(count-1), fontsize=14, style='italic')
            plot.set_xlabel('Realization No. ' + str(realnum), fontsize = 14)
       
    else:
        for realnum in tqdm(range(12)):
            grid_data = m_smpls[realnum,:].reshape(k_dim, j_dim, i_dim)      
            plot=fig.add_subplot(fig_row, 4, count)
            count = count+1
            #plot=fig.add_subplot(3,plot_num/3,count)
            prop_mean = format(np.mean(grid_data),'.4f')
            plot.set_xlabel('whole field "' + m_type + '" = ' + str(prop_mean), fontsize = 14)
            plt.title('posterior model sample #'+str(count-1), fontsize=14, style='italic')
            if m_type == 'Sw':
                plt.imshow(grid_data[layernum-1],cmap='jet_r',extent=[0,50000,0,25000], \
                           vmin=np.min(grid_data[layernum-1]),vmax=np.max(grid_data[layernum-1])*1.05)
            else:
                plt.imshow(grid_data[layernum-1],cmap='jet',extent=[0,50000,0,25000], \
                           vmin=np.min(grid_data[layernum-1]),vmax=np.max(grid_data[layernum-1])*1.05)                
            plt.xticks(fontsize = 13)
            plt.yticks(fontsize = 13)
            #print(realnum)
    
    plt.colorbar(fraction = 0.03)     
    plt.subplots_adjust(top=0.55, bottom=0.08, left=0.10, right=0.95, hspace=0.15,
                    wspace=0.35)
    return
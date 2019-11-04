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

def mc_samples_plot(model_ndarray, m_name, m_type, pri_post, i_dim, j_dim, k_dim, layernum):
    '''
    Plot the 1st to 12th monte carlo model samples. 
    Args:
        model_ndarray: (str) the ndarray of monte carlo model samples, N_realizations x Grid_dims
        i_dim, j_dim, k_dim: x, y, z dimensions of one model realization. 
        layernum: which layer to show
        m_type: type of the model, 1 - continous; 2 - categorical
        
    Output:
        Pareto Plot for SA
    '''
    fig=plt.figure(figsize=(15,14))
    count = 1
    k_dim = int(len(model_ndarray[1])/(i_dim*j_dim))
    if m_type == 2: # 2 is for cate gorical models; 1 is for continous models. 
        
        for realnum in range(12):
            if count  == 12:
                plot=fig.add_subplot(3, 4, count)
                plt.text(0.1, 0.48, '...', fontsize=50)
                plt.text(0.0, 0.6, 'Total '+str(len(model_ndarray))+' samples', fontsize=16, style='italic')
                plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
                count = count + 1
            else:
                grid_data = model_ndarray[realnum].reshape(k_dim, j_dim, i_dim)       
                plot=fig.add_subplot(3, 4, count)
                count = count+1
                prop_mean = format(np.mean(grid_data),'.4f')
                plt.imshow(grid_data[layernum-1],cmap='viridis_r') # for poro
                plt.xticks(fontsize = 13)
                plt.yticks(fontsize = 13)
                plt.title(pri_post +' ' + m_name+ ' model #'+str(count-1), fontsize=14, style='italic')

       
    else:
        for realnum in range(12):
            if count  == 12:
                plot=fig.add_subplot(3, 4, count)
                plt.text(0.1, 0.48, '...', fontsize=50)
                plt.text(0.0, 0.6, 'Total ' + str(len(model_ndarray))+' samples', fontsize=16, style='italic')
                plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
                count = count + 1
            else:
            
                grid_data = model_ndarray[realnum].reshape(k_dim, j_dim, i_dim)         
                plot=fig.add_subplot(3, 4, count)
                count = count+1

                prop_mean = format(np.mean(grid_data),'.4f')
                plot.set_xlabel('average "' + m_name + '" = ' + str(prop_mean), fontsize = 14)
                c_max = np.max(grid_data[layernum-1])*1.05
                c_min = np.min(grid_data[layernum-1])

                plt.imshow(grid_data[layernum-1],cmap='jet', \
                               vmin=c_min,vmax=c_max*1.05)                
                plt.xticks(fontsize = 13)
                plt.yticks(fontsize = 13)
                plt.title(pri_post +' ' +m_name+ ' model #'+str(count-1), fontsize=14, style='italic')

#             plt.colorbar(fraction = 0.02)
                plt.colorbar(fraction = 0.02, ticks=np.around([c_min*1.1, c_max], decimals=1))
    plt.subplots_adjust(top=0.55, bottom=0.08, left=0.10, right=0.95, hspace=0.15,
                    wspace=0.35)
    return
#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: Apri 03, 2019


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def m_ensampl_plt(m_smpls_pos, m_smpls_pri, model_name, layernum, i_dim, j_dim, k_dim, dobs):
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
               cmap='jet', \
               vmax=np.max(m_ens_mean), vmin=np.min(m_ens_mean))
    
    plt.colorbar(fraction = 0.02)
    plt.scatter(dobs[len(dobs)*(layernum-1):len(dobs)*(layernum), 0], \
                100-dobs[len(dobs)*(layernum-1):len(dobs)*(layernum), 1], \
                c= dobs[len(dobs)*(layernum-1):len(dobs)*(layernum), 3], \
                cmap='jet', edgecolors=(0, 0, 0), linewidth =0.9, s=80, vmax=np.max(m_ens_mean), vmin=np.min(m_ens_mean))
    plt.tick_params(labelsize=13)
    plt.title('Ensemble Mean of Posterior "' + model_name + '"', fontsize = 15, style='italic')

    plt.subplot(222)
    m_ens_mean_pri = np.mean(m_smpls_pri[:,j_dim*i_dim*(layernum-1):j_dim*i_dim*layernum], axis=0)
    
    plt.imshow(m_ens_mean_pri.reshape(j_dim,i_dim), \
               cmap='jet', \
               vmax=np.max(m_ens_mean), vmin=np.min(m_ens_mean))
    
    plt.colorbar(fraction = 0.02)
    
    plt.scatter(dobs[len(dobs)*(layernum-1):len(dobs)*(layernum), 0], \
                100-dobs[len(dobs)*(layernum-1):len(dobs)*(layernum), 1], \
                c= dobs[len(dobs)*(layernum-1):len(dobs)*(layernum), 3], \
                cmap='jet', edgecolors=(0, 0, 0), linewidth =0.9, s=80, \
                vmax=np.max(m_ens_mean), vmin=np.min(m_ens_mean))
    plt.tick_params(labelsize=13)
    plt.title('Ensemble Mean of Prior "' + model_name + '"', fontsize = 15, style='italic')
  
    
    plt.subplot(223)
    m_ens_var =  np.var(m_smpls_pos[:,j_dim*i_dim*(layernum-1):j_dim*i_dim*layernum], axis=0)
    plt.imshow(m_ens_var.reshape(j_dim,i_dim), cmap='bwr', \
               vmin = np.min(m_ens_var), vmax=np.max(m_ens_var))
    plt.colorbar(fraction = 0.02)
    plt.scatter(dobs[len(dobs)*(layernum-1):len(dobs)*(layernum), 0], \
                100-dobs[len(dobs)*(layernum-1):len(dobs)*(layernum), 1], \
                color='white', edgecolors=(0, 0, 0), linewidth =1.2, s=80, alpha=0.7)
    plt.tick_params(labelsize=13)
    plt.title('Ensemble Variance of Posterior "' + model_name + '"', fontsize = 15, style='italic')
    

    plt.subplot(224)
    m_ens_var_pri =  np.var(m_smpls_pri[:,j_dim*i_dim*(layernum-1):j_dim*i_dim*layernum], axis=0)
    plt.imshow(m_ens_var_pri.reshape(j_dim,i_dim), cmap='bwr', \
               vmin = np.min(m_ens_var), vmax=np.max(m_ens_var))
    plt.colorbar(fraction = 0.02)
    plt.scatter(dobs[len(dobs)*(layernum-1):len(dobs)*(layernum), 0], \
                100-dobs[len(dobs)*(layernum-1):len(dobs)*(layernum), 1], \
                color='white', edgecolors=(0, 0, 0), linewidth =1.2, s=80, alpha=0.7)
    plt.tick_params(labelsize=13)
    plt.title('Ensemble Variance of Prior "' + model_name + '"', fontsize = 15, style='italic')    
    
    plt.tight_layout()
    
    return 

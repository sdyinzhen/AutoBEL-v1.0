#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: Apr 11, 2019
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.decomposition import PCA

from grdecl_read_plot import *
from combine_m_sampls import *
from signed_distance_functions import *
from giip_cal import *
from scree_plot import scree_plot
from evd_fast import *
#gd_wellog_data(grid_type_list, real_num, well_path, i_dim, j_dim, k_dim):
from gd_wellog_data import gd_wellog_data
from Sensitivity_Analysis import *
from dgsa_sensitive_comps import dgsa_sensitive_h_pcs
from dgsa_plt import dgsa_plt
from falsifications import fals_1Dplt, fals_2Dplt
from outlier_detection import outlier_2d
from RobustMD_flsification import RobustMD_flsification
from rgrplt_all_dh import rgrplt_all_dh 
from cca_and_ccaplt import cca_d_h
## def cca_d_h(h_var, d_var, components_num):
from cca_and_ccaplt import cca_plt
## def cca_plt(d_cca, h_cca, d_obs_cca, com_num):
from gaussianregression_c_dd_star import *
from post_est_rtn_val import post_est_rtn_val
from post_est_rtn_val import post_est_clsplt
from plt_pos_pri_comp import plt_pos_pri_comp

def Auto_BEL(pri_m_samples_dir, model_name,mgl,samples_size,x_dim, y_dim, z_dim, dobs_file):

    '''This is the main function for runing Auto_BEL'''
    #############################################################################
            ##          STEP 1.  Analyze prior model samples          ## 
    #############################################################################
    print("1. Analyze prior model realizations")

    grdecl_plot(pri_m_samples_dir+model_name+'_',x_dim, y_dim, z_dim, 1, model_name)
    m_pri = np.load('output/model/pri_m_samples.npy')
    
    #############################################################################
            ##            STEP 2.  Prior prediction              ## 
    #############################################################################    
    print("  ")
    print("2. Prior prediction")
    GIIP  = GIIP_cal(1, 0, m_pri, pri_m_samples_dir+'Bulk_volume.GRDECL', True)
    np.save('output/prediction/GIIP_pri', GIIP)
    well_loc = np.loadtxt(fname=dobs_file, skiprows=1)[:,:3]

    # d_pri =  gd_wellog_data([pri_m_samples_dir+model_name+'_'], samples_size, well_loc, x_dim, y_dim, z_dim)
    d_pri = np.load('output/data/d_pri.npy')
    # np.save('data/d_pri',d_pri[:, 74:675:75, 0])
    
    #############################################################################
            ##              STEP 3.  Dimension Reduction              ## 
    #############################################################################    

    print("  ")
    print("3. Dimension reduction of model and data")

    m_pcnums =scree_plot(m_pri, 90)

    m_mean_pri = m_pri.mean(axis=0)
    m_eigvec_pri = evd_fast(m_pri, len(m_pri))
    m_pcscr_pri=(m_pri-m_mean_pri).dot(m_eigvec_pri)
    np.save('output/model/m_mean_pri', m_mean_pri)
    np.save('output/model/m_eigvec_pri', m_eigvec_pri)
    np.save('output/model/m_pcscr_pri', m_pcscr_pri)
    print("QC of the model eigen images")
    eigen_imgs(m_eigvec_pri, [1,3,5,10], x_dim, y_dim)

    d_pri = np.load('output/data/d_pri.npy')
    d_obs = np.loadtxt(fname=dobs_file,skiprows=1)[74:675:75, 3:].T

    d_pca = PCA(n_components=len(d_pri[1]))
    d_pca.fit(d_pri)
    d_pcscr = d_pca.transform(d_pri)

    d_pcscr_obs = d_pca.transform(d_obs)
    np.save('output/data/d_pcscr_pri', d_pcscr)
    np.save('output/data/d_pcscr_obs', d_pcscr_obs)
    
    #############################################################################
            ##              STEP 4.   Falsification             ## 
    #############################################################################
    print("  ")
    print("4. Prior falsification")
    RMD_obs, RMD_Qquantile = RobustMD_flsification(d_pcscr, d_pcscr_obs, True, 95)
    if RMD_obs >= RMD_Qquantile:
        print("  ")
        print("******************************************************")
        print(' >> Prior is falsified! Please re-design your prior <<')
        print("******************************************************")
        return
    print("*******************************")
    print('>> Prior CANNOT be falsified!')
    print("*******************************")
    d_pcscr_pri = np.load('output/data/d_pcscr_pri.npy')
    m_pcscr_pri = np.load('output/model/m_pcscr_pri.npy')[:,:m_pcnums]
    
    #############################################################################
            ##           STEP 5. GSA               ## 
    #############################################################################
    print("  ")
    print("5. Global Sensitivity anlaysis-DGSA")

    headers = []
    for i in range(m_pcnums):
        headers.append('pc'+str(i+1))
    
    try:
        SA_measure= DGSA(m_pcscr_pri.T, d_pcscr_pri.T, headers, 3)
    except Exception as error:
        print(str(error))

    dgsa_plt(SA_measure, 25)
    sensitive_pcnum = np.argwhere(SA_measure.values[:,0]>1)[:len(d_pcscr_pri[1]),0]
    
    #############################################################################
            ##           STEP 6.  Uncertainty reduction              ## 
    #############################################################################
    print("  ")
    print("6. Unceratinty reduction") 
    
    #############################################################################
            ##          STEP 6.1     QC statistical relationships           ##     
    print("  ")
    print("6.1 QC model and data statistical relationships")    
    rgrplt_all_dh(d_pcscr_pri, m_pcscr_pri, d_pcscr_obs, [1,2,3], [1,2,3,4, 5, 6, 7])

    m_star =  m_pcscr_pri[:, sensitive_pcnum]
    d_star = d_pcscr_pri
    dobs_star = d_pcscr_obs

    #############################################################################
            ##          STEP 6.2  CCA             ## 
    print("  ")
    print("6.2 Canonical Corrleation Analysis")  

    d_c, m_c, ad, am = cca_d_h(d_star , m_star, len(m_star[0, :]))
    dobs_c = np.matmul(dobs_star, ad)
    d_c = np.matmul(d_star,ad)
    m_c = np.matmul(m_star,am)
    cca_plt(d_c, m_c, dobs_c,1)

    #############################################################################
            ##          STEP 6.3 Gaussian Regression                ## 
    print("  ")
    print("6.3 Parametric Gaussian Regression & posterior sampling")   

    err_levl = 0.00
    cdd_star = cal_c_dd_star_pca_cca(d_pca.components_, ad, err_levl, d_obs)
    post_est_clsplt([1, 2, 3, 4], m_c, d_c, dobs_c, cdd_star , 2, 2)

    #############################################################################
            ##          STEP 6.4 Reconstruct posterior model                ## 
    print("  ")
    print("6.4 Reconstruct posterior model") 
    all_mc_post=[]
    for cca_comp in range(1, len(m_star[0,:])+1):
        all_mc_post.append(post_est_rtn_val(cca_comp, m_c, d_c, dobs_c, cdd_star, 0))
    all_mc_post=np.asarray(all_mc_post).T
    m_pcscr_post_SA=all_mc_post.dot(np.linalg.inv(am))
    m_pcscr_post = np.load('output/model/m_pcscr_pri.npy')
    m_pcscr_post[:, sensitive_pcnum] = m_pcscr_post_SA
    plt_pos_pri_comp(sensitive_pcnum[0]+1, sensitive_pcnum[1]+1, m_pcscr_pri, m_pcscr_post)

    m_eigvec = np.load('output/model/m_eigvec_pri.npy')
    m_pri_mean = np.load('output/model/m_mean_pri.npy')
    m_post = m_pcscr_post.dot(m_eigvec.T)  + m_pri_mean
    np.save('output/model/pos_m_samples', m_post)
    print("  ")
    print("AUTO-BEL completed :-)!")  
    return 
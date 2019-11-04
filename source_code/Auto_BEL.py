#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: Apr 11, 2019

import numpy as np
from sklearn.decomposition import PCA

from source_code.combine_mc_samples import *
from source_code.signed_distance_functions import *
from source_code.giip_cal import *
from source_code.scree_plot import scree_plot
from source_code.evd_fast import *
from source_code.gd_wellog_data import gd_wellog_data
from source_code.DGSA_light import DGSA_light
from source_code.gsa_pareto_plt import gsa_pareto_plt
from source_code.falsifications import fals_1Dplt, fals_2Dplt
from source_code.outlier_detection import outlier_2d
from source_code.RobustMD_flsification import RobustMD_flsification
from source_code.rgrplt_all_dh import rgrplt_all_dh 
from source_code.cca_and_ccaplt import cca_d_h
from source_code.cca_and_ccaplt import cca_plt
from source_code.gaussianregression_c_dd_star import *
from source_code.post_est_rtn_val import post_est_rtn_val
from source_code.post_est_rtn_val import post_est_clsplt
from source_code.plt_pos_pri_comp import plt_pos_pri_comp
from source_code.plt_MC_models_smpls import m_ensampl_plt
from source_code.plt_MC_models_smpls import mc_samples_plot
from source_code.qc_reslts_plt import m_ensampl_plt


def Auto_BEL(pri_m_samples_dir, model_names, model_types, mgl, samples_size, x_dim, y_dim, z_dim, grid_h_resolution, dobs_file):
    '''This is the main function for runing Auto_BEL'''
    #############################################################################
     ##          STEP 1.  Analyze prior model samples & extract data          ## 
    #############################################################################
    print("1. Initialization")
    '''Loading and Visualization of prior MC samples'''
    m = []
    for i in range(len(model_names)):
        m.append(np.load(pri_m_samples_dir + model_names[i]+'.npy'))
        mc_samples_plot(m[i], model_names[i], model_types[i], 'Prior', x_dim, y_dim, z_dim, 1)

    '''obtain well trajectory index ("G_d") from dobs_file'''
    well_path = np.loadtxt(fname=dobs_file, skiprows=1)[:,:3].astype(int)  
    well_path = well_path.astype(int)  
    G_d= x_dim*y_dim*(well_path[:,2]-1) + (y_dim-well_path[:,1])*x_dim + well_path[:,0]-1
    np.save('output/model/G_d', G_d)
    '''Extract prior data samples'''
    d=[]
    for i in range(len(model_names)):
        d.append(m[0][:,G_d])
    
    #############################################################################
            ##            STEP 2.  Prior prediction              ## 
    #############################################################################
    print("  ")
    print("2. Prior prediction")
    GIIP  = GIIP_cal(1, 0, m[0], grid_h_resolution , True)
    np.save('output/prediction/GIIP_pri', GIIP)
    
    #############################################################################
            ##           STEP 3.  Dimension Reduction of data & model       ## 
    #############################################################################
    
    print("  ")
    print("3. Dimension reduction of model and data")
    '''m_pcnumsc: the number of model PCs to preserve in dimension reduction'''
    m_pcnums = []
    for i in range(len(model_names)):
        m_pri = m[i]
        d_pri = d[i]
        
        m_pcnum =scree_plot(m_pri, 'model', 90, 'plot')
        m_pcnums.append(m_pcnum)
        m_mean_pri = m_pri.mean(axis=0)
        m_eigvec_pri = evd_fast(m_pri, len(m_pri))
        m_pcscr_pri=(m_pri-m_mean_pri).dot(m_eigvec_pri)

        np.save('output/model/'+model_names[i]+'_mean_pri', m_mean_pri)
        np.save('output/model/'+model_names[i]+'_eigvec_pri', m_eigvec_pri)
        np.save('output/model/'+model_names[i]+'_pcscr_pri', m_pcscr_pri)
        print("QC of the model eigen images")
        eigen_imgs(m_eigvec_pri, [1,3,5,10], x_dim, y_dim)
        
        d_obs = np.loadtxt(fname=dobs_file, skiprows=1)[:,3+i:4+i].astype(int).T
        d_pca = PCA(n_components=d_pri.shape[1])
        d_pca.fit(d_pri)
        d_pcscr = d_pca.transform(d_pri)
        d_pcscr_obs = d_pca.transform(d_obs)
        np.save('output/data/dpcscr_pri_'+model_names[i], d_pcscr)
        np.save('output/data/dpcscr_obs_'+model_names[i], d_pcscr_obs)
        np.save('output/data/dpca_eigenvec_'+model_names[i], d_pca.components_)
        
    #############################################################################
            ##              STEP 4.   Falsification             ## 
    #############################################################################
    print("  ")
    print("4. Prior falsification")
    for i in range(len(model_names)):   
        d_pri = np.load('output/data/dpcscr_pri_'+model_names[i]+'.npy')
        d_obs = np.load('output/data/dpcscr_obs_'+model_names[i]+'.npy')
        RMD_obs, RMD_Qquantile = RobustMD_flsification(d_pri, d_obs, model_names[i], True, 95)

        if RMD_obs >= RMD_Qquantile:
            print("  ")
            print("******************************************************")
            print(' >> "'+model_names[i]+'" Prior is falsified! Please re-design your prior <<')
            print("******************************************************")
            return
        print("*******************************")
        print('>> "'+model_names[i]+'" Prior CANNOT be falsified!')
        print("*******************************")

        
    #############################################################################
            ##           STEP 5. GSA               ## 
    #############################################################################
    print("  ")
    print("5. Global Sensitivity anlaysis-DGSA")
    for i in range(len(model_names)):
        m_pcscr_pri = np.load('output/model/'+model_names[i]+'_pcscr_pri.npy')[:,:m_pcnums[i]]
        d_pcscr_pri = np.load('output/data/dpcscr_pri_'+model_names[i]+'.npy')
        headers = []
        for para in range(m_pcnums[i]):
            headers.append('pc'+str(para+1))
        try:
            SA_measure= DGSA_light(m_pcscr_pri, d_pcscr, headers)
        except Exception as error:
            print(str(error))
        gsa_pareto_plt(SA_measure, model_names[i])
        np.save('output/data/SA_measure_'+model_names[i], SA_measure)
        
        
    #############################################################################
            ##           STEP 6.  Uncertainty reduction              ## 
    #############################################################################
    print("  ")
    print("6. Unceratinty reduction") 
    print("  ")
    #############################################################################
            ##          STEP 6.1     QC statistical relationships           ## 
    print("6.1 QC model and data statistical relationships")    
    for i in range(len(model_names)):
        SA_measure = np.load('output/data/SA_measure_'+model_names[i]+'.npy')[:,0]
        sensitive_pcnum = np.argwhere(SA_measure>1)[:len(d_pcscr[1]),0]
        
        m_pcscr_pri = np.load('output/model/'+model_names[i]+'_pcscr_pri.npy')[:,:m_pcnums[i]]
        d_pcscr_pri = np.load('output/data/dpcscr_pri_'+model_names[i]+'.npy')
        d_pcscr_obs = np.load('output/data/dpcscr_obs_'+model_names[i]+'.npy')
        
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

        #############################################################################
                ##          STEP 6.4 Reconstruct posterior model                ## 
        err_levl = 0.00
        dpca_eigenvec = np.load('output/data/dpca_eigenvec_'+model_names[i]+'.npy')
        cdd_star = cal_c_dd_star_pca_cca(dpca_eigenvec, ad, err_levl, d_obs)
        post_est_clsplt([1, 2, 3, 4], m_c, d_c, dobs_c, cdd_star , 2, 2)

        print("  ")
        print("6.4 Reconstruct posterior model") 
        all_mc_post=[]
        for cca_comp in range(1, len(m_star[0,:])+1):
            all_mc_post.append(post_est_rtn_val(cca_comp, m_c, d_c, dobs_c, cdd_star, 0))
        all_mc_post=np.asarray(all_mc_post).T
        m_pcscr_post_SA=all_mc_post.dot(np.linalg.inv(am))
        m_pcscr_post = np.load('output/model/'+model_names[i]+'_pcscr_pri.npy')
        m_pcscr_post[:, sensitive_pcnum] = m_pcscr_post_SA
        plt_pos_pri_comp(sensitive_pcnum[0]+1, sensitive_pcnum[1]+1, m_pcscr_pri, m_pcscr_post)

        m_eigvec = np.load('output/model/'+model_names[i]+'_eigvec_pri.npy')
        m_pri_mean = np.load('output/model/'+model_names[i]+'_mean_pri.npy')
        m_post = m_pcscr_post.dot(m_eigvec.T)  + m_pri_mean
        np.save('output/model/'+model_names[i]+'_model_post', m_post)
        
        #############################################################################
                ##          STEP 6.5 QC posterior results               ## 

        print("  ")
        print("6.5. QC posterior results") 
        print("  ")
        print("6.5.1 Plot posterior models") 
        m_pri = np.load(pri_m_samples_dir + model_names[i]+'.npy')
        d_obs = np.c_[np.loadtxt(fname=dobs_file,skiprows=1)[:, :3], np.loadtxt(fname=dobs_file,skiprows=1)[:, i+3:i+4]]
        # m_sampls_plt(m_post, samples_size, model_names[i], x_dim, y_dim, z_dim)
        mc_samples_plot(m_post, model_names[i], model_types[i], 'Posterior', x_dim, y_dim, z_dim, 1)
        m_ensampl_plt(m_post, m_pri,  model_names[i], 1, x_dim, y_dim, 1, d_obs)


        print("  ")
        print("6.5.2 Calculate posterior prediction") 
        GIIP_post  = GIIP_cal(1, 0, m_post, grid_h_resolution, False)
        np.save('output/prediction/GIIP_post', GIIP_post)
        GIIP_pri = np.load('output/prediction/GIIP_pri.npy')
        giip_compare(GIIP_pri, GIIP_post, model_names[i])
    
    print("  ")
    print("AUTO-BEL completed :-)!") 
    return 
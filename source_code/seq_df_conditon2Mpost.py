## Author: David Yin 
## Contact: yinzhen@stanford.edu
## Date: Feb 23, 2019


import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
def sedf_post_sample_rtn(cca_comp, h_c,d_c, d_obs_c, c_dd_star):
    '''This is the function to sample posterior in Canonical space, based on d_obs_c
        ---------------
        Parameters
        ----------------
        cca_comp: the number of canonical component to sample
        h_c: ndarray of prediction variable in canonical space, #realizations x #features
        d_c: ndarray of data variable in canonical space, #realizations x #features
        d_obs_c: 1Darray of data observations in canonical space, #1 x #features
        c_dd_star: ndarrary of error covaraince, #features x #features
        '''
    com = cca_comp
    #### Estimate f(h*) of priors
    h_c_pri=np.sort(h_c[:,com-1])
    h_star_bar=np.mean(h_c[:,com-1])
    c_hst_hst = np.var(h_c_pri)
    f_pri=1/(c_hst_hst * np.sqrt(2 * np.pi))*np.exp(-(h_star_bar-h_c_pri)*(h_star_bar-h_c_pri)/(c_hst_hst*2))

    ####### ESTIMATE Posteriors ###
    g_dh=stats.linregress(h_c[:,com-1],d_c[:,com-1])[0]
    epsil=(g_dh*h_c[:,com-1]-d_c[:,com-1])
    c_epsil=np.var(epsil)
    
    c_dst_dst = c_epsil + c_dd_star[com-1, com-1]

    h_hat_star=1/(g_dh/c_dst_dst+1/c_hst_hst)*(g_dh/c_dst_dst*d_obs_c[:,com-1]+1/c_hst_hst*h_star_bar)
    c_hat_hsthst_star=1/(g_dh/c_dst_dst*g_dh+1/c_hst_hst)

    mean_post=h_hat_star
#     std_post=np.sqrt(c_hat_hsthst_star)
#     h_c_post=np.random.normal(mean_post,std_post,1)

    return mean_post

def sedf_est_hpost_scrs(h_c, d_c, ad, ah, dobs_star, cdd_star):
    '''This is the function for the DF for generating posterior PC scores based on depended posterior models, 
            e.g.: generate posterior porosity based on the posterior facies
        -----------------
        parameters
        -----------------
        d_c, h_c:data and prediction variables in cannonical space
        ad, ah: CCA operators of data and prediction variables
        dobs_star: ndarray, observations data in pca space (before CCA), same #feature as corresponding to d_c. 
        cdd_star:  ndarrary of error covaraince, #features x #features
    '''
    hpost_scr = []
    for i in range(len(dobs_star)):
        dobs_star_=dobs_star[i:i+1,:]
        dobs_c = np.matmul(dobs_star_, ad)
        all_hc_post=[]
        for cca_comp in range(1, len(h_c[0,:])+1):
            all_hc_post.append(sedf_post_sample_rtn(cca_comp, h_c, d_c, dobs_c, cdd_star))
        all_hc_post=np.asarray(all_hc_post).T
        h_pcscr_post=all_hc_post.dot(np.linalg.inv(ah))
        hpost_scr.append(h_pcscr_post[0])
    hpost_scr = np.asarray(hpost_scr)
    
    return hpost_scr

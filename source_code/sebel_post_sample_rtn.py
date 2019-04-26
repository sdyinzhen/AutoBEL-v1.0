
# coding: utf-8

# In[5]:


## rpt_var_size()
## Author: David Yin 
## Contact: yinzhen@stanford.edu
## Date: Feb 23, 2019

### This function estimates and returns the 1 sample of posterior h_c for se-bel. 
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
def sebel_post_sample_rtn(cca_comp, h_c,d_c, d_obs_c, c_dd_star, fig_ind):
    com = cca_comp
    #### Estimate f(h*) of priors
    h_c_pri=np.sort(h_c[:,com-1])
    h_star_bar=np.mean(h_c[:,com-1])
    c_hst_hst = np.var(h_c_pri)
    f_pri=1/(c_hst_hst * np.sqrt(2 * np.pi))*    np.exp(-(h_star_bar-h_c_pri)*(h_star_bar-h_c_pri)/(c_hst_hst*2))
    #### **** ####

    ####### ESTIMATE Posteriors ###
    g_dh=stats.linregress(h_c[:,com-1],d_c[:,com-1])[0]
    epsil=(g_dh*h_c[:,com-1]-d_c[:,com-1])
    c_epsil=np.var(epsil)
    
    c_dst_dst = c_epsil + c_dd_star[com-1, com-1]

    h_hat_star=1/(g_dh/c_dst_dst+1/c_hst_hst)*(g_dh/c_dst_dst*d_obs_c[:,com-1]+1/c_hst_hst*h_star_bar)
    c_hat_hsthst_star=1/(g_dh/c_dst_dst*g_dh+1/c_hst_hst)

    mean_post=h_hat_star
    std_post=np.sqrt(c_hat_hsthst_star)
    
    h_c_post=np.random.normal(mean_post,std_post,1)

    return h_c_post


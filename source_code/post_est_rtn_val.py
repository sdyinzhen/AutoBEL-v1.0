
# coding: utf-8

# In[1]:


#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: September 12, 2018

### This function estimates and returns the sampled posterior h_c and its distribution. 
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
### cca_comp - the number of canonical components for posterior estimation
### h_c, d_c, d_obs_c- The　array of cca scores of h, d, and d_obs
### c_dd_star - the array of uncertainty covariance in the canonical space, \
            # obtained from cal_c_dd_star_pca_cca(pca_compnts, cca_ad, dat_err_levl, d_obs) or cal_c_dd_star_cca functions
### fig_ind- figure index, 1 for produce figure, not 1 for no figure. 
def post_est_rtn_val(cca_comp, h_c,d_c, d_obs_c, c_dd_star, fig_ind):

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
    
    #c_dd_star=d_pca_comp[0][0]*0.1*cca.x_loadings_[0][0]
    #c_dd_star= c_dd_star[com-1, com-1]

    c_dst_dst = c_epsil + c_dd_star[com-1, com-1]

    h_hat_star=1/(g_dh/c_dst_dst+1/c_hst_hst)*(g_dh/c_dst_dst*d_obs_c[:,com-1]+1/c_hst_hst*h_star_bar)
    c_hat_hsthst_star=1/(g_dh/c_dst_dst*g_dh+1/c_hst_hst)

    mean_post=h_hat_star

    std_post=np.sqrt(c_hat_hsthst_star)
    
    ##*** SAMPLE POSTERIORS ***##
    h_c_post=np.random.normal(mean_post,std_post,len(h_c))
    #####*****####
     
    ###### Plots #####
    if fig_ind == 1:
        fig, ax1 = plt.subplots(figsize=(6,4))
        ax2=ax1.twinx()

        bins=np.sort(h_c_post)
        ## Priors
        ax1.hist(h_c[:,com-1], 15, edgecolor='black', facecolor='aqua')
        ax2.plot(h_c_pri, f_pri, '--', linewidth = 2, c='b')
        ## Posteriors
        ax1.hist(h_c_post,bins=15, edgecolor='black', facecolor='lightcoral', alpha = 0.8)

        ax2.plot(bins, 1/(std_post * np.sqrt(2 * np.pi))*np.exp( - (bins - mean_post)**2 / (2 * std_post**2) ), linewidth=2, color='red')

        ax1.set_xlabel('h (CCA componennt_'+ str(com) +' score)',fontsize = 14)
    
    return h_c_post

	

def post_est_clsplt(cca_complist, h_c, d_c, d_obs_c, c_dd_star, pic_row, pic_col):
    '''function post_est_clsplt returns the cluster plot of posteriors. 
        cca_complist - the list contains the number of canonical components for posterior estimation
        h_c, d_c, d_obs_c- The　array of cca scores of h, d, and d_obs
        pic_row, pic_col: the total row and column number of the figures. '''
    fig = plt.figure( figsize=(4*pic_col, 4.5*pic_row))
    #plt.title('Parametric Gaussian Regression & posterior sampling', fontsize=18, loc='left', weight='bold')	
    count = 1
    for com in cca_complist:     
        #### Estimate f(h*) of priors
        h_c_pri=np.sort(h_c[:,com-1])
        h_star_bar=np.mean(h_c[:,com-1])
        c_hst_hst = np.var(h_c_pri)
        f_pri=1/(c_hst_hst * np.sqrt(2 * np.pi))*\
        np.exp(-(h_star_bar-h_c_pri)*(h_star_bar-h_c_pri)/(c_hst_hst*2))
        #### **** ####

        ####### ESTIMATE Posteriors ###
        g_dh=stats.linregress(h_c[:,com-1],d_c[:,com-1])[0]
        epsil=(g_dh*h_c[:,com-1]-d_c[:,com-1])
        c_epsil=np.var(epsil)
        c_dst_dst = c_epsil+c_dd_star[com-1, com-1]
        h_hat_star=1/(g_dh/c_dst_dst+1/c_hst_hst)*(g_dh/c_dst_dst*d_obs_c[:,com-1]+1/c_hst_hst*h_star_bar)
        c_hat_hsthst_star=1/(g_dh/c_dst_dst*g_dh+1/c_hst_hst)
        mean_post=h_hat_star
        std_post=np.sqrt(c_hat_hsthst_star)
    
        ##*** SAMPLE POSTERIORS ***##
        h_c_post=np.random.normal(mean_post,std_post,len(h_c))
        #####*****####
     
        ###### Plots #####
        ax1 = fig.add_subplot(pic_row, pic_col, count)
        ax2=ax1.twinx()

        bins=np.sort(h_c_post)
        ## Priors
        ax1.hist(h_c[:,com-1], 15, edgecolor='black', facecolor='aqua')
        ax2.plot(h_c_pri, f_pri, '--', linewidth = 2, c='blue')
        ## Posteriors
        ax1.hist(h_c_post,bins=15, edgecolor='black', facecolor='lightcoral', alpha = 0.9)

        ax2.plot(bins, 1/(std_post * np.sqrt(2 * np.pi))*np.exp( - (bins - mean_post)**2 / (2 * std_post**2) ), linewidth=2, color='red')

        ax1.set_xlabel('m (CCA component '+ str(com) +')',fontsize = 14)
        ax1.set_ylabel('Count',fontsize = 14)
        plt.setp(plt.gca(), frame_on=False, yticks=())
        plt.title('posterior vs prior model', fontsize=14, style='italic', weight='bold')
        count = count +1
    
    plt.subplots_adjust(top=0.75, bottom=0.08, left=0.10, right=0.95, hspace=0.40,
                    wspace=0.35)
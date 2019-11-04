## MD_flsification(d_var, d_obs, plt_OrNot, Q_quantile)
## Author: David Yin 
## Contact: yinzhen@stanford.edu
## Date: April 29, 2019


from sklearn.covariance import MinCovDet as MCD
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
def RobustMD_flsification(d_var, d_obs, prior_name, plt_OrNot, Q_quantile):
    
    '''
    This function falsifies the prior using Robust Mahalanobis Distance RMD.  
    d_var: the data variable, (nXp)
    d_obs: the data observation variable, (1xp)
    prior_name: name of the prior model for falsification, string
    plt_OrNot: True or False, to create the distribution plot of the calculated RMDs. 
    Q_quantile：the Q_quantile of the RMD distribution, 95 or 97.5 is suggested
    example: MD_flsification(d_pri, d_obs, True, 95) will produce the RMD_obs, RMD_pri, RMD_Q95, and plot them. 
    '''
    
    mcd = MCD(random_state=0).fit(d_var)
    new_obs = d_obs-mcd.location_
    md_obs= np.sqrt(new_obs.dot(np.linalg.inv(mcd.covariance_)).dot(new_obs.T))
    print('Robust Mahalanobis Distance of d_obs = ', md_obs[0,0].round(decimals = 3))
    md_samples=[]
    for i in range(len(d_var)):
        sample = d_var[i:i+1, :]-mcd.location_
        md_samp = np.sqrt(sample.dot(np.linalg.inv(mcd.covariance_)).dot(sample.T))[0,0]
        md_samples.append(md_samp)
    md_samples = np.asarray(md_samples)
    print(str(Q_quantile)+'th Quantile of Robust Mahalanobis Distance is', \
          stats.scoreatpercentile(md_samples, Q_quantile).round(decimals=3))

    if plt_OrNot == True:
        plt.figure(figsize=(6,5))
        plt.scatter(np.arange(1,(len(d_var)+1)), md_samples, c=abs(md_samples),                     cmap ='winter_r', s=50, vmax = md_samples.max(), vmin=md_samples.min(),                    linewidths=1, edgecolor='k')
        plt.scatter([0], md_obs, c=md_obs,                     cmap ='winter_r', marker='D', s=110, vmax = md_samples.max(), vmin=md_samples.min(),                    linewidths=3, edgecolor='red')
        plt.ylabel('Robust Mahalanobis dist', fontsize=12)
        plt.xlabel('realization No.', fontsize=12)
        plt.xlim(-8, 259)
        plt.hlines(y=stats.scoreatpercentile(md_samples, Q_quantile), xmin= -10, xmax=259, colors='red', linewidths=2, linestyles='--')
        cbar = plt.colorbar(fraction=0.035)
        cbar.ax.set_ylabel('RMD')
        plt.title('Prior falsification of "'+ prior_name+'" using Robust Mahalanobis Distance', \
                  fontsize=18, loc='left', style='italic')
    
    return md_obs[0,0].round(decimals = 3), stats.scoreatpercentile(md_samples, Q_quantile).round(decimals=3)
#Author: David Zhen Yin, Yizheng Wang
#Contact: yinzhen@stanford.edu, yizhengw@stanford.edu
#Date: August 22, 2019
import numpy as np
from tqdm import tqdm
from KMedoids import KMedoids
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
def DGSA_light(parameters, responses, ParametersNames=0, n_clsters=3, n_boots = 3000):
    '''
    Main function of DGSA light version
    Parameters
    ------------
    parameters: input model parameters, 2D array (#samples, #parameters)
    responses: model responses to the input parameters, 2D array (#samples, #responses)
    ParametersNames (optional): name of the input model parameters, 1D list, default = ['p1', 'p2', ....]
    n_clsters (optional): number of KMedoids clusters to classify the model responses, default = 3
    n_boots (optional): number of boostrap resamplings, default = 3000
    
    Output
    ------------
    dgsa_measures_main:  main sensitivity of parameters measured by DGSA, (pd.DataFrame)data frame. 
    '''
    n_samples, n_parameters = parameters.shape[0], parameters.shape[1]
    '''STEP 1. K-Medoids clustering'''
    OK = False
    while not OK:
        try: 
            model = KMedoids(n_clusters=n_clsters)
            Medoids, clsters = model.fit(responses, plotit=False)
            OK = True
        except valueError:
            OK = False
    '''STEP 2. Calculate L1-Norm distance between sample distribution and cluster distributions'''
    '''STEP 2.1 Calucate the CDF of the original parameters'''
    percentiles = np.arange(100)
    cdf_parameters = np.percentile(parameters, percentiles,axis = 0)

    '''STEP 2.2 Calculate the L1 norm for the clusters & Run bootstrap sampling'''
    def L1norm_cls(k):
        '''Define function to calculate L1-norm for clustered parameters'''
        parameters_cls = parameters[clsters[k]]
        L1norm_clster[k, :] = np.sum(abs(np.percentile(parameters_cls, percentiles,axis=0) - cdf_parameters),axis=0)
        return L1norm_clster[k,:]
    L1norm_clster = np.zeros((n_clsters, n_parameters))
    [L1norm_cls(n_c) for n_c in range(n_clsters)]

    '''STEP 2.3 Calculate the L1 norm for the n bootstraps'''
    def L1norm_Nboots(k,p):
        '''Define function to calculate L1-norm distances for N boostrap sampling'''        
        parameters_Nb = parameters[np.random.choice(len(parameters), len(clsters[k]), replace=False)]
        L1norm_Nb[p,k, :] = np.sum(abs(np.percentile(parameters_Nb, percentiles, axis=0) - cdf_parameters), axis=0)
        return L1norm_Nb[p,k, :]
    L1norm_Nb = np.zeros((n_boots, n_clsters, n_parameters))
    [[L1norm_Nboots(n_c,p) for n_c in range(n_clsters)] for p in tqdm(range(n_boots))]

    '''STEP 3. Calculate main DGSA measurements'''
    dgsa_measures_cls = L1norm_clster/(np.percentile(L1norm_Nb, 95, axis=0))
    dgsa_measures_main = np.max(dgsa_measures_cls, axis=0)
    
    if ParametersNames == 0:
        dgsa_measures_main = pd.DataFrame(dgsa_measures_main, ['p{}'.format(i) for i in range(1, n_parameters+1)])
    else:
        dgsa_measures_main = pd.DataFrame(dgsa_measures_main, ParametersNames)
        
    return dgsa_measures_main
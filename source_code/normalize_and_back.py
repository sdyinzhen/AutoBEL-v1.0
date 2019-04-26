# NormalScoreTransform(Untransformed, plt_or_not)
# Back_transform_N2U(Input, Match)
## Author: David Yin 
## Contact: yinzhen@stanford.edu
## Date: Nov 03, 2018

### The functions run the normal score transform and back transform

from scipy.stats import percentileofscore
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

def NormalScoreTransform(Untransformed, plt_or_not):
# Untransformed: 2D array (realizations No. x features No.), the orignal imput 2D array to transform to Gaussian. 
# plt_or_not: True or False, to plot the distribution of the Original & Normal Transformed Matrix (True) or not plot (False)
    Transformed_3D = np.empty([Untransformed.shape[0], Untransformed.shape[1]])
    Untransformed_3D =  Untransformed
    for j in range(Untransformed_3D.shape[1]):
        ecdf = []
        Untransformed = Untransformed_3D[:,j]
        for i in Untransformed:
            percntile = percentileofscore(Untransformed, i)
            ecdf.append(percntile)
        ecdf = np.asarray(ecdf)/100

        ## avoid infinity in the normal Score transforme.
        ecdf[ecdf>0.999] = 0.999   
        Transformed = norm.ppf(ecdf) 
        if plt_or_not == True:
            plt.figure(figsize=(6, 3))
            plt.subplot(121)
            plt.hist(Untransformed)
            plt.subplot(122)
            plt.hist(Transformed)
        
        Transformed_3D[:,j] = Transformed
    return Transformed_3D

# This function is back-transforms the Normal distribution (Input) to the Origal Uniform Distribution form (Match). 
# Input: 2D array, Gaussian variable to be transformed
# Match: 2D array, Variable who's histogram we are trying to match, Uniform distribution

def Back_transform_N2U(Input, Match):
    Input_3D = Input
    Match_3D = Match
    Transformed_3D = np.empty([Match_3D.shape[0], Match_3D.shape[1]])
    for j in range(Match.shape[1]):
        Input = Input_3D[:,j]
        Match = Match_3D[:,j]
        # calculate the cedf of the Input data
        cdf = []
        for i in Input:
            cdf.append(norm.cdf(i))
        cdf = np.asarray(cdf)

        # calculated ecdf of the Uniform distributed Match data: ecdf_Match = a*x + b  
        a = (1- 1/len(Match))/(Match.max()-Match.min())
        b = (Match.min()*1 - Match.max()/len(Match))/(Match.min()-Match.max())
        Transformed = []
        for i in cdf:
            Transformed.append((i-b)/a)
        Transformed = np.asarray(Transformed)
        Transformed_3D[:,j] = Transformed
    return Transformed_3D

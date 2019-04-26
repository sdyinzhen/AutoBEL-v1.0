# Author: Lijing Wang
# Contact: lijing52@stanford.edu
# Created: Nov 10, 2018

# We directly call R functions modified from Ogy Grujic's Github https://github.com/ogru/DGSA
# Make sure you have R installed before run this code, installed reshape and ggplot2 library

# SA(A|B)
# Given B, how sensitive are the parameters in A?

import numpy as np
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects import pandas2ri
from sklearn.cluster import KMeans
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd

def DGSA(A, B, name_A, num_cluster):
	'''
	Distance based Generalized Sensitivity Analysis method 
	Args:
		A: (np.array) A in SA(A|B), # features * # realizations
		B: (np.array) B in SA(A|B), # features * # realizations
		name_A: (np.vector) Row names of A
		num_cluster: (int) number of cluster
	Output:
		SA_dataframe: (pd.DataFrame) StandardizedSensitivity data frame
	'''
	## Load DGSA R script
	ro.r('source(\'./source_code/dgsa/dgsa_rightcolor.R\')')
	rpy2.robjects.numpy2ri.activate()
	## Clustering
	score = B.T
	kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(score)
	clustering = kmeans.labels_
	clustering = ro.Vector(clustering+1)

	## DGSA
	pandas2ri.activate()
	A = pd.DataFrame(A.T, columns = name_A)
	r_A = pandas2ri.py2ri(A)

	r_dgsa = ro.r['dgsa']
	myDGSA = r_dgsa(clustering, r_A)

	SensitivityMatrix = np.asarray(myDGSA.rx2(1))
	SA_stats = np.diag(np.nan_to_num(SensitivityMatrix).max(axis=0))

	names = np.asarray(myDGSA.rx2(2))

	SA_dataframe = pd.DataFrame(SA_stats, index = names)

	return SA_dataframe 




def DGSA_plot(SA_dataframe, num_plot, title = None):
	'''
	Distance based Generalized Sensitivity Analysis method 
	Args:
		SA_dataframe: (pd.DataFrame) StandardizedSensitivity data frame from DGSA
		num_plot: (int) Numbers of bar plot, from the most sensitivity parameter
	Output:
		Pareto Plot for SA
	'''
	SA_stats = SA_dataframe.values
	max_SA_stats = np.max(SA_stats)
	min_SA_stats = np.min(SA_stats)

	SA_color = np.zeros(SA_stats.shape)
	if max_SA_stats == 1:
		SA_color[SA_stats>=1] = (SA_stats[SA_stats>=1]-0.999)/(1.01-1)
	else:
		SA_color[SA_stats>=1] = (SA_stats[SA_stats>=1]-1)/(max_SA_stats-1)
	SA_color[SA_stats<1] = (SA_stats[SA_stats<1]-1)/(1-min_SA_stats)

	names = SA_dataframe.index

	fig, ax = plt.subplots(figsize=(6,9))
	order = np.argsort(-SA_stats[:,0])[:num_plot]

	ax.barh(np.arange(num_plot),SA_stats[order,0],edgecolor='k', color = cm.RdBu_r(SA_color[order,0]*0.5+0.5),linewidth=0.5)
	ax.invert_yaxis()
    
	plt.yticks(np.arange(num_plot), names[order], fontsize = 12)
	plt.xticks(fontsize=12)

	plt.axvline(x=1, linestyle = '--', c ='k')

	if title:
	    plt.title(title, fontsize = 18)
	plt.show()








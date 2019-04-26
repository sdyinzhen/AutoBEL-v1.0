#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: September 11, 2018
# coding: utf-8

# In[1]:


from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
def outlier_2d(x_var, y_var, x_obs, y_obs, n_samples, outlier_fraction):

    classifiers = {
        "one-class SVM": svm.OneClassSVM(nu=0.1, kernel='rbf',gamma=0.01),
        "Elliptic Envelope": EllipticEnvelope(contamination=outlier_fraction),
        "Isolation Forest": IsolationForest(max_samples=n_samples, contamination=outlier_fraction),
        "Local Outlier Factor": LocalOutlierFactor(n_neighbors=35,
                                              contamination=outlier_fraction) }

    X = np.vstack((x_var, y_var)).transpose()

    xx, yy = np.meshgrid(np.linspace(np.min(X)*1.5 ,np.max(X)*1.5, 100),                      np.linspace(np.min(X)*1.5 ,np.max(X)*1.5, 100))

    plt.figure(figsize=(9, 7))

    for i, (clf_name, clf) in enumerate(classifiers.items()):
        
            # fit the data and tag outliers
            if clf_name == "Local Outlier Factor":
                y_pred = clf.fit_predict(X)
                scores_pred = clf.negative_outlier_factor_
            else:
                clf.fit(X)
                scores_pred = clf.decision_function(X)
                y_pred = clf.predict(X)
            threshold = stats.scoreatpercentile(scores_pred,
                                                100 * outlier_fraction)
        
            # plot the levels lines and the points
            if clf_name == "Local Outlier Factor":
                # decision_function is private for LOF
                Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            subplot = plt.subplot(2, 2, i + 1)
            subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
                             cmap=plt.cm.Blues_r)
            a = subplot.contour(xx, yy, Z, levels=[threshold],
                                linewidths=2, colors='r')
            subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                             colors='orange')
            b = subplot.scatter(X[:, 0], X[:, 1], c='white', s=30, edgecolor='k',linewidths=0.7)
        
            c = plt.scatter(x_obs, y_obs, color='r',s=100, marker='D', linewidths=1,edgecolor='yellow')

            subplot.axis('tight')
            subplot.set_xlabel("%d. %s " % (i + 1, clf_name))
            
    plt.subplots_adjust(0.04, 0.04, 0.96, 0.94, 0.2, 0.30)
    plt.suptitle("Outlier detection")
    plt.show()        


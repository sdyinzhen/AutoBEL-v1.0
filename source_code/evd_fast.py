## evd_fast(X, n_components)
## David Yin, yinzhen@stanford.edu
## Date: Oct 29, 2018

## This is the function to caculated eigen vectors for matrix with semi-large dimension, e.g.: matrix with dimension (LxP), where L<<P.
## X: input matrix with dimension LxP, where L<<P. 
## n_components: (int), the estimated number of eigen vectors (PC componets)
## example: Xeig_vecs = pca_fast(X, 10) will calculate the first 10 eigen vectors for matrix X. 

import numpy as np
import datetime
import matplotlib.pyplot as plt
def evd_fast(X, n_components):
    print((datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')))
    X = X - X.mean(axis=0)
    eig_val, eig_vec = np.linalg.eig(X.dot(X.transpose()))
    new_eig_vecs = []
    for i in range(n_components):
        new_vec = X.transpose().dot(eig_vec[:,i:i+1])
        new_eig_vecs.append(new_vec[:,0]/np.linalg.norm(new_vec))
    new_eig_vecs = np.asarray(new_eig_vecs).T
    print((datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')))
    return new_eig_vecs
# This function returns the first eigen value of matrix X.
def first_eigval(X):
    X = X - X.mean(axis=0)
    eig_val, eig_vec = np.linalg.eig(X.dot(X.transpose()))
    return eig_val[0]

def eigen_imgs(eigen_vecs, eig_nums, i_dim,j_dim):
    '''
    This is the function to plot the eigen_images
    arg:
        eigen_vecs: the ndarray of the eigen vectors
        eig_nums: 1d arrary defines which pc numbers to plot
        i_dim, j_dim: the i and j dimension of the grid model        
    '''
    plot_num = len(eig_nums)
    fig_row = int((plot_num+3)/4)
    fig=plt.figure(figsize=(15, fig_row*3))
    
    count = 1
    for i in eig_nums:
        plot=fig.add_subplot(fig_row, 4, count)
        count = count+1
        plt.imshow(eigen_vecs[:,i-1].reshape(j_dim,i_dim), cmap='jet')       
        plt.xticks(fontsize = 13)
        plt.yticks(fontsize = 13)
        plt.title('model eigen_img (PC' + str(i) +')', fontsize = 14)
    plt.subplots_adjust(top=0.55, bottom=0.08, left=0.10, right=0.95, hspace=0.15,
                    wspace=0.35)
    
    #t = (" ")
    #plt.figure(figsize=(3, 0.1))
    #plt.text(0, 0, t, style='normal', ha='center', fontsize=16, weight = 'bold')
    #plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
    #plt.show()
    return

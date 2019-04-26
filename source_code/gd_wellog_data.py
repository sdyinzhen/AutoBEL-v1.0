## This is the g_d(.) function to get the well log data from models

#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: Oct 12, 2018


# e.g. get_wellog_data(['Facies_', 'Poro'], 40, well_loc, 200, 100, 75)
# grid_type_list: is a list that contains the prefix name of each type of model properties. 
# real_num is number of sample realization
# well_path is a 3D array that contains (i,j,k) locations of the well path.
# i_dim,j_dim,k_dim: is the max dimension number of i, j, k
import numpy as np

from tqdm import tqdm

import sys
sys.path.insert(0, 'python_functions/')
from grdecl_read_plot import *

def gd_wellog_data(grid_type_list, real_num, well_path, i_dim, j_dim, k_dim):
    data_var = []
    
    for realnum in tqdm(range(real_num)):        
        all_log_types=[]
        for item in grid_type_list:
            log_type =[]            
            
            data = grdecl_read(item + str(realnum+1) + '.GRDECL',i_dim,j_dim,k_dim)                                 
            for row in well_path:
                well_data = data[int(row[2]-1), int(row[1]-1),int(row[0]-1)]
               # i+=1
                #sys.stdout.write(str(i)+' ')
                log_type.append(well_data)
            
            all_log_types.append(log_type)
            
        all_log_types=np.asarray(all_log_types).transpose()
    
        data_var.append(all_log_types)
        
    data_var=np.asarray(data_var)
    
    return data_var 


import numpy as np
def grdecl_read(file_name, i_dim, j_dim, k_dim):
    
   
    ####### MAIN FUNCTION ######
    
    ## import the GRDECl files by ignore the first 15 header lines, and then unwrap the unified grid values, then plot them
    new_data=[]
    f = open(file_name, 'r')
    data= f.readlines()[15:]
    f.close()
    ## save the original data into a new list to avoid the original row data wraps
    for row in data:
        for a in row.split():
            new_data.append(a)
                
            
        ## final_data will contain all original data grid by grid which is not unified.        
    final_data=[] 
        
    # 'a' is the data element of the new list from the last step
    for a in new_data:
        #index is to find the number before '*', which is the number of cells that have the same values
        index = a.find('*')        
        if index >=0:
            cell_num = int(a[:index])
            for i in range(cell_num):
                val=float(a[index+1:])
                # print(val)
                final_data.append(val)
        elif a !='/':
            val=float(a)
            # print(val)
            final_data.append(val)
        

    # set the grid data to be a k_dim x j_dim x i_dim matrix        
    final_data=np.asarray(final_data)        
    grid_data = final_data.reshape(k_dim,j_dim,i_dim)
    # The grid_data will be grid_data[k_dim][j_dim][i_dim]

    return grid_data
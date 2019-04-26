
# coding: utf-8

# In[2]:


# errorCorrection_PatchDS_3D(Sim_posmod, Ti_primod, wel_obs_data, i_dim, j_dim, k_dim, run_smooth)
# Authors: Chen Zuo, David Zhen Yin
# Contact: yinzhen@stanford.edu
# Date: Oct 22, 2018

# This is the function to correct the posterior residual errors at mismatched wells using "MPS Direct Sampling" for 3D models of posterior. 
# This function returns the DS corrected 3D models posterior. 
# e.g. corrected_posterior = errorCorrection_PatchDS_3D(Sim_posmod, Ti_primod, wel_obs_data, i_dim, j_dim, k_dim, True)

# Sim_posmod: the 2D array matrix that constains all the posterior realizations(each realization is vectorized) that need to correct, 
#            Sim_posmod = realization_number x total_grid_number_per_model(i_dim x j_dim x k_dim)
# Ti_primod: the 2D array matrix that contains all the prior realizations(each realization is vectorized), used as trainning image, same dimension as the "Sim_posmod"
# wel_obs_data: the 3D arrays that contains the well observation data,
#           wel_obs_data = well_number x well_sample_points x 4 (i_location, j_location, k_location, and observation value)
# i_dim, j_dim, k_dim: the i, j, k dimension of the model
# run_smooth: False or True. 
#              If "True", the input "Sim_posmod" will be smoothed to remove the noise, before running the DS error correction. 
#              If "False", the “Sim_posmod” will be directly used for DS error correction.

import numpy as np
import random
from tqdm import tqdm

def errorCorrection_PatchDS_3D(Sim_posmod, Ti_primod, wel_obs_data, i_dim, j_dim, k_dim, run_smooth):
    ###################################
    ####set up some constant variables
    ##################################
    
    realnums = len(Sim_posmod)
    d_obs_loc_val = wel_obs_data
    Well_Amount = len(wel_obs_data[:,0,0])
    Well_Depth = len(wel_obs_data[0,:,0])
    
#     Realization_Index = layer_num
    Realization_Layer = k_dim
    Realization_Height = j_dim
    Realization_Width = i_dim

#     TI_Index = layer_num
    TI_Layer = Realization_Layer
    TI_Height = Realization_Height
    TI_Width = Realization_Width
    
    corrected_posterior = []
    for real_num in tqdm(range(realnums)):
        ### Posterior to re-construct

        
        if run_smooth == True: 
            # use the statistical filter to smooth out noise (isolated points)
            realization = Sim_posmod[real_num].reshape(Realization_Layer,Realization_Height,Realization_Width)
            old_realization = np.zeros((Realization_Layer,Realization_Height,Realization_Width))
            smooth_x = 1
            smooth_y = 1
            noisedPattern = []
            for realization_z in range(Realization_Layer):
                for realization_y in range(Realization_Height):
                    for realization_x in range(Realization_Width):
                        bottom_y = max(realization_y-smooth_y,0)
                        up_y = min(realization_y+smooth_y+1,Realization_Height)
                        bottom_x = max(realization_x-smooth_x,0)
                        up_x = min(realization_x+smooth_x+1,Realization_Width)
                        noisedPattern = []
                        for sample_y in range(bottom_y,up_y):
                            for sample_x in range(bottom_x,up_x):
                                noisedPattern.append(realization[realization_z][sample_y][sample_x])
                        noisedPattern.sort();
                        sample_value = noisedPattern[int(len(noisedPattern)/2)];
                        old_realization[realization_z][realization_y][realization_x] = sample_value
            
        else:
            old_realization = Sim_posmod[real_num].reshape(Realization_Layer,Realization_Height,Realization_Width)
        
        
        ### Prior as training image
        TI = Ti_primod[real_num].reshape(TI_Layer,TI_Height,TI_Width)
  
        
        #############################################
        ##### set up some temporary variables  #####
        #############################################  

        well_index = 0
        well_depth_index = 0
        well_value = 0
        well_x = 0
        well_y = 0
        well_z = 0

        realization_value = 0
        realization_z = 0
        realization_y = 0
        realization_x = 0

        point_x = 0
        point_y = 0
        point_z = 0

        # this is the new realization
        new_realization = np.empty((Realization_Layer,Realization_Height,Realization_Width))
        new_realization_operation = np.empty((Realization_Layer,Realization_Height,Realization_Width)) # 0 no operation 1 mismatch 2 match

        # initial the new realization
        for realization_z in range(Realization_Layer):
            for realization_y in range(Realization_Height):
                for realization_x in range(Realization_Width):
                    new_realization[realization_z][realization_y][realization_x] = -1.0;
                    
        #############################################
        ##### find the mismatch location #####
        #############################################
        for well_index in range(Well_Amount):
            # attention: the well data in d_obs_loc_val is reservely ranked

            well_x = int(d_obs_loc_val[well_index][well_depth_index][0])   # this is the x corridinate of well data
            well_y = int(d_obs_loc_val[well_index][well_depth_index][1])   # this is the y corridinate of well data

            #print("the " + str(well_index)+"-th well x: "+ str(well_x)+" y: "+str(well_y))

            for well_depth_index in range(Well_Depth):
                well_z = int(d_obs_loc_val[well_index][well_depth_index][2])   # this is the z corridinate of well data
                well_value = d_obs_loc_val[well_index][well_depth_index][3]   # this is the value of well data

                new_realization[well_z][well_y][well_x] = well_value;
                new_realization_operation[well_z][well_y][well_x] = 2;

                realization_value = old_realization[well_z][well_y][well_x] # this is the value of old realization

                # key 1: determine the points that need to be resimulated
                if well_value != realization_value:
                    bottom_z = max(well_z-1,0)
                    up_z = min(well_z+2,Realization_Layer)
                    #bottom_z = max(well_z,0)
                    #up_z = min(well_z+1,Realization_Layer)

                    bottom_y = max(well_y-1,0)
                    up_y = min(well_y+2,Realization_Height)
                    bottom_x = max(well_x-1,0)
                    up_x = min(well_x+2,Realization_Width)

                    for point_z in range(bottom_z,up_z):
                        for point_y in range (bottom_y,up_y):
                            for point_x in range (bottom_x,up_x):
                                if(new_realization_operation[point_z][point_y][point_x]==0):
                                    new_realization_operation[point_z][point_y][point_x] = 1;

        #############################################
        ###### set up the parameter of DS ##########
        #############################################
        Neighborhood = 30
        Threshold = 0.01
        Fraction = Realization_Layer * Realization_Height * Realization_Width * 0.6
        weight_new = 0.6
        weight_old = 0.4
        
        #############################################
        ####### perform DS to correct mismatch ######
        #############################################
        # this cell is used for defining variables

        probability = 20
        probability_step = 3 

        pattern_new_value = []
        pattern_new_x = []
        pattern_new_y = []
        pattern_new_z = []

        pattern_old_value = []
        pattern_old_x = []
        pattern_old_y = []
        pattern_old_z = []

        circle = 1

        continueGather = True

        distance = 10.0
        distance_new = 10.0
        distance_old = 10.0
        distance_min = 10.0
        sample_x = 0
        sample_y = 0
        sample_z = 0
        sample_value = 0
        sample_Amount = 0
        conditioning_pattern_value = 0
        training_pattern_value = 0
        pattern_index = 0

        bottom_z = 0
        up_z = 0
        bottom_y = 0
        up_y = 0
        bottom_x = 0
        up_x = 0

        loop_status = True
        
        ########################################################
        ######### MPS - DIRECT SAMPLING MAIN FUNCTION #########
        ########################################################        
        # this is the solution and patch direct sampling: (1) a limited searching area (2) paste a patch at a time 
        # core idea: (1) the solution of the closest simulated point privides the guidance (2) simulate a patch at a time

        # control parameter
        loop_CloseSolution_Patch_count = 0
        loop_CloseSolution_Patch_MaxCount = 10

        # parameter: the template stripe to collect the informed points
        stride_x = 1
        stride_y = 1
        stride_z = 4
        circle_x = 0
        circle_y = 0
        circle_z = 0

        # parameter: the size of simulated patch
        radius_simulationPatch_x = 4  # the size of simulation patch is 2*radius+1
        radius_simulationPatch_y = 4
        radius_simulationPatch_z = 1
        radius_simulationPatch_min_x = 1
        radius_simulationPatch_min_y = 1
        radius_simulationPatch_min_z = 0
        radius_simulationPatch_slope = 1

        # parameter: the size of previous soluation search 
        radius_solution_x = radius_simulationPatch_x + 1
        radius_solution_y = radius_simulationPatch_y + 1
        radius_solution_z = 0

        # parameter: the size of current searching area: closest solution method
        radius_search_solution_x = 3
        radius_search_solution_y = 3
        radius_search_solution_z = 1

        # parameter: the size of current searching area: exhaustive search method
        radius_search_exhaustive_x = 10
        radius_search_exhaustive_y = 10
        radius_search_exhaustive_z = 3

        new_realization_temporary = np.zeros((Realization_Layer,Realization_Height,Realization_Width)) 
        new_realization_operation_temporary = np.zeros((Realization_Layer,Realization_Height,Realization_Width))

        solution_array_x = np.zeros((Realization_Layer,Realization_Height,Realization_Width))
        solution_array_y = np.zeros((Realization_Layer,Realization_Height,Realization_Width))
        solution_array_z = np.zeros((Realization_Layer,Realization_Height,Realization_Width))
        # initial the new realization
        for realization_z in range(Realization_Layer):    
            for realization_y in range(Realization_Height):
                for realization_x in range(Realization_Width):
                    solution_array_x[realization_z][realization_y][realization_x] = -1;
                    solution_array_y[realization_z][realization_y][realization_x] = -1;
                    solution_array_z[realization_z][realization_y][realization_x] = -1;

        solution_x = 0
        solution_y = 0
        solution_z = 0
        previous_x = 0
        previous_y = 0
        previous_z = 0
        distance_min_x = 0
        distance_min_y = 0
        distance_min_z = 0

        patch_realization_z = 0
        patch_realization_y = 0
        patch_realization_x = 0
        patch_TI_z = 0
        patch_TI_y = 0
        patch_TI_x = 0

        loop_status = True
        while loop_CloseSolution_Patch_count < loop_CloseSolution_Patch_MaxCount and loop_status == True:
            loop_CloseSolution_Patch_count += 1
            loop_status = False
            #print("start to correct the error: extend the circle")
#             print(str(loop_CloseSolution_Patch_count)+"-th loop:")

            radius_simulationPatch_x = max(radius_simulationPatch_min_x,radius_simulationPatch_x-radius_simulationPatch_slope)
            radius_simulationPatch_y = max(radius_simulationPatch_min_y,radius_simulationPatch_y-radius_simulationPatch_slope)
            radius_simulationPatch_z = max(radius_simulationPatch_min_z,radius_simulationPatch_z-radius_simulationPatch_slope)

            probability += probability_step

            for realization_z in range(Realization_Layer):
                for realization_y in range(Realization_Height):
                    for realization_x in range(Realization_Width):
                        new_realization_temporary[realization_z][realization_y][realization_x] = -1
                        new_realization_operation_temporary[realization_z][realization_y][realization_x] = 0
#             '''        
#             print("before processing:")
#             for realization_z in range(Realization_Layer):  
#                 print("slice_z: "+str(realization_z))
#                 plt.imshow(new_realization[realization_z],vmin=-1, vmax=3)
#                 plt.show()
#                 plt.imshow(new_realization_operation[realization_z],vmin=-1, vmax=3)
#                 plt.show()
#             '''

            for realization_z in range(Realization_Layer):  
                for realization_y in range(Realization_Height):
                    for realization_x in range(Realization_Width):

                        # if this point needs to be simulated
                        if(new_realization_operation[realization_z][realization_y][realization_x]==1 and new_realization_temporary[realization_z][realization_y][realization_x]==-1.0):
                            pattern_new_value.clear()
                            pattern_new_z.clear()
                            pattern_new_y.clear()
                            pattern_new_x.clear()
                            pattern_old_value.clear()
                            pattern_old_z.clear()
                            pattern_old_y.clear()
                            pattern_old_x.clear()

                            # gather the conditioning points
                            circle = 1
                            continueGather = True
                            while continueGather :

                                circle_x = int((circle-1)/stride_x);
                                circle_y = int((circle-1)/stride_y);
                                circle_z = int((circle-1)/stride_z);

                                #print("circle:"+str(circle))
                                #print("circle_z:"+str(circle_z))

                                circle += 1

                                bottom_z = max(realization_z-circle_z,0)
                                up_z = min(realization_z+circle_z+1,Realization_Layer)
                                bottom_y = max(realization_y-circle_y,0)
                                up_y = min(realization_y+circle_y+1,Realization_Height)
                                bottom_x = max(realization_x-circle_x,0)
                                up_x = min(realization_x+circle_x+1,Realization_Width)

                                for point_z in range(bottom_z,up_z):
                                    for point_y in range(bottom_y,up_y):
                                        for point_x in range(bottom_x,up_x):
                                            if continueGather == False:
                                                continue;
                                            if(abs(point_z-realization_z)==circle_z or 
                                               abs(point_y-realization_y)==circle_y or 
                                               abs(point_x-realization_x)==circle_x):
                                                if(new_realization[point_z][point_y][point_x]!=-1):
                                                    pattern_new_value.append(new_realization[point_z][point_y][point_x])
                                                    pattern_new_z.append(point_z-realization_z)
                                                    pattern_new_y.append(point_y-realization_y)
                                                    pattern_new_x.append(point_x-realization_x)
                                                elif (old_realization[point_z][point_y][point_x]!=-1):
                                                    if random.randint(0,99) < probability :
                                                        pattern_old_value.append(old_realization[point_z][point_y][point_x])
                                                        pattern_old_z.append(point_z-realization_z)
                                                        pattern_old_y.append(point_y-realization_y)
                                                        pattern_old_x.append(point_x-realization_x)

                                                if len(pattern_new_value)+len(pattern_old_value) > Neighborhood :
                                                    continueGather = False


                            #find the closest simulated point
                            distance = 10.0
                            distance_min = 10000.0

                            bottom_z = max(realization_z-radius_solution_z,0)
                            up_z = min(realization_z+radius_solution_z+1,Realization_Height)
                            bottom_y = max(realization_y-radius_solution_y,0)
                            up_y = min(realization_y+radius_solution_y+1,Realization_Height)
                            bottom_x = max(realization_x-radius_solution_x,0)
                            up_x = min(realization_x+radius_solution_x+1,Realization_Width)

                            for point_z in range(bottom_z,up_z):
                                for point_y in range(bottom_y,up_y):
                                    for point_x in range(bottom_x,up_x):
                                        if solution_array_z[point_z][point_y][point_x] != -1:
                                            distance = (point_z-realization_z)*(point_z-realization_z)
                                            + (point_y-realization_y)*(point_y-realization_y)
                                            + (point_x-realization_x)*(point_x-realization_x)

                                            if distance < distance_min:
                                                solution_x = int(solution_array_x[point_z][point_y][point_x])
                                                solution_y = int(solution_array_y[point_z][point_y][point_x])
                                                solution_z = int(solution_array_z[point_z][point_y][point_x])
                                                previous_x = point_x
                                                previous_y = point_y
                                                previous_z = point_z
                                                distance_min = distance

                            # this point has the close simulated point
                            if distance_min < 1000.0:
                                #print("find the close solution")
                                distance = 10.0
                                distance_min = 10.0

                                bottom_z = max(realization_z-previous_z+solution_z-radius_search_solution_z,0)
                                up_z = min(realization_z-previous_z+solution_z+radius_search_solution_z+1,Realization_Layer)
                                bottom_y = max(realization_y-previous_y+solution_y-radius_search_solution_y,0)
                                up_y = min(realization_y-previous_y+solution_y+radius_search_solution_y+1,Realization_Height)
                                bottom_x = max(realization_x-previous_x+solution_x-radius_search_solution_x,0)
                                up_x = min(realization_x-previous_x+solution_x+radius_search_solution_x+1,Realization_Width)

                                for sample_z in range(bottom_z,up_z):
                                    for sample_y in range(bottom_y,up_y):
                                        for sample_x in range(bottom_x,up_x):
                                            distance_new = 0.0
                                            #this is the distance to the new data
                                            for pattern_index in range(len(pattern_new_value)):
                                                point_z = pattern_new_z[pattern_index] + sample_z
                                                point_y = pattern_new_y[pattern_index] + sample_y
                                                point_x = pattern_new_x[pattern_index] + sample_x
                                                if (point_z >= 0 and point_z < TI_Layer
                                                    and point_y >= 0 and point_y < TI_Height
                                                    and point_x >= 0 and point_x < TI_Width):
                                                    training_pattern_value = TI[point_z][point_y][point_x]
                                                    conditioning_pattern_value = pattern_new_value[pattern_index]
                                                    if(training_pattern_value != conditioning_pattern_value):
                                                        distance_new += 1
                                                else:
                                                    distance_new += 1
                                            if len(pattern_new_value)==0 :
                                                distance_new = 0
                                            else:
                                                distance_new = distance_new / float(len(pattern_new_value))
                                            # this is the distance to the old data
                                            distance_old = 0.0
                                            for pattern_index in range(len(pattern_old_value)):
                                                point_z = pattern_old_z[pattern_index] + sample_z
                                                point_y = pattern_old_y[pattern_index] + sample_y
                                                point_x = pattern_old_x[pattern_index] + sample_x
                                                if (point_z >= 0 and point_z < TI_Layer
                                                   and point_y >= 0 and point_y < TI_Height
                                                        and point_x >= 0 and point_x < TI_Width):
                                                    training_pattern_value = TI[point_z][point_y][point_x]
                                                    conditioning_pattern_value = pattern_old_value[pattern_index]
                                                    if(training_pattern_value != conditioning_pattern_value):
                                                        distance_old += 1
                                                else:
                                                    distance_old += 1
                                            if len(pattern_old_value) ==0:
                                                distance_old = 0
                                            else:
                                                distance_old = distance_old / float(len(pattern_old_value))

                                            distance = weight_new * distance_new + weight_old * distance_old

                                            if(distance < distance_min):
                                                distance_min = distance
                                                sample_value = TI[sample_z][sample_y][sample_x]
                                                distance_min_x = sample_x
                                                distance_min_y = sample_y
                                                distance_min_z = sample_z
                            # this point does not have the clost simulated point         
                            else:      
                                #an local exhaustive search in TI
                                #print("local exhaustive search")
                                distance = 10.0
                                distance_min = 10.0

                                bottom_z = max(realization_z-radius_search_exhaustive_z,0)
                                up_z = min(realization_z+radius_search_exhaustive_z,Realization_Layer)
                                bottom_y = max(realization_y-radius_search_exhaustive_y,0)
                                up_y = min(realization_y+radius_search_exhaustive_y,Realization_Height)
                                bottom_x = max(realization_x-radius_search_exhaustive_x,0)
                                up_x = min(realization_x+radius_search_exhaustive_x,Realization_Width)

                                for sample_z in range(bottom_z,up_z):
                                    for sample_y in range(bottom_y,up_y):
                                        for sample_x in range(bottom_x,up_x):
                                            # this is the distance to the new data
                                            distance_new = 0.0
                                            for pattern_index in range(len(pattern_new_value)):
                                                point_z = pattern_new_z[pattern_index] + sample_z
                                                point_y = pattern_new_y[pattern_index] + sample_y
                                                point_x = pattern_new_x[pattern_index] + sample_x
                                                if (point_z >= 0 and point_z < TI_Layer
                                                    and point_y >= 0 and point_y < TI_Height
                                                    and point_x >= 0 and point_x < TI_Width):
                                                    training_pattern_value = TI[point_z][point_y][point_x]
                                                    conditioning_pattern_value = pattern_new_value[pattern_index]
                                                    if(training_pattern_value != conditioning_pattern_value):
                                                        distance_new += 1
                                                else:
                                                    distance_new += 1
                                            if len(pattern_new_value)==0 :
                                                distance_new = 0
                                            else:
                                                distance_new = distance_new / float(len(pattern_new_value))
                                            # this is the distance to the old data
                                            distance_old = 0.0
                                            for pattern_index in range(len(pattern_old_value)):
                                                point_z = pattern_old_z[pattern_index] + sample_z
                                                point_y = pattern_old_y[pattern_index] + sample_y
                                                point_x = pattern_old_x[pattern_index] + sample_x
                                                if (point_z >= 0 and point_z < TI_Layer
                                                    and point_y >= 0 and point_y < TI_Height
                                                    and point_x >= 0 and point_x < TI_Width):
                                                    training_pattern_value = TI[point_z][point_y][point_x]
                                                    conditioning_pattern_value = pattern_old_value[pattern_index]
                                                    if(training_pattern_value != conditioning_pattern_value):
                                                        distance_old += 1
                                                else:
                                                    distance_old += 1
                                            if len(pattern_old_value) ==0:
                                                distance_old = 0
                                            else:
                                                distance_old = distance_old / float(len(pattern_old_value))

                                            distance = weight_new * distance_new + weight_old * distance_old

                                            if(distance < distance_min):
                                                distance_min = distance
                                                distance_min_x = sample_x
                                                distance_min_y = sample_y
                                                distance_min_z = sample_z

                            #assign the training point to the conditioning point
                            solution_array_x[realization_z][realization_y][realization_x] = distance_min_x
                            solution_array_y[realization_z][realization_y][realization_x] = distance_min_y
                            solution_array_z[realization_z][realization_y][realization_x] = distance_min_z
                            #print("minimum pattern distance is" + str(distance_min));
                            #print("the distance between solution and simulated point is" + str(max(abs(distance_min_x-realization_x),abs(distance_min_y-realization_y))));

                            # paste a patch in the simulation domain
                            for point_z in range(-radius_simulationPatch_z,+radius_simulationPatch_z+1):
                                for point_y in range(-radius_simulationPatch_y,+radius_simulationPatch_y+1):
                                    for point_x in range(-radius_simulationPatch_x,+radius_simulationPatch_x+1):
                                        patch_realization_z = realization_z + point_z
                                        patch_realization_y = realization_y + point_y
                                        patch_realization_x = realization_x + point_x
                                        patch_TI_z = distance_min_z + point_z
                                        patch_TI_y = distance_min_y + point_y
                                        patch_TI_x = distance_min_x + point_x

                                        if (patch_realization_z >=0 and patch_realization_z < Realization_Layer
                                            and patch_realization_y >=0 and patch_realization_y < Realization_Height
                                            and patch_realization_x >= 0 and patch_realization_x < Realization_Width):

                                            if new_realization[patch_realization_z][patch_realization_y][patch_realization_x] == -1.0:
                                                if  (patch_TI_z >=0 and patch_TI_z < TI_Layer
                                                     and patch_TI_y >=0 and patch_TI_y < TI_Height
                                                     and patch_TI_x >= 0 and patch_TI_x < TI_Width):

                                                    sample_value = TI[patch_TI_z][patch_TI_y][patch_TI_x]
                                                    new_realization_temporary[patch_realization_z][patch_realization_y][patch_realization_x] = sample_value
                                                    new_realization_operation_temporary[patch_realization_z][patch_realization_y][patch_realization_x] = 2

                                                    #print("the simulated point is x:"+str(patch_realization_x) + " y: "+str(patch_realization_y)+" z: "+str(patch_realization_z))

                                                    sample_z = patch_realization_z
                                                    # determine the points that need to be simulated in the next iteration
                                                    # bottom side
                                                    if point_y == -radius_simulationPatch_y and sample_value != old_realization[patch_realization_z][patch_realization_y][patch_realization_x]:
                                                        sample_y = patch_realization_y - 1
                                                        if sample_y >= 0:
                                                            loop_status = True
                                                            for sample_x in range(max(patch_realization_x-1,0),min(patch_realization_x+2,Realization_Width)):
                                                                if new_realization_operation[sample_z][sample_y][sample_x] == 0 and new_realization_operation_temporary[sample_z][sample_y][sample_x]==0:
                                                                        new_realization_operation_temporary[sample_z][sample_y][sample_x] = 1
                                                    # top side
                                                    elif point_y == radius_simulationPatch_y and sample_value != old_realization[patch_realization_z][patch_realization_y][patch_realization_x]:
                                                        sample_y = patch_realization_y + 1
                                                        if sample_y < Realization_Height:
                                                            loop_status = True
                                                            for sample_x in range(max(patch_realization_x-1,0),min(patch_realization_x+2,Realization_Width)):
                                                                if new_realization_operation[sample_z][sample_y][sample_x] == 0 and new_realization_operation_temporary[sample_z][sample_y][sample_x]==0:
                                                                        new_realization_operation_temporary[sample_z][sample_y][sample_x] = 1 

                                                    # left side
                                                    if point_x == -radius_simulationPatch_x and sample_value != old_realization[patch_realization_z][patch_realization_y][patch_realization_x]:
                                                        sample_x = patch_realization_x - 1
                                                        if sample_x >= 0:
                                                            loop_status = True
                                                            for sample_y in range(max(patch_realization_y-1,0),min(patch_realization_y+2,Realization_Height)):
                                                                if new_realization_operation[sample_z][sample_y][sample_x] == 0 and new_realization_operation_temporary[sample_z][sample_y][sample_x]==0:
                                                                        new_realization_operation_temporary[sample_z][sample_y][sample_x] = 1
                                                    # right side
                                                    elif point_x == radius_simulationPatch_x and sample_value != old_realization[patch_realization_z][patch_realization_y][patch_realization_x]:
                                                        sample_x = patch_realization_x + 1
                                                        if sample_x < Realization_Width:
                                                            loop_status = True
                                                            for sample_y in range(max(patch_realization_y-1,0),min(patch_realization_y+2,Realization_Height)):
                                                                if new_realization_operation[sample_z][sample_y][sample_x] == 0 and new_realization_operation_temporary[sample_z][sample_y][sample_x]==0:
                                                                        new_realization_operation_temporary[sample_z][sample_y][sample_x] = 1



                        # these points are simulated by the previous block
                        elif(new_realization_operation[realization_z][realization_y][realization_x]==1 
                             and new_realization_temporary[realization_z][realization_y][realization_x]!=-1.0):
                            sample_value = 0;

                        # these points have already been processed
                        elif(new_realization_operation[realization_z][realization_y][realization_x]==2): 
                            sample_value = new_realization[realization_z][realization_y][realization_x]
                            new_realization_temporary[realization_z][realization_y][realization_x] = sample_value
                            new_realization_operation_temporary[realization_z][realization_y][realization_x] = 2 

                        elif (new_realization_operation[realization_z][realization_y][realization_x]==0 
                              and new_realization_temporary[realization_z][realization_y][realization_x]!=-1):
                            sample_value = 0;

                        else:
                            sample_value = new_realization[realization_z][realization_y][realization_x]
                            new_realization_temporary[realization_z][realization_y][realization_x] = sample_value
                            new_realization_operation_temporary[realization_z][realization_y][realization_x] = 0 

            for realization_z in range(Realization_Layer):
                for realization_y in range(Realization_Height):
                    for realization_x in range(Realization_Width):
                        sample_value = new_realization_temporary[realization_z][realization_y][realization_x]
                        new_realization[realization_z][realization_y][realization_x] = sample_value
                        sample_value = new_realization_operation_temporary[realization_z][realization_y][realization_x]
                        new_realization_operation[realization_z][realization_y][realization_x] = sample_value



        for realization_z in range(Realization_Layer):
            for realization_y in range(Realization_Height):
                for realization_x in range(Realization_Width):
                    if new_realization[realization_z][realization_y][realization_x] != -1.0:
                        conditioning_pattern_value = new_realization[realization_z][realization_y][realization_x]
                        new_realization_temporary[realization_z][realization_y][realization_x] = conditioning_pattern_value
                    else:
                        conditioning_pattern_value = old_realization[realization_z][realization_y][realization_x]
                        new_realization_temporary[realization_z][realization_y][realization_x] = conditioning_pattern_value
                        
        corrected_posterior.append(new_realization_temporary)
    corrected_posterior = np.asarray(corrected_posterior)
    return corrected_posterior


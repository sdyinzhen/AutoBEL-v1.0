#Authors: Chen Zuo, David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: Oct 15, 2018

# This is the function to correct the posterior residual errors at mismatched wells using "MPS Direct Sampling" for one layer. 
# This function returns the DS corrected 2D layers of posterior. 
# e.g. corrected_posterior = errorCorrection_PatchDS_2D(Sim_posmod, Ti_primod, wel_obs_data, layer_num, i_dim, j_dim, k_dim)

# Sim_posmod: the 2D array matrix that constains all the posterior realizations(each realization is vectorized) that need to correct, 
#            Sim_posmod = realization_number x total_grid_number_per_model(i_dim x j_dim x k_dim)
# Ti_primod: the 2D array matrix that contains all the prior realizations(each realization is vectorized), used as trainning image, same dimension as the "Sim_posmod"
# wel_obs_data: the 3D arrays that contains the well observation data,
#           wel_obs_data = well_number x well_sample_points x 4 (i_location, j_location, k_location, and observation value)
# layer_num: the number of the layer to reconstruct. 
# i_dim, j_dim, k_dim: the i, j, k dimension of the model
import numpy as np
import random
from tqdm import tqdm
def errorCorrection_PatchDS_2D(Sim_posmod, Ti_primod, wel_obs_data, layer_num, i_dim, j_dim, k_dim):
    ###################################
    ####set up some constant variables
    ##################################
    
    realnums = len(Sim_posmod)
    d_obs_loc_val = wel_obs_data
    Well_Amount = len(d_obs_loc_val[:,0,0])
    Well_Depth_Sum = len(d_obs_loc_val[0,:,0])
    
    Realization_Index = layer_num
    Realization_Layer = k_dim
    Realization_Height = j_dim
    Realization_Width = i_dim

    TI_Index = layer_num
    TI_Layer = Realization_Layer
    TI_Height = Realization_Height
    TI_Width = Realization_Width
    
    corrected_posterior = []
    for real_num in tqdm(range(realnums)):
        ### Posterior to re-construct
        pos_fac_var = Sim_posmod[real_num,:].reshape(Realization_Layer,Realization_Height,Realization_Width)
        old_realization = pos_fac_var[Realization_Index,:,:]
        ### Prior as training image

        pri_fac_var = Ti_primod[real_num,:].reshape(TI_Layer,TI_Height,TI_Width)
        TI = pri_fac_var[TI_Index,:,:]


        #############################################
        ##### set up some temporary variables  #####
        #############################################
        well_index = 0
        well_value = 0
        well_x = 0
        well_y = 0

        realization_value = 0
        realization_y = 0
        realization_x = 0

        point_x = 0
        point_y = 0

        # this is the new realization
        new_realization = np.empty((Realization_Height,Realization_Width))
        new_realization_operation = np.zeros((Realization_Height,Realization_Width)) # 0 no operation 1 mismatch 2 match

        # initial the new realization
        for realization_y in range(Realization_Height):
            for realization_x in range(Realization_Width):
                new_realization[realization_y][realization_x] = -1.0;
        #plt.imshow(new_realization,vmin=-1, vmax=3)
        #############################################
        ##### find the mismatch location #####
        #############################################
        for well_index in range(Well_Amount):
            # attention the well data in d_obs_loc_val is reservely ranked
            well_x = int(d_obs_loc_val[well_index,Well_Depth_Sum-1-Realization_Index,0])   # this is the x corridinate of well data
            well_y = int(d_obs_loc_val[well_index,Well_Depth_Sum-1-Realization_Index,1])  # this is the y corridinate of well data
            well_value = int(d_obs_loc_val[well_index,Well_Depth_Sum-1-Realization_Index,3])  # this is the value of well data
            realization_value = pos_fac_var[0,int(well_y),int(well_x)] # this is the value of old realization
            if well_value != realization_value:
                new_realization[well_y][well_x] = well_value
        #         print("Mismatch x: "+ str(well_x) + " y: "+str(well_y)+" well data: "+ str(well_value))
                for point_y in range(well_y-1,well_y+2):
                    for point_x in range(well_x-1,well_x+2):
                        if point_y>=0 and point_y < Realization_Height and point_x>=0 and point_x<Realization_Width:
                            new_realization_operation[point_y][point_x] = 1;  # the surrounding points need to be updated
                new_realization_operation[well_y][well_x] = 2;  # the mismatch point does not need to be updated
        #     else:
        #         print("Match x: "+ str(well_x) + " y: "+str(well_y)+" well data: "+ str(well_value))

        # plt.imshow(new_realization,vmin=-1, vmax=3)
        # plt.show()
        # plt.imshow(new_realization_operation,vmin=-1, vmax=3)
        # plt.show()


        #############################################
        ###### set up the parameter of DS ##########
        #############################################
        Neighborhood = 30
        Threshold = 0.01
        Fraction = Realization_Height * Realization_Width * 0.6
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

        pattern_old_value = []
        pattern_old_x = []
        pattern_old_y = []

        circle = 1

        continueGather = True

        distance = 10.0
        distance_new = 10.0
        distance_old = 10.0
        distance_min = 10.0
        sample_x = 0
        sample_y = 0
        sample_value = 0
        sample_Amount = 0
        conditioning_pattern_value = 0
        training_pattern_value = 0
        pattern_index = 0

        loop_status = True

        TI_judge = np.zeros((TI_Height,TI_Width)) # 0 non checked 1 need to be checked 2 already checked
        new_realization_temporary = np.zeros((Realization_Height,Realization_Width)) 
        new_realization_operation_temporary = np.zeros((Realization_Height,Realization_Width))


        ########################################################
        ######### MPS - DIRECT SAMPLING MAIN FUNCTION #########
        ########################################################

        ### this is the solution and patch direct sampling: (1) a limited searching area (2) paste a patch at a time 
        # core idea: (1) the solution of the closest simulated point privides the guidance (2) simulate a patch at a time

        # parameter: the size of simulated patch
        radius_simulationPatch = 5  # the size of simulation patch is 2*radius+1
        radius_simulationPatch_slope = 1
        radius_simulationPatch_min = 1

        # parameter: the size of previous soluation search 
        radius_solution = radius_simulationPatch + 1

        # parameter: the size of current searching area
        radius_search = 2
        radius_search_slope = 0
        radius_search_min = 2

        loop_CloseSolution_Patch_count = 0
        loop_CloseSolution_Patch_MaxCount = 100

        new_realization_temporary = np.zeros((Realization_Height,Realization_Width)) 
        new_realization_operation_temporary = np.zeros((Realization_Height,Realization_Width))

        solution_array_x = np.zeros((Realization_Height,Realization_Width))
        solution_array_y = np.zeros((Realization_Height,Realization_Width))
        # initial the new realization
        for realization_y in range(Realization_Height):
            for realization_x in range(Realization_Width):
                solution_array_x[realization_y][realization_x] = -1;
                solution_array_y[realization_y][realization_x] = -1;

        solution_x = 0
        solution_y = 0
        previous_x = 0
        previous_y = 0
        distance_min_x = 0
        distance_min_y = 0

        patch_realization_y = 0
        patch_realization_x = 0
        patch_TI_y = 0
        patch_TI_x = 0

        loop_status = True
        while loop_CloseSolution_Patch_count < loop_CloseSolution_Patch_MaxCount and loop_status == True:
            loop_CloseSolution_Patch_count += 1
            loop_status = False
            #print("start to correct the error: extend the circle")
        #     print(str(loop_CloseSolution_Patch_count)+"-th loop:")

            radius_simulationPatch = max(radius_simulationPatch_min,radius_simulationPatch-radius_simulationPatch_slope)
            radius_search = max(radius_search_min,radius_search-radius_search_slope)

            probability += probability_step

            for realization_y in range(Realization_Height):
                for realization_x in range(Realization_Width):
                    new_realization_temporary[realization_y][realization_x] = -1
            new_realization_operation_temporary = np.zeros((Realization_Height,Realization_Width))

            for realization_y in range(Realization_Height):
                for realization_x in range(Realization_Width):

                    # if this point needs to be simulated
                    if(new_realization_operation[realization_y][realization_x]==1 and new_realization_temporary[realization_y][realization_x]==-1):

                        #print("the mismatch point is x:"+str(realization_x) + " y: "+str(realization_y) )       

                        #gather the information for neighborhood
                        #a conditioning pattern extraction
                        pattern_new_value.clear()
                        pattern_new_y.clear()
                        pattern_new_x.clear()
                        pattern_old_value.clear()
                        pattern_old_y.clear()
                        pattern_old_x.clear()

                        # gather the conditioning points
                        circle = 0
                        continueGather = True
                        while continueGather :
                            circle += 1
                            # top line
                            point_y = -circle
                            for point_x in range(-circle,+circle+1):
                                if continueGather and realization_y+point_y >= 0 and                                 realization_y+point_y < Realization_Height and realization_x+point_x>=0                                 and realization_x+point_x<Realization_Width:
                                    if(new_realization[realization_y+point_y][realization_x+point_x]!=-1):
                                        pattern_new_value.append(new_realization[realization_y+point_y][realization_x+point_x])
                                        pattern_new_y.append(point_y)
                                        pattern_new_x.append(point_x)
                                    elif (old_realization[realization_y+point_y][realization_x+point_x]!=-1):
                                        if random.randint(0,99) < probability :
                                            pattern_old_value.append(old_realization[realization_y+point_y][realization_x+point_x])
                                            pattern_old_y.append(point_y)
                                            pattern_old_x.append(point_x)

                                    if len(pattern_new_value)+len(pattern_old_value) > Neighborhood :
                                        continueGather = False

                            # right line
                            point_x = +circle
                            for point_y in range(-circle+1,+circle+1):
                                if continueGather and realization_y+point_y >= 0 and realization_y+point_y < Realization_Height and realization_x+point_x>=0 and realization_x+point_x<Realization_Width:
                                    if(new_realization[realization_y+point_y][realization_x+point_x]!=-1):
                                        pattern_new_value.append(new_realization[realization_y+point_y][realization_x+point_x])
                                        pattern_new_y.append(point_y)
                                        pattern_new_x.append(point_x)
                                    elif (old_realization[realization_y+point_y][realization_x+point_x]!=-1):
                                        if random.randint(0,99) < probability :
                                            pattern_old_value.append(old_realization[realization_y+point_y][realization_x+point_x])
                                            pattern_old_y.append(point_y)
                                            pattern_old_x.append(point_x)

                                    if len(pattern_new_value)+len(pattern_old_value) > Neighborhood :
                                        continueGather = False

                             # bottom line
                            point_y = +circle
                            for point_x in range(-circle,circle):
                                if continueGather and realization_y+point_y >= 0 and realization_y+point_y < Realization_Height and realization_x+point_x>=0 and realization_x+point_x<Realization_Width:
                                    if(new_realization[realization_y+point_y][realization_x+point_x]!=-1):
                                        pattern_new_value.append(new_realization[realization_y+point_y][realization_x+point_x])
                                        pattern_new_y.append(point_y)
                                        pattern_new_x.append(point_x)
                                    elif (old_realization[realization_y+point_y][realization_x+point_x]!=-1):
                                        if random.randint(0,99) < probability :
                                            pattern_old_value.append(old_realization[realization_y+point_y][realization_x+point_x])
                                            pattern_old_y.append(point_y)
                                            pattern_old_x.append(point_x)

                                    if len(pattern_new_value)+len(pattern_old_value) > Neighborhood :
                                        continueGather = False

                            # left line
                            point_x = -circle
                            for point_y in range(circle-1,circle):                    
                                if continueGather and realization_y+point_y >= 0 and realization_y+point_y < Realization_Height and realization_x+point_x>=0 and realization_x+point_x<Realization_Width:
                                    if(new_realization[realization_y+point_y][realization_x+point_x]!=-1):
                                        pattern_new_value.append(new_realization[realization_y+point_y][realization_x+point_x])
                                        pattern_new_y.append(point_y)
                                        pattern_new_x.append(point_x)
                                    elif (old_realization[realization_y+point_y][realization_x+point_x]!=-1):
                                        if random.randint(0,99) < probability :
                                            pattern_old_value.append(old_realization[realization_y+point_y][realization_x+point_x])
                                            pattern_old_y.append(point_y)
                                            pattern_old_x.append(point_x)

                                    if len(pattern_new_value)+len(pattern_old_value) > Neighborhood :
                                        continueGather = False

                        #find the closest simulated point
                        distance = 10.0
                        distance_min = 10000.0
                        for point_y in range(realization_y-radius_solution,realization_y+radius_solution+1):
                            for point_x in range(realization_x-radius_solution,realization_x+radius_solution+1):
                                if point_y >=0 and point_y < TI_Height and point_x >= 0 and point_x < TI_Width and solution_array_y[point_y][point_x] != -1 :
                                    distance = (point_y-realization_y)*(point_y-realization_y) + (point_x-realization_x)*(point_x-realization_x)
                                    if distance < distance_min:
                                        solution_x = int(solution_array_x[point_y][point_x])
                                        solution_y = int(solution_array_y[point_y][point_x])
                                        previous_x = point_x
                                        previous_y = point_y
                                        distance_min = distance

                        # this point has the close simulated point
                        if distance_min < 10000.0:
                            #print("find the close solution")
                            distance = 10.0
                            distance_min = 10.0
                            sample_Amount = 0
                            for sample_y in range(realization_y-previous_y+solution_y-radius_search,realization_y-previous_y+solution_y+radius_search+1):
                                for sample_x in range(realization_x-previous_x+solution_x-radius_search,realization_x-previous_x+solution_x+radius_search+1):
                                    if sample_y >=0 and sample_y < TI_Height and sample_x >= 0 and sample_x < TI_Width:
                                        distance_new = 0.0
                                        #this is the distance to the new data
                                        for pattern_index in range(len(pattern_new_value)):
                                            point_y = pattern_new_y[pattern_index] + sample_y
                                            point_x = pattern_new_x[pattern_index] + sample_x
                                            if point_y >=0 and point_y < TI_Height and point_x >= 0 and point_x < TI_Width:
                                                training_pattern_value = TI[point_y][point_x]
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
                                            point_y = pattern_old_y[pattern_index] + sample_y
                                            point_x = pattern_old_x[pattern_index] + sample_x
                                            if point_y >=0 and point_y < TI_Height and point_x >= 0 and point_x < TI_Width:
                                                training_pattern_value = TI[point_y][point_x]
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
                                            sample_value = TI[sample_y][sample_x]
                                            distance_min_x = sample_x
                                            distance_min_y = sample_y

                        # this point does not have the clost simulated point         
                        else:
                            #an exhaustive search in TI
                            #print("exhaustive search")
                            distance = 10.0
                            distance_min = 10.0
                            sample_Amount = 0
                            TI_judge = np.zeros((TI_Height,TI_Width)) # 0 non checked 1 checked
                            while distance_min > Threshold and sample_Amount < Fraction :                    
                                sample_y = random.randint(0,TI_Height-1)
                                sample_x = random.randint(0,TI_Width-1)
                                if TI_judge[sample_y][sample_x]==1:
                                    continue
                                TI_judge[sample_y][sample_x]=1
                                sample_Amount += 1
                                # this is the distance to the new data
                                distance_new = 0.0
                                for pattern_index in range(len(pattern_new_value)):
                                    point_y = pattern_new_y[pattern_index] + sample_y
                                    point_x = pattern_new_x[pattern_index] + sample_x
                                    if point_y >=0 and point_y < TI_Height and point_x >= 0 and point_x < TI_Width:
                                        training_pattern_value = TI[point_y][point_x]
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
                                    point_y = pattern_old_y[pattern_index] + sample_y
                                    point_x = pattern_old_x[pattern_index] + sample_x
                                    if point_y >=0 and point_y < TI_Height and point_x >= 0 and point_x < TI_Width:
                                        training_pattern_value = TI[point_y][point_x]
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
                                    sample_value = TI[sample_y][sample_x]
                                    distance_min_x = sample_x
                                    distance_min_y = sample_y

                        #assign the training point to the conditioning point
                        solution_array_x[realization_y][realization_x] = distance_min_x
                        solution_array_y[realization_y][realization_x] = distance_min_y

                        #print("the center point of the simulated patch is x:"+str(realization_x) + " y: "+str(realization_y) )

                        # paste a patch in the simulation domain
                        for point_y in range(-radius_simulationPatch,+radius_simulationPatch+1):
                            for point_x in range(-radius_simulationPatch,+radius_simulationPatch+1):
                                patch_realization_y = realization_y + point_y
                                patch_realization_x = realization_x + point_x
                                patch_TI_y = distance_min_y + point_y
                                patch_TI_x = distance_min_x + point_x
                                if  patch_realization_y >=0 and patch_realization_y < Realization_Height and patch_realization_x >= 0 and patch_realization_x < Realization_Width:
                                    if new_realization[patch_realization_y][patch_realization_x] == -1:
                                        if  patch_TI_y >=0 and patch_TI_y < TI_Height and patch_TI_x >= 0 and patch_TI_x < TI_Width:
                                            sample_value = TI[patch_TI_y][patch_TI_x]
                                            new_realization_temporary[patch_realization_y][patch_realization_x] = sample_value
                                            new_realization_operation_temporary[patch_realization_y][patch_realization_x] = 2

                                            #print("the simulated point is x:"+str(patch_realization_x) + " y: "+str(patch_realization_y) )

                                            # determine the points that need to be simulated in the next iteration
                                            # bottom side
                                            if point_y == -radius_simulationPatch and sample_value != old_realization[patch_realization_y][patch_realization_x]:
                                                sample_y = patch_realization_y - 1
                                                if sample_y >= 0:
                                                    loop_status = True
                                                    for sample_x in range(max(patch_realization_x-1,0),min(patch_realization_x+2,Realization_Width)):
                                                        if new_realization_operation[sample_y][sample_x] == 0 and new_realization_operation_temporary[sample_y][sample_x]==0:
                                                                new_realization_operation_temporary[sample_y][sample_x] = 1
                                            # top side
                                            elif point_y == radius_simulationPatch and sample_value != old_realization[patch_realization_y][patch_realization_x]:
                                                sample_y = patch_realization_y + 1
                                                if sample_y < Realization_Height:
                                                    loop_status = True
                                                    for sample_x in range(max(patch_realization_x-1,0),min(patch_realization_x+2,Realization_Width)):
                                                        if new_realization_operation[sample_y][sample_x] == 0 and new_realization_operation_temporary[sample_y][sample_x]==0:
                                                                new_realization_operation_temporary[sample_y][sample_x] = 1 

                                            # left side
                                            if point_x == -radius_simulationPatch and sample_value != old_realization[patch_realization_y][patch_realization_x]:
                                                sample_x = patch_realization_x - 1
                                                if sample_x >= 0:
                                                    loop_status = True
                                                    for sample_y in range(max(patch_realization_y-1,0),min(patch_realization_y+2,Realization_Height)):
                                                        if new_realization_operation[sample_y][sample_x] == 0 and new_realization_operation_temporary[sample_y][sample_x]==0:
                                                                new_realization_operation_temporary[sample_y][sample_x] = 1
                                            # right side
                                            elif point_x == radius_simulationPatch and sample_value != old_realization[patch_realization_y][patch_realization_x]:
                                                sample_x = patch_realization_x + 1
                                                if sample_x < Realization_Width:
                                                    loop_status = True
                                                    for sample_y in range(max(patch_realization_y-1,0),min(patch_realization_y+2,Realization_Height)):
                                                        if new_realization_operation[sample_y][sample_x] == 0 and new_realization_operation_temporary[sample_y][sample_x]==0:
                                                                new_realization_operation_temporary[sample_y][sample_x] = 1 

                    # these points are simulated by the previous block
                    elif(new_realization_operation[realization_y][realization_x]==1 and new_realization_temporary[realization_y][realization_x]!=-1):
                        sample_value = 0;
                    # these points have already been processed
                    elif(new_realization_operation[realization_y][realization_x]==2): 
                        sample_value = new_realization[realization_y][realization_x]
                        new_realization_temporary[realization_y][realization_x] = sample_value
                        new_realization_operation_temporary[realization_y][realization_x] = 2 
                    elif (new_realization_operation[realization_y][realization_x]==0 and new_realization_temporary[realization_y][realization_x]!=-1):
                        sample_value = 0;
                    else:
                        sample_value = new_realization[realization_y][realization_x]
                        new_realization_temporary[realization_y][realization_x] = sample_value
                        new_realization_operation_temporary[realization_y][realization_x] = 0 

            for realization_y in range(Realization_Height):
                for realization_x in range(Realization_Width):
                    sample_value = new_realization_temporary[realization_y][realization_x]
                    new_realization[realization_y][realization_x] = sample_value
                    sample_value = new_realization_operation_temporary[realization_y][realization_x]
                    new_realization_operation[realization_y][realization_x] = sample_value


        #     plt.imshow(new_realization,vmin=-1, vmax=3)
        #     plt.show()
            #plt.imshow(new_realization_operation,vmin=-1, vmax=3)
            #plt.show()


        # print((datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')))

        for realization_y in range(Realization_Height):
            for realization_x in range(Realization_Width):
                if new_realization[realization_y][realization_x] != -1:
                    conditioning_pattern_value = new_realization[realization_y][realization_x]
                    new_realization_temporary[realization_y][realization_x] = conditioning_pattern_value
                else:
                    conditioning_pattern_value = old_realization[realization_y][realization_x]
                    new_realization_temporary[realization_y][realization_x] = conditioning_pattern_value
        
        corrected_posterior.append(new_realization_temporary)
    
    corrected_posterior = np.asarray(corrected_posterior)
    #this is the resimulation area
    # print("this is the resimulation area: ")
    # plt.imshow(new_realization,vmin=-1, vmax=3)
    # plt.colorbar(fraction=0.02)
    # plt.show()   
    
    return corrected_posterior


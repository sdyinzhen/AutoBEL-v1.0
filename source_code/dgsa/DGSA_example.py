from Sensitivity_Analysis import *

## Name for model (global + spatial)
model_name = ['PC'+str(i) for i in (np.arange(n_component)+1)]
[model_name.append(name) for name in ['mean', 'sill', 'rangeX', 'rangeY', 'rangeZ']]

SA_dataframe = DGSA(np.vstack((model_PCA['pc_scores'][:n_component,:],model['global'].T)),
                    data_PCA['pc_scores'],
                    model_name)

## Plot Pareto Chart for DGSA
SA_fig = DGSA_plot(SA_dataframe, 25, title = 'SA(m|d)')
#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: Apri 03, 2019

from qc_reslts_plt import *
from giip_cal import *

def QC_results(dobs_file, m_post_file, m_pri_file, x_dim, y_dim, layernum, GIIP_pri_file, pri_m_samples_dir):
    
    ''' This is the function to QC the Auto-BEL poseterior results'''
    
    print("  ")
    print("1. QC posterior results") 
    print("1.1 Plot posterior models") 
    m_post = np.load(m_post_file)
    m_pri = np.load(m_pri_file)
    d_obs = np.loadtxt(fname=dobs_file,skiprows=1)[74:675:75, :].T
    m_sampls_plt(m_post, 'thc_coef', x_dim, y_dim, 1)
    m_ensampl_plt(m_post, m_pri, 1, x_dim, y_dim, 1, d_obs)
    
    print("  ")
    print("1.2 Calculate posterior prediction") 
    GIIP_post  = GIIP_cal(1, 0, m_post, pri_m_samples_dir+'Bulk_volume.GRDECL', False)
    np.save('output/prediction/GIIP_post', GIIP_post)
    GIIP_pri = np.load(GIIP_pri_file)
    giip_compare(GIIP_pri, GIIP_post)
    print("  ")
    print("Completed :-)!")  
    return 
    
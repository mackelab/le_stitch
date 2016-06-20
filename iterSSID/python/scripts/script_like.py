import numpy as np
import scipy as sp
import glob, os
#from scipy.io import savemat # store results for comparison with Matlab code   

os.chdir('../core')
from utility import get_subpop_stats, draw_sys
from SSID_Hankel_loss import test_run
os.chdir('../dev')

# load ancient code for drawing from LDS ...
os.chdir('../../../../pyRRHDLDS/core')
from ssm_scripts import sim_data
os.chdir('../../code_le_stitch/iterSSID/python/dev')


def run_test(p,n,Ts,k,l,sub_pops,max_iter_nl=1000, save_file = None):

    # we fixate the following simulation parameters for now:
    nr = n//2 if 2*((n-n//2)//2) == (n-n//2) else n//2 - 1
    eig_m_r, eig_M_r, eig_m_c, eig_M_c = 0.8, 0.99, 0.8, 0.99
    batch_size = p 
    a, b1, b2, e = 0.001, 0.9, 0.99, 1e-8
    reps = 1

    linearity = 'False'
    stable      = False

    test_run(p,n,Ts=Ts,k=k,l=l,batch_size=batch_size,sub_pops=sub_pops,
                 nr=nr, eig_m_r=eig_m_r, eig_M_r=eig_M_r, eig_m_c=eig_m_c, eig_M_c=eig_M_c,
                 a=a, b1=b1, b2=b2, e=e, max_iter_nl = max_iter_nl, verbose=False,
                 linearity=linearity,stable=stable,save_file=save_file,
                 get_subpop_stats=get_subpop_stats, draw_sys=draw_sys,sim_data=sim_data)


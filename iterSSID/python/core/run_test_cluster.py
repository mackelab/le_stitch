import os
import numpy as np

np.random.seed(0)

from script_like import run_test

p = 1000
k,l = 5,5

ns = (3,10,30)
Ts = (100, 1000, 10000, np.inf)
num_reps = 10

max_iter_nl = 500

for rep in range(num_reps):

    #"""
    # fully observed
    sub_pops = (np.arange(p), np.arange(p))
    for n in ns:
        save_file = 'sweep_n_T_p' + str(p) + 'n' + str(n) + 'k' + str(k) + 'l' + str(l)
        save_file = save_file + '_fullyObs_nr' + str(rep)
        #print('save_file')
        run_test(p,n,Ts,k,l,sub_pops,max_iter_nl=max_iter_nl, save_file = save_file)
    #"""

    #"""        
    # non-overlapping
    sub_pops = (np.arange(0,p//2), np.arange(p//2,p))
    for n in ns:
        save_file = 'sweep_n_T_p' + str(p) + 'n' + str(n) + 'k' + str(k) + 'l' + str(l)
        save_file = save_file + '_nonOverlap_nr' + str(rep)
        #print('save_file')
        run_test(p,n,Ts,k,l,sub_pops,max_iter_nl=max_iter_nl, save_file = save_file)
    #"""

    #"""        
    # leap-frogging
    assert p == 1000
    sub_pops = (np.arange(400), np.arange(200,600), np.arange(400,800), np.arange(600,p))
    #assert p == 50
    #sub_pops = (np.arange(20), np.arange(10,30), np.arange(20,40), np.arange(30,p))
    for n in ns:
        save_file = 'sweep_n_T_p' + str(p) + 'n' + str(n) + 'k' + str(k) + 'l' + str(l)
        save_file = save_file + '_LeapFrog_nr' + str(rep)
        #print('save_file')
        run_test(p,n,Ts,k,l,sub_pops,max_iter_nl=max_iter_nl, save_file = save_file)
    #"""
        

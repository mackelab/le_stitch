#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import glob, os, psutil, time

import sys, os
sys.path.append('/groups/turaga/home/speisera/_guest/python/core')

from utility import get_subpop_stats, gen_data
from SSID_Hankel_loss import run_bad, print_slim, plot_outputs_l2_gradient_test

#np.random.seed(0)

# define problem size
p, n, k, l, T = 100, 10, 2, 2, 100

# settings for fitting algorithm
batch_size, max_zip_size, max_iter = 1, 100, 1000
a, b1, b2, e = 0.01, 0.9, 0.99, 1e-8
linearity, stable, sym_psd = 'False', False, False

# I/O matter
mmap, chunksize = True, np.min((p,2000))
data_path, save_file = '/groups/turaga/home/speisera/_guest/python/fits/', 'test'
verbose=True

# create subpopulations
sub_pops = (np.arange(0,p), np.arange(0,p))
obs_pops = np.array([0,1])
obs_time = np.array([T//2, T])

obs_idx, idx_grp, co_obs, _, _, _, Om, _, _ = \
    get_subpop_stats(sub_pops=sub_pops, p=p, verbose=False)

# draw system matrices 
print('\n (p,n,k+l,T) = ', (p,n,k+l,T), '\n')
nr = 0 # number of real eigenvalues
eig_m_r, eig_M_r, eig_m_c, eig_M_c = 0.8, 0.99, 0.8, 0.99
pars_true, x, y, Qs, idx_a, idx_b = gen_data(p,n,k,l,T, nr,
                                             eig_m_r, eig_M_r, 
                                             eig_m_c, eig_M_c,
                                             mmap, chunksize,
                                             data_path)

pars_init='default'  


# settings for fitting algorithm
batch_size, max_zip_size, max_iter = 1, 100, 100
a, b1, b2, e = 0.01, 0.9, 0.99, 1e-8
a_R = 100 * a
linearity, stable, sym_psd = 'False', False, False

t = time.time()
pars_init, pars_est, traces = run_bad(k=k,l=l,n=n,y=y, Qs=Qs,Om=Om,idx_a=idx_a, idx_b=idx_b,
                                      sub_pops=sub_pops,idx_grp=idx_grp,co_obs=co_obs,obs_idx=obs_idx,
                                      obs_pops=obs_pops,obs_time=obs_time,
                                      linearity=linearity,stable=stable,init=pars_init,
                                      alpha=a,alpha_R=a_R,b1=b1,b2=b2,e=e,max_iter=max_iter,batch_size=batch_size,
                                      verbose=verbose, sym_psd=sym_psd, max_zip_size=max_zip_size)

print('fitting time was ', time.time() - t, 's')
print('\n')
print(psutil.virtual_memory())
print(psutil.swap_memory())

print('\n final correlations est. vs. true covs \n')
print_slim(Qs,k,l,pars_est,idx_a,idx_b,traces,mmap,data_path)

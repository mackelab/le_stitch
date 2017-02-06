import numpy as np
import scipy as sp
import glob, os, psutil, time
from scipy import linalg as la

from ssidid.SSID_Hankel_loss import run_bad, plot_slim, print_slim, f_l2_Hankel_nl, f_l2_Hankel_comp_Q_Om
from ssidid.utility import get_subpop_stats, gen_data
from ssidid import ObservationScheme
from subtracking import Grouse, calc_subspace_proj_error
from ssidid import progprint_xrange

data_path = '/home/nonnenmacher/stitching/results/'
run = '_test'

# define problem size
lag_range = np.arange(10)
kl_ = np.max(lag_range)+1
p, n, T = 2100, 10, 1000-kl_ + kl_

nr = 0 # number of real eigenvalues
snr = (0.1, 0.1)
whiten = True
eig_m_r, eig_M_r, eig_m_c, eig_M_c = 0.90, 0.95, 0.90, 0.95


# I/O matter
mmap, chunksize = True, np.min((p,1000))
verbose=True

# create subpopulations

sub_pops = (np.arange(p//2), np.arange(p//2, p))

reps = 100
obs_pops = np.concatenate([ np.arange(len(sub_pops)) for r in range(reps) ])
obs_time = np.linspace(0,T, len(obs_pops)+1)[1:].astype(int)
obs_scheme = ObservationScheme(p=p, T=T, 
                                sub_pops=sub_pops, 
                                obs_pops=obs_pops, 
                                obs_time=obs_time)
obs_scheme.comp_subpop_stats()

missing_at_random, frac_obs = False, 0.5
if missing_at_random:
    n_obs = np.ceil(p * frac_obs)
    mask = np.zeros((T,p))
    for t in range(T):
        for i in range(len(obs_time)):
            if t < obs_time[i]:
                mask[t, np.random.choice(p, n_obs, replace=False)] = 1
                break                       
    obs_scheme.mask = mask
    del mask
else:
    if p*T < 1e8:
        obs_scheme.gen_mask_from_scheme()
        obs_scheme.use_mask = False

print('(p,n,k+l,T) = ', (p,n,len(lag_range),T), '\n')
    
pars_est = 'default'


disp('ensuring zero-mean data for given observation scheme')
pars_true, x, y, _, _ = gen_data(p,n,lag_range,T, nr,
                                 eig_m_r, eig_M_r, 
                                 eig_m_c, eig_M_c,
                                 mmap, chunksize,
                                 data_path,
                                 snr=snr, whiten=whiten)    

if len(obs_scheme.sub_pops) > 1:
    disp('ensuring zero-mean data for given observation scheme')
    if mmap: 
        for i in progprint_xrange(p//chunksize, perline=10):
            y = np.memmap(data_path+'y', dtype=np.float, mode='r+', shape=(T,p))
            y[:, i*chunksize:(i+1)*chunksize] = y[:, i*chunksize:(i+1)*chunksize] - y[:, i*chunksize:(i+1)*chunksize].mean(axis=0)
            del y
        if (p//chunksize)*chunksize < p:
            y = np.memmap(data_path+'y', dtype=np.float, mode='r+', shape=(T,p))
            y[:, (p//chunksize)*chunksize:] = y[:, (p//chunksize)*chunksize:] - y[:, (p//chunksize)*chunksize:].mean(axis=0)
            del y        
        y = np.memmap(data_path+'y', dtype=np.float, mode='r', shape=(T,p))
    else:
        y -= y.mean(axis=0)

idx_a = np.sort(np.random.choice(p, 1000, replace=False)) if p > 1000 else np.arange(p)
idx_b = idx_a.copy()

W = obs_scheme.comp_coocurrence_weights(lag_range, sso=True, idx_a=idx_a, idx_b=idx_b)
Qs, Om = f_l2_Hankel_comp_Q_Om(n=n,y=y,lag_range=lag_range,obs_scheme=obs_scheme,
                      idx_a=idx_a,idx_b=idx_b,W=W,
                      mmap=mmap,data_path=data_path,ts=None,ms=None)


parametrization='nl'
sso = True

# settings for quick initial SGD fitting phase for our model
batch_size, max_zip_size, max_iter = 1, np.inf, 50
a, b1, b2, e = 0.005, 0.98, 0.99, 1e-8
a_R = 1 * a

proj_errors = np.zeros((max_iter,n+1))
def principal_angle(A, B):
    "A and B must be column-orthogonal."    
    A = np.atleast_2d(A).T if (A.ndim<2) else A
    B = np.atleast_2d(B).T if (B.ndim<2) else B
    A = la.orth(A)
    B = la.orth(B)
    svd = la.svd(A.T.dot(B))
    return np.arccos(np.minimum(svd[1], 1.0)) / (np.pi/2)
    
def pars_track(pars,t): 
    C = pars[0]
    proj_errors[t] = np.hstack((0, principal_angle(pars_true['C'], C)))
            
_, pars_est, traces, Qs, Om, W, t = run_bad(lag_range=lag_range,n=n,y=y, idx_a=idx_a, idx_b=idx_b,
                                      obs_scheme=obs_scheme,init=pars_est,
                                      parametrization=parametrization, sso=sso,
                                      Qs=Qs, Om=Om, W=W,
                                      alpha=a,b1=b1,b2=b2,e=e,max_iter=max_iter,
                                      batch_size=batch_size,verbose=verbose, max_epoch_size=max_zip_size,
                                      pars_track=pars_track)

print_slim(Qs,Om,lag_range,pars_est,idx_a,idx_b,traces,False,data_path)

save_dict = {'p' : p,
             'n' : n,
             'T' : T,
             'snr' : snr,
             'obs_scheme' : obs_scheme,
             'lag_range' : lag_range,
             'x' : x,
             'mmap' : mmap,
             'y' : data_path if mmap else y,
             'pars_true' : pars_true,
             'pars_est' : pars_est,
             'idx_a' : idx_a,
             'idx_b' : idx_b,
             'W' : W,
             'Qs' : Qs,
             'Om' : Om
            }
file_name = 'p' + str(p) + 'n' + str(n) + 'T' + str(T) + 'snr' + str(np.int(np.mean(snr)//1)) + '_run' + str(run)
np.savez(data_path + file_name, save_dict)
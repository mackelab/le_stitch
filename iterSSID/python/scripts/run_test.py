import numpy as np
import scipy as sp
import glob, os
from scipy.io import savemat # store results for comparison with Matlab code   

os.chdir('../core')
from utility import get_subpop_stats, draw_sys
from SSID_Hankel_loss import test_run
os.chdir('../dev')

# load ancient code for drawing from LDS ...
os.chdir('../../../../pyRRHDLDS/core')
from ssm_scripts import sim_data
os.chdir('../../code_le_stitch/iterSSID/python/dev')


# define simulation setup

# system size, number of data points, smallest and largest time constants of data
p,n,nr = 500, 20, 10
T = p
eig_m_r, eig_M_r, eig_m_c, eig_M_c = 0.8, 0.99, 0.8, 0.99

k,l = 5,5

batch_size = p # batch_size = 1 (size-1 mini-batches), p (column mini-batches), None (full gradients)

a, b1, b2, e = 0.001, 0.9, 0.99, 1e-8
max_iter_nl =  100
reps = 1

linearity = 'False'
stable      = False

# create subpopulations
sub_pops = (np.arange(0,p//2+1), np.arange(p//2-1,p))

#sub_pops = (np.arange(0,p), np.arange(0,p))
#sub_pops = (np.arange(0,200), np.arange(100,300), np.arange(200,400), np.arange(300,p))
#sub_pops = (np.arange(0,2*p//10), np.arange(p//10,3*p//10), np.arange(2*p//10,4*p//10), np.arange(3*p//10,5*p//10),
#            np.arange(4*p//10,6*p//10), np.arange(5*p//10,7*p//10), np.arange(6*p//10,8*p//10), 
#            np.arange(7*p//10,9*p//10), np.arange(8*p//10,p))
#sub_pops = (np.arange(10,p), np.arange(0,p-10))

pars_true, pars_init, pars_est, traces, x, y, Qs, Qs_full, options = \
test_run(p,n,T=T,k=k,l=l,batch_size=batch_size,sub_pops=sub_pops,
             nr=nr, eig_m_r=eig_m_r, eig_M_r=eig_M_r, eig_m_c=eig_m_c, eig_M_c=eig_M_c,
             a=a, b1=b1, b2=b2, e=e, max_iter_nl = max_iter_nl, verbose=False,
             linearity=linearity,stable=stable,
             get_subpop_stats=get_subpop_stats, draw_sys=draw_sys,sim_data=sim_data)

# save results to .mat and .npz files

os.chdir('../fits/')

save_file = 'test_p' + str(p) + 'n' + str(n) + 'r' + str(len(sub_pops))

save_file_m = {'linearity': linearity,
               'A_true': pars_true['A'],
               'Pi_true' : pars_true['Pi'], 
               'C_true' : pars_true['C'],
               'A_0': pars_init['A'],
               'Pi_0': pars_init['Pi'],
               'C_0': pars_init['C'],
               'A_est': pars_est['A'],
               'Pi_est' :  pars_est['Pi'], 
               'C_est' :  pars_est['C'],
               'fs' : traces[0],
               'corrs' : traces[1],
               'p': p, 'n': n,
               'k': k, 'l': l, 
               'T': T,
               'a': a, 'b1': b1, 'b2': b2, 'e': e, 'max_iter_nl': max_iter_nl,
               'r': len(sub_pops),
               'sub_pops': sub_pops,
               'batch_size': batch_size,
               'y': y,
               'x': x,
               'Qs': Qs, 
               'Qs_full': Qs_full}

savemat(save_file,save_file_m) # does the actual saving

pars_true_vec = np.hstack((pars_true['A'].reshape(n*n,),
                    pars_true['Pi'].reshape(n*n,),
                    pars_true['C'].reshape(p*n,)))
pars_init_vec = np.hstack((pars_init['A'].reshape(n*n,),
                    pars_init['Pi'].reshape(n*n,),
                    pars_init['C'].reshape(p*n,)))
pars_est_vec  = np.hstack((pars_est['A'].reshape(n*n,),
                    pars_est['Pi'].reshape(n*n,),
                    pars_est['C'].reshape(p*n,)))

np.savez(save_file, 
         y=y,
         x=x,
         T=T,
         Qs_full=Qs_full,
         Qs=Qs,
         linearity=linearity,
         pars_init=pars_init,
         pars_est =pars_est,
         pars_true=pars_true,         
         pars_0_vec=pars_init_vec,
         pars_true_vec=pars_true_vec, 
         pars_est_vec=pars_est_vec,
         traces=traces,
         p=p, n=n, k=k, l=l, 
         batch_size=batch_size,
         a=a, b1=b1, b2=b2, e=e, max_iter_nl=max_iter_nl, 
         sub_pops = sub_pops,
         r=len(sub_pops))  

    

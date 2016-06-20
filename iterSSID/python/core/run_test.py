import os
os.chdir('../scripts')
from script_like import run_test
import numpy as np

p = 100
n = 10
k, l = 5,5
Ts = (p,)
sub_pops = (np.arange(p//2), np.arange(p//2,p))
save_file = 'test'
max_iter_nl = 1000
run_test(p=p,n=n,Ts=T,k=k,l=l,sub_pops=sub_pops, save_file = save_file, max_iter_nl = max_iter_nl )

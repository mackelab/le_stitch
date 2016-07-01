import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import warnings
#import control
from scipy.linalg import solve_discrete_lyapunov as dlyap
from numpy.lib.stride_tricks import as_strided
import cvxopt
import SDLS
import os

def mat(X):
    return cvxopt.matrix(X, tc='d')

###########################################################################
# constructing Hankel matrices
###########################################################################

def yy_Hankel_cov_mat(C,A,Pi,k,l,Om=None,linear=True):
    "matrix with blocks cov(y_t+m, y_t) m = 1, ..., k+l-1 on the anti-diagonal"
    
    p,n = C.shape
    if linear:
        assert n == A.shape[1] and n == Pi.shape[0] and n == Pi.shape[1]
    else:
        assert n*n == A.shape[0] and k+l-1 <= A.shape[1]
        
    assert (Om is None) or (Om.shape == (p,p))
    if not Om is None:
        Om_idx = np.asarray(Om, dtype=float)
        Om_idx[~Om] = np.nan

    H = np.empty((k*p, l*p))
    
    for kl_ in range(k+l-1):        
        AmPi = np.linalg.matrix_power(A,kl_+1).dot(Pi) if linear else A[:,kl_].reshape(n,n)
        lamK = C.dot(AmPi).dot(C.T)
        
        lamK = lamK if Om is None else lamK * Om_idx
        if kl_ < k-0.5:     
            for l_ in range(0, min(kl_ + 1,l)):
                offset0, offset1 = (kl_-l_)*p, l_*p
                H[offset0:offset0+p, offset1:offset1+p] = lamK
                
        else:
            for l_ in range(0, min(k+l - kl_ -1, l, k)):
                offset0, offset1 = (k - l_ - 1)*p, ( l_ + kl_ + 1 - k)*p
                H[offset0:offset0+p,offset1:offset1+p] = lamK
            
    return H

def yy_Hankel_cov_mat_Qs(Qs,idx,k,l,n,Om=None):
    "matrix with blocks cov(y_t+m, y_t) m = 1, ..., k+l-1 on the anti-diagonal"
    
    p = Qs[1].shape[0] if Qs[0] is None else Qs[0].shape[0]     
    pe = idx.size
        
    assert (Om is None) or (Om.shape == (p,p))
    if not Om is None:
        Om_idx = np.asarray(Om, dtype=float)[np.ix_(idx,idx)]
        Om_idx[~Om[np.ix_(idx,idx)]] = np.nan
    
    H = np.empty((k*pe, l*pe))
    
    for kl_ in range(k+l-1):        
        lamK = Qs[kl_+1][np.ix_(idx,idx)]
        
        lamK = lamK if Om is None else lamK * Om_idx
        if kl_ < k-0.5:     
            for l_ in range(0, min(kl_ + 1,l)):
                offset0, offset1 = (kl_-l_)*pe, l_*pe
                H[offset0:offset0+pe, offset1:offset1+pe] = lamK
                
        else:
            for l_ in range(0, min(k+l - kl_ -1, l, k)):
                offset0, offset1 = (k - l_ - 1)*pe, ( l_ + kl_ + 1 - k)*pe
                H[offset0:offset0+pe,offset1:offset1+pe] = lamK
            
    return H    




###########################################################################
# 'vanilla' optimisation: following gradients w.r.t. C, A, B = sqrt(Pi)
###########################################################################

def l2_setup(k,l,n,Qs,Om,idx_grp,obs_idx):
    "returns error function and gradient for use with gradient descent solvers"

    def co_observed(x, i):
        for idx in obs_idx:
            if x in idx and i in idx:
                return True
        return False        

    num_idx_grps = len(idx_grp)
    co_obs, mat_obs = [], np.zeros((num_idx_grps,num_idx_grps))
    for i in range(num_idx_grps):    
        for j in range(num_idx_grps):
            if co_observed(i,j):
                mat_obs[i,j] = True             
        co_obs.append([idx_grp[x] for x in np.arange(len(idx_grp)) \
            if co_observed(x,i)])
        co_obs[i] = np.sort(np.hstack(co_obs[i]))

    def g(parsv):
        return g_l2_Hankel(parsv,k,l,n,Qs,idx_grp,co_obs)

    def f(parsv):                        
        return f_l2_Hankel(parsv,k,l,n,Qs,Om)*np.sum(Om)*(k*l)

    return f,g

def l2_vec_to_system_mats(parsv,p,n):
    "translates vectorised parameters to matrix forms"

    A = parsv[:n*n].reshape(n, n)
    B = parsv[(n*n):(2*n*n)].reshape(n, n)
    C = parsv[-p*n:].reshape(p, n)

    return A,B,C

def l2_system_mats_to_vec(A,B,C):
    "translates matrices to vector for use with gradient descent solvers"

    p,n = C.shape
    parsv = np.hstack((A.reshape(n*n,),B.reshape(n*n,),C.reshape(p*n,)))

    return parsv

def f_l2_Hankel(parsv,k,l,n,Qs,Om):
    "returns overall l2 Hankel reconstruction error"

    p = Qs[1].shape[0] if Qs[0] is None else Qs[0].shape[0]     
    A,B,C = l2_vec_to_system_mats(parsv,p,n)
    Pi = B.dot(B.T)

    err = 0.
    for m in range(1,k+l):
        APi = np.linalg.matrix_power(A, m).dot(Pi)  
        err += f_l2_block(C,APi,Qs[m],Om)
            
    return err/(k*l)
    
def f_l2_block(C,AmPi,Q,Om):
    "Hankel reconstruction error on an individual Hankel block"

    v = (C.dot(AmPi.dot(C.T)))[Om] - Q[Om] # this is not efficient for spars Om

    return v.dot(v)/(2*np.sum(Om))


def g_l2_Hankel(parsv,k,l,n,Qs,idx_grp,co_obs):
    "returns overall l2 Hankel reconstruction gradient w.r.t. A, B, C"

    p = Qs[1].shape[0] if Qs[0] is None else Qs[0].shape[0]     
    A,B,C = l2_vec_to_system_mats(parsv,p,n)
    Pi = B.dot(B.T)

    Aexpm = np.zeros((n,n,k+l))
    Aexpm[:,:,0]= np.eye(n)
    for m in range(1,k+l):
        Aexpm[:,:,m] = A.dot(Aexpm[:,:,m-1])

    grad_A, grad_B, grad_C = np.zeros((n,n)), np.zeros((n,n)), np.zeros((p,n))
    for m in range(1,k+l):

        AmPi = Aexpm[:,:,m].dot(Pi)

        # the expensive part: handling p x p ovserved-space matrices 
        CAPiC_L, CTC = C.dot(AmPi).dot(C.T) - Qs[m], np.zeros((n,n))
        for i in range(len(idx_grp)):
            a,b = idx_grp[i],co_obs[i]
            Ci = CAPiC_L[np.ix_(a,b)].dot(C[b,:])
            CiT =  CAPiC_L[np.ix_(b,a)].T.dot(C[b,:])
            grad_C[idx_grp[i],:] += g_C_l2_idxgrp(Ci,CiT,AmPi)
            CTC += C[a,:].T.dot(Ci)

        grad_A += g_A_l2_block(CTC,Aexpm,m,Pi)
        grad_B += g_B_l2_block(CTC,Aexpm[:,:,m],B)
        

    return l2_system_mats_to_vec(grad_A,grad_B,grad_C)

def g_A_l2_block(CTC,Aexpm,m,Pi):
    "returns l2 Hankel reconstr. gradient w.r.t. A for a single Hankel block"
    
    CTCPi = CTC.dot(Pi)
    grad = np.zeros(Aexpm.shape[:2])
    for q in range(m):
        grad += Aexpm[:,:,q].T.dot(CTCPi.dot(Aexpm[:,:,m-1-q].T))

    return grad

def g_B_l2_block(CTC,Am,B):
    "returns l2 Hankel reconstr. gradient w.r.t. B for a single Hankel block"
            
    return (CTC.T.dot(Am) + Am.T.dot(CTC)).dot(B)

def g_C_l2_idxgrp(Ci,CiT,AmPi):
    "returns l2 Hankel reconstr. gradient w.r.t. A for a single fate group"

    return Ci.dot(AmPi.T) + CiT.dot(AmPi)

############################
# SGD, subsampling in space
############################

def adam_zip_stable(f,g,s,tau,theta_0,a,a_A,b1,b2,e,max_iter,
                converged,Om,idx_grp,co_obs,batch_size=None):
    
    N = theta_0.size
    p = Om.shape[0]
    n = np.int(np.round( np.sqrt( p**2/16 + N/2 ) - p/4 ))
    NnA = p*n + n*n
    
    if batch_size is None:
        print('doing full gradients - switching to plain gradient descent')
        b1, b2, e, v_0 = 0, 1.0, 0, np.ones(NnA)
    elif batch_size == 1:
        print('using size-1 mini-batches')
        v_0 = np.zeros(NnA)        
    elif batch_size == p:
        print('using size-p mini-batches (coviarance columms)')
        v_0 = np.zeros(NnA)
    else: 
        raise Exception('cannot handle selected batch size')

    
    # setting up Adam
    t_iter, t, t_zip = 0, 0, 0
    m, v = np.zeros(NnA), v_0.copy()
    theta, theta_old = theta_0.copy(), np.inf * np.ones(NnA)

    # setting up the stochastic batch selection:
    batch_draw, g_sis  = l2_sis_draw(p, batch_size, idx_grp, co_obs, None, Om)

    # trace function values
    fun = np.empty(max_iter)    
    sig = np.empty(max_iter)    


    
    while not converged(theta_old, theta, e, t_iter):

        theta_old = theta.copy()

        t_iter += 1
        idx_use, idx_co = batch_draw()
        zip_size = get_zip_size(batch_size, p, idx_use)
        for idx_zip in range(zip_size):
            t += 1

            # get data point(s) and corresponding gradients:                    
            grad = g_i(theta,idx_use,idx_co, idx_zip)
            grad_nA = grad[n*n:]
            m = (b1 * m + (1-b1)* grad_nA)     
            v = (b2 * v + (1-b2)*(grad_nA**2)) 
            if b1 != 1.:                    # delete those eventually 
                mh = m / (1-b1**t)
            else:
                mh = m
            if b2 != 1.:
                vh = v / (1-b2**t)
            else:
                vh = v
            mh /= (np.sqrt(vh) + e)

            theta[n*n:2*n*n] -= a_A * mh[:n*n] # updating Pi
            theta[-p*n:]     -= a   * mh[n*n:] # updating C


        # shift-cutting for A
        A, a_tmp = (theta[:n*n] - a_A * grad[:n*n]).reshape(n,n), a_A
        c = 0
        while not s(A) and c < 10000:
            c += 1
            a_tmp = tau * a_tmp
            A = (theta[:n*n] - a_tmp * grad[:n*n]).reshape(n,n)
        if c >= 10000:
            print(('Warning: crossed maximum number of stability-ensuring ',
                   'step-size reductions. Dynamics matrix might be unstable!'))
            print('maximum singular value:', np.max(np.linalg.svd(A)[1]))
        theta[:n*n] = A.reshape(n*n,).copy()


        if t_iter <= max_iter:          # outcomment this eventually - really expensive!
            fun[t_iter-1] = f(theta)
            sig[t_iter-1] = np.max(np.max(np.linalg.svd(A)[1]))
            
        if np.mod(t_iter,max_iter//10) == 2:
            print('f = ', fun[t_iter-1])
            
    print('total iterations: ', t)
        
    return theta, (fun,sig)    

def g_l2_Hankel_sis(parsv,k,l,n,Qs,idx_grp,co_obs):
    "returns l2 Hankel reconstr. stochastic gradient w.r.t. A, B, C"

    # sis: subsampled/sparse in space
    
    p = Qs[1].shape[0] if Qs[0] is None else Qs[0].shape[0]     
    A,B,C = l2_vec_to_system_mats(parsv,p,n)
    Pi = B.dot(B.T)

    Aexpm = np.zeros((n,n,k+l))
    Aexpm[:,:,0]= np.eye(n)
    for m in range(1,k+l):
        Aexpm[:,:,m] = A.dot(Aexpm[:,:,m-1])

    grad_A, grad_B, grad_C = np.zeros((n,n)), np.zeros((n,n)), np.zeros((p,n))
    for m in range(1,k+l):

        AmPi = Aexpm[:,:,m].dot(Pi)            

        CTC = np.zeros((n,n))
        for i in range(len(idx_grp)):
            a,b = idx_grp[i],co_obs[i]
            C_a, C_b  = C[a,:], C[b,:]

            # multi-neuron case
            ix_ab = np.ix_(a,b)
            Ci  = C_a.dot(AmPi.dot(  C_b.T.dot(C_b))) - Qs[m][ix_ab].dot(C_b)
            CiT = C_a.dot(AmPi.T.dot(C_b.T.dot(C_b))) - Qs[m].T[ix_ab].dot(C_b)

            # single-neuron-pair case:
            #Ci  = C_a.dot(AmPi.dot(  C_b.T.dot(C_b))) - Qs[m][a,b].dot(C_b)
            #CiT = C_a.dot(AmPi.T.dot(C_b.T.dot(C_b))) - Qs[m].T[a,b].dot(C_b)

            grad_C[a,:] += g_C_l2_idxgrp(Ci,CiT,AmPi)
            CTC += C_a.T.dot(Ci)
            
        grad_A += g_A_l2_block(CTC,Aexpm,m,Pi)
        grad_B += g_B_l2_block(CTC,Aexpm[:,:,m],B)
        

    return l2_system_mats_to_vec(grad_A,grad_B,grad_C)

def l2_sis_setup(k,l,n,Qs,Om,idx_grp,obs_idx):
    "returns error function and gradient for use with gradient descent solvers"

    def co_observed(x, i):
        for idx in obs_idx:
            if x in idx and i in idx:
                return True
        return False        

    num_idx_grps = len(idx_grp)
    co_obs, mat_obs = [], np.zeros((num_idx_grps,num_idx_grps))
    for i in range(num_idx_grps):    
        for j in range(num_idx_grps):
            if co_observed(i,j):
                mat_obs[i,j] = True             
        co_obs.append([idx_grp[x] for x in np.arange(len(idx_grp)) \
            if co_observed(x,i)])
        co_obs[i] = np.sort(np.hstack(co_obs[i]))
    (is_, js_) = np.where(Om)
    def g(parsv):
        return g_l2_Hankel_sis(parsv,k,l,n,Qs,is_,js_)

    def f(parsv):                        
        return f_l2_Hankel(parsv,k,l,n,Qs,Om)*np.sum(Om)*(k*l)

    return f,g

def l2_sis_draw(p, batch_size, idx_grp, co_obs, g_C, Om=None):
    "returns sequence of indices for sets of neuron pairs for SGD"

    if batch_size is None:
        def batch_draw():
            return idx_grp, co_obs    
        def g_sis(C, X, use, co, i):
            return g_C(C, X, idx_use,idx_co)
    elif batch_size == p:
        def batch_draw():
            idx_co = np.empty(p, dtype=np.int32)
            start = 0
            for idx in range(len(idx_grp)):
                end = start + idx_grp[idx].size
                idx_co[start:end] = idx
                start = end
            idx_use = np.hstack(idx_grp)
            idx_perm = np.random.permutation(np.arange(p))
            return idx_use[idx_perm], idx_co[idx_perm]
        def g_sis(C, X, use, co, i):
            a,b = (co_obs[co[i]],), (np.array((use[i],)),)
            return g_C(C, X, a, b)
    elif batch_size == 1:
        is_,js_ = np.where(Om)
        def batch_draw():
            idx = np.random.permutation(len(is_))
            return is_[idx], js_[idx]       
        def g_sis(C, X, use, co, i):
            return g_C(C, X, (np.array((use[i],)),),(np.array((co[i],)),))

    return batch_draw, g_sis

    

###########################################################################
# block coordinate descent: following gradients w.r.t. C, cov(x_{t+m}, x_t)
###########################################################################

def iter_X_m(CdQCdT_obs, C, Cd, p, n, idx_grp, not_co_obs, X_m):
    "'fast', because only recomputing the non-observed parts in each iteration"

    CdQCdT_stitch = np.zeros(X_m.shape)
    for i in range(len(idx_grp)):
        a, b = idx_grp[i], not_co_obs[i]
        CdQCdT_stitch += (Cd[:,a].dot(C[a,:])).dot(X_m).dot(Cd[:,b].dot(C[b,:]).T)
    return CdQCdT_obs + CdQCdT_stitch

def run_bad(k,l,n,Qs,
            Om,sub_pops,idx_grp,co_obs,obs_idx,
            linearity='False',stable=False,init='SSID',
            a=0.001, b1=0.9, b2=0.99, e=1e-8, 
            max_iter=100, max_zip_size=np.inf, batch_size=1,
            verbose=False, Qs_full=None, sym_psd=True):

    if not Om is None:
        p = Om.shape[0]
    else:
        p = Qs_full[1].shape[0] if Qs_full[0] is None else Qs_full[0].shape[0] 

    if isinstance(init, dict):
        assert 'C' in init.keys()

        pars_init = init.copy()

    elif init =='SSID':

        print('getting initial parameter values (SSID on largest subpopulation)')
        sub_pop_sizes = [ len(sub_pops[i]) for i in range(len(sub_pops))]
        idx = sub_pops[np.argmax(sub_pop_sizes)]
        H_kl = yy_Hankel_cov_mat_Qs(Qs=Qs,idx=idx,k=k,l=l,n=n,Om=Om)
        pars_ssid = ssidSVD(H_kl, Qs[0][np.ix_(idx,idx)], n, pi_method='proper')
        U,S,_ = np.linalg.svd(pars_ssid['Pi'])
        M = np.diag(1/np.sqrt(S)).dot(U.T)    
        pars_init = {'A'  : M.dot(pars_ssid['A']).dot(np.linalg.inv(M)),
             'Pi' : M.dot(pars_ssid['Pi']).dot(M.T),
             'B'  : np.eye(n), 
             'C'  : np.random.normal(size=(p,n))} #pars_ssid['C'].dot(np.linalg.inv(M))}   

    elif init =='default':
        pars_init = {'A'  : np.diag(np.linspace(0.89, 0.91, n)),
             'Pi' : np.eye(n),
             'B'  : np.eye(n), 
             'C'  : np.random.normal(size=(p,n))} #pars_ssid['C'].dot(np.linalg.inv(M))}   


    f_i, g_C, g_A, g_Pi,track_corrs = l2_bad_sis_setup(k=k,l=l,n=n,Qs=Qs,Qs_full=Qs_full,
                                           Om=Om,idx_grp=idx_grp,obs_idx=obs_idx,
                                           linearity=linearity,stable=stable,
                                           verbose=verbose,sym_psd=sym_psd)
    print('starting descent')    
    def converged(theta_old, theta, e, t):
        return True if t >= max_iter else False
    pars_est, traces = adam_zip_bad_stable(f=f_i,g_C=g_C,g_A=g_A,g_Pi=g_Pi,
                                       pars_0=pars_init,
                                       a=a,b1=b1,b2=b2,e=e,max_zip_size=max_zip_size,
                                       max_iter=max_iter,converged=converged,
                                       Om=Om,idx_grp=idx_grp,co_obs=co_obs,
                                       batch_size=batch_size,linearity=linearity,
                                       track_corrs=track_corrs)                 

    return pars_init, pars_est, traces


def l2_bad_sis_setup(k,l,n,Qs,Om,idx_grp,obs_idx,Qs_full=None, 
                        linearity='True', stable=False, sym_psd=True, W=None,
                        verbose=False):
    "returns error function and gradient for use with gradient descent solvers"

    if not Om is None:
        p = Om.shape[0]
    else:
        p = Qs_full[1].shape[0] if Qs_full[0] is None else Qs_full[0].shape[0] 

    def co_observed(x, i):
        for idx in obs_idx:
            if x in idx and i in idx:
                return True
        return False        

    num_idx_grps = len(idx_grp)
    co_obs = []
    for i in range(num_idx_grps):    
        co_obs.append([idx_grp[x] for x in np.arange(len(idx_grp)) \
            if co_observed(x,i)])
        co_obs[i] = np.sort(np.hstack(co_obs[i]))

    if linearity == 'True':
        linear_A = True
        linearise_X = False
        def g_C(C,X,idx_grp,co_obs):
            return g_C_l2_Hankel_bad_sis(C,X,k,l,Qs,idx_grp,co_obs,linear=True)

    elif linearity == 'first_order':
        linear_A = True
        linearise_X = True
        def g_C(C,X,idx_grp,co_obs):
            return g_C_l2_Hankel_bad_sis(C,X,k,l,Qs,idx_grp,co_obs,linear=False)

    elif linearity == 'False':
        linear_A = False
        linearise_X = False
        def g_C(C,X,idx_grp,co_obs):
            return g_C_l2_Hankel_bad_sis(C,X,k,l,Qs,idx_grp,co_obs,linear=False)


    def g_A(C,idx_grp,co_obs,A=None):
        return s_A_l2_Hankel_bad_sis(C,k,l,Qs,idx_grp,co_obs,
            linear=linear_A,linearise_X=linearise_X,stable=stable,A_old=A,
            verbose=verbose)

    def g_Pi(X,A,idx_grp,co_obs,Pi=None):
        return s_Pi_l2_Hankel_bad_sis(X,A,k,l,Qs,Pi,verbose=verbose, sym_psd=sym_psd)

    if linearity == 'True':
        def f(parsv):                        
            return f_l2_Hankel_Pi(parsv,k,l,n,Qs,Om)*np.sum(Om)*(k*l)
    else:
        def f(C,X):
            return f_l2_Hankel_nl(C,X,k,l,n,Qs,Om)*np.sum(Om)*(k*l)

    def track_corrs(C, A, Pi, X) :
         return track_correlations(Qs_full, p, n, Om, C, A, Pi, X, linearity=linearity)

    return f,g_C,g_A,g_Pi,track_corrs

def f_l2_Hankel_nl(C,X,k,l,n,Qs,Om):
    "returns overall l2 Hankel reconstruction error"

    p,n = C.shape

    err = 0.
    for m in range(1,k+l):
        err += f_l2_block(C,X[:,m-1].reshape(n,n),Qs[m],Om)
            
    return err/(k*l)

def f_l2_Hankel_Pi(parsv,k,l,n,Qs,Om):
    "returns overall l2 Hankel reconstruction error"

    if Om is None:
        p = Qs[1].shape[0] if Qs[0] is None else Qs[0].shape[0]
    else:
        p = Om.shape[0]

    A,Pi,C = l2_vec_to_system_mats(parsv,p,n)

    err = 0.
    for m in range(1,k+l):
        APi = np.linalg.matrix_power(A, m).dot(Pi)  
        err += f_l2_block(C,APi,Qs[m],Om)
            
    return err/(k*l)    
    


def adam_zip_bad_stable(f,g_C,g_A,g_Pi,track_corrs,pars_0,
                a,b1,b2,e,max_iter,
                converged,batch_size,max_zip_size,
                Om,idx_grp,co_obs,linearity='False'):

    # initialise pars
    C,A,X,Pi = set_adam_init(pars_0, g_A, g_Pi, idx_grp, co_obs, linearity)
    p, n = C.shape

    # setting up Adam
    b1,b2,e,v_0 = set_adam_pars(batch_size,p,n,b1,b2,e)
    t_iter, t, t_zip = 0, 0, 0
    ct_iter, corrs  = 0, np.zeros((2, 11))
    m, v = np.zeros((p,n)), v_0.copy()

    # setting up the stochastic batch selection:
    batch_draw, g_sis = l2_sis_draw(p, batch_size, idx_grp, co_obs, g_C, Om)

    # trace function values
    fun = np.empty(max_iter)    
    
    C_old = np.inf * np.ones((p,n))
    while not converged(C_old, C, e, t_iter):

        # updating C: full SGD pass over data
        C_old = C.copy()
        idx_use, idx_co = batch_draw()
        zip_size = get_zip_size(batch_size, p, idx_use, max_zip_size)
        for idx_zip in range(zip_size):
            t += 1

            # get data point(s) and corresponding gradients: 
            grad = g_sis(C,X,idx_use,idx_co, idx_zip)
            m = (b1 * m + (1-b1)* grad)     
            v = (b2 * v + (1-b2)*(grad**2)) 
            if b1 != 1.:                    # delete those eventually 
                mh = m / (1-b1**t)
            else:
                mh = m
            if b2 != 1.:
                vh = v / (1-b2**t)
            else:
                vh = v

            C -= a * mh/(np.sqrt(vh) + e)

        # updating A: solving linear least squares
        A,X,X1 =  g_A(C,idx_grp,co_obs,A)
        Pi = g_Pi(X,A,idx_grp,co_obs,Pi) if linearity=='True' else Pi
        linearise_latent_covs(A, X, Pi, X1, linearity) 

        if t_iter < max_iter:          # outcomment this eventually - really expensive!
            if linearity=='True':
                theta = np.zeros( (p + 2*n)*n )
                theta[:n*n] = A.reshape(-1,)
                theta[n*n:2*n*n] = Pi.reshape(-1,)
                theta[-p*n:] = C.reshape(-1,)
                fun[t_iter] = f(theta) 
            else:
                fun[t_iter] = f(C,X)

        if np.mod(t_iter,max_iter//10) == 0:
            print('finished %', 100*t_iter/max_iter)
            print('f = ', fun[t_iter])
            corrs[:,ct_iter] = track_corrs(C, A, Pi, X) 
            ct_iter += 1

        t_iter += 1

    corrs[:,ct_iter] = track_corrs(C, A, Pi, X)            

    print('total iterations: ', t)

    pars_out = {'C' : C, 'X': X }
    if linearity=='True': 
        pars_out['A'], pars_out['Pi'] = A, Pi
    elif linearity=='first_order':
        pars_out['A'], pars_out['Pi'] = A, np.zeros((n,n))     
    if linearity=='False': 
        pars_out['A'], pars_out['Pi'] = np.zeros((n,n)), np.zeros((n,n))

    return pars_out, (fun,corrs)    


def set_adam_pars(batch_size,p,n,b1,b2,e):
    if batch_size is None:
        print('doing full gradients - switching to plain gradient descent')
        b1, b2, e, v_0 = 0, 1.0, 0, np.ones((p,n))
    elif batch_size == 1:
        print('using size-1 mini-batches')
        v_0 = np.zeros((p,n))
    elif batch_size == p:
        print('using size-p mini-batches (coviarance columms)')
        v_0 = np.zeros((p,n))
    else: 
        raise Exception('cannot handle selected batch size')

    return b1,b2,e,v_0

def set_adam_init(pars_0, g_A, g_Pi, idx_grp, co_obs, linearity):
    C = pars_0['C'].copy()
    A,X,_ = g_A(C,idx_grp,co_obs,None)
    Pi =  None
    if linearity=='True':
        if 'A' in pars_0.keys() and not pars_0['A'] is None:
            A,X,Pi = pars_0['A'].copy(), None, pars_0['Pi'].copy()
        else: 
            print('getting initial Pi now')
            Pi = g_Pi(X,A,idx_grp,co_obs,None)
            pars_0['A'], pars_0['Pi'] = A.copy(), Pi.copy()

    return C,A,X,Pi

def get_zip_size(batch_size, p=None, idx_use=None, max_zip_size=np.inf):
    if batch_size is None:
        zip_size = 1
    elif batch_size == 1:
        zip_size = idx_use.size
    elif batch_size == p:
        zip_size = len(idx_use)  

    return int(np.min((zip_size, max_zip_size)))      


def g_C_l2_Hankel_bad_sis(C,X,k,l,Qs,idx_grp,co_obs,linear=True, W=None):
    "returns l2 Hankel reconstr. stochastic gradient w.r.t. C"

    # sis: subsampled/sparse in space
    
    grad_C = np.zeros(C.shape)
    n = grad_C.shape[1]

    for i in range(len(idx_grp)):

        a,b = idx_grp[i],co_obs[i]
        C_a, C_b  = C[a,:], C[b,:]
        ix_ab = np.ix_(a,b)

        for m in range(1,k+l):

            tmp = X[:,m-1].reshape(n,n).dot(C_b.T)            
            grad_C[a,:] += ( C_a.dot(tmp)   - Qs[m][ix_ab]   ).dot(tmp.T)
            tmp = C_b.dot(X[:,m-1].reshape(n,n))            
            grad_C[a,:] += ( C_a.dot(tmp.T) - Qs[m].T[ix_ab] ).dot(tmp)
            
    return grad_C

def id_C(C, A, Pi, idx_use,idx_co):

    return C

def s_A_l2_Hankel_bad_sis(C,k,l,Qs,idx_grp,co_obs, 
                            linear=True, linearise_X=False, stable=False,
                            A_old=None,verbose=False):
    "returns l2 Hankel reconstr. solution for A given C and the covariances Qs"

    # sis: subsampled/sparse in space
    
    p,n = C.shape

    if p > 1000:
        print('starting extraction of A')
    
    M = np.zeros((n**2, n**2))
    c = np.zeros((n**2, k+l-1))
    for i in range(len(idx_grp)):
        a,b = idx_grp[i], co_obs[i]
        M += np.kron(C[b,:].T.dot(C[b,:]), C[a,:].T.dot(C[a,:]))

        if a.size * b.size * n**2 > 10e6: # size of variable Mab (see below)
            for s in range(b.size):
                Mab = np.kron(C[b[s],:], C[a,:]).T
                for m_ in range(1,k+l):
                    c[:,m_-1] += Mab.dot(Qs[m_][a,b[s]].T.reshape(-1,))
        else:                             # switch to single row of cov mats (size < p * n^2)
            Mab = np.kron(C[b,:], C[a,:]).T
            for m_ in range(1,k+l):
                c[:,m_-1] +=  Mab.dot(Qs[m_][np.ix_(a,b)].T.reshape(-1,)) # Mab * vec(Qs[m](a,b)
    X = np.linalg.solve(M,c)
    for m in range(k+l-1):
        X[:,m] = (X[:,m].reshape(n,n).T).reshape(-1,)
    A, X1 = A_old, None


    if linear:
        cvxopt.solvers.options['show_progress'] = False
        X1,X2 = np.zeros((n, n*(k+l-2))), np.zeros((n, n*(k+l-2)))
        for m in range(k+l-2):
            X1[:,m*n:(m+1)*n] = X[:,m].reshape(n,n)
            X2[:,m*n:(m+1)*n] = X[:,m+1].reshape(n,n)
            
        P = cvxopt.matrix( np.kron(np.eye(n), X1.dot(X1.T)), tc='d')
        q = cvxopt.matrix( - (X2.dot(X1.T)).reshape(n**2,), tc='d')
        #r = np.trace(X2.T.dot(X2))/2

        sol = cvxopt.solvers.qp(P=P,q=q)
        assert sol['status'] == 'optimal'

        A = np.asarray(sol['x']).reshape(n,n)

        # enforcing stability of A
        if stable:
            thresh = 1.
        else:
            thresh = np.inf # no stability enforced 

        lam0 = np.inf
        G, h = np.zeros((0,n**2)), np.array([]).reshape(0,)
        while lam0 > thresh and G.shape[0] < 1000:

            initvals = {'x' : sol['x']}
            
            #print('GC iteration #', G.shape[0])
            
            sol = cvxopt.solvers.qp(P=P,q=q,G=mat(G),h=mat(h),initvals=initvals)

            assert sol['status'] == 'optimal'
            A = np.asarray(sol['x']).reshape(n,n)

            lam0 = np.max(np.abs(np.linalg.eigvals(A)))    
            U,s,V = np.linalg.svd(A)
            g = np.outer( U[:,0], V[0,:] ).reshape(n**2,)
            G = np.vstack((G,np.atleast_2d(g)))
            h = np.ones(G.shape[0])
            #if np.mod(G.shape[0],10) == 0:
            #    print('largest singular value: ' , s[0])
            #    print('largest eigenvalue: ' , lam0)

        AX1 = A.dot(X1)
        if verbose:
            print('MSE reconstruction for A^m X1 - X2',  np.mean( (AX1 - X2)**2 ))
            print('MSE X1', np.mean(X1**2))
            print('MSE X2', np.mean(X2**2))

        #print(lam0)
        if G.shape[0] >= 1000:
            print('Warning! CG failed to guarantee stable A within iteration max')

    return A, X, X1

def linearise_latent_covs(A, X, Pi=None, X1=None, linearity='True'):

    n = int(np.sqrt(X.shape[0]))
    klm1 = X.shape[1]

    if linearity=='True':
        for m in range(1,klm1): # leave X[:,0] = cov(x_{t+1},x_t) unchanged!
            X[:,m] = np.linalg.matrix_power(A,m).dot(Pi).reshape(-1,)


    elif linearity=='first_order':
        if X1 is None:
            X1 = np.zeros((n, n*(klm1-1)))
            for m in range(klm1-1):
                X1[:,m*n:(m+1)*n] = X[:,m].reshape(n,n)

        AX1 = A.dot(X1)
        for m in range(1,klm1): # leave X[:,0] = cov(x_{t+1},x_t) unchanged!
            X[:,m] = AX1[:,(m-1)*n:m*n].reshape(-1,)

    elif linearity=='False':
        pass 
    return X

def id_A(C,idx_grp,co_obs,A):

    return A

def s_Pi_l2_Hankel_bad_sis(X,A,k,l,Qs,Pi=None,verbose=False, sym_psd=True):    

    # requires solution of semidefinite least-squares problem (no solver known for Python):
    # minimize || [A;A^2;A^3] * Pi - [X1; X2; X3] ||
    # subt. to   Pi is symmetric and psd.

    n = A.shape[1]

    As = np.empty((n*(k+l-1),n))
    XT = np.empty((n*(k+l-1),n))
    for m in range(k+l-1):
        XT[m*n:(m+1)*n,:] = X[:,m].reshape(n,n)
        As[m*n:(m+1)*n,:] = np.linalg.matrix_power(A,m+1)
        
    if sym_psd:
        Pi,_,norm_res,muv,r,fail = SDLS.sdls(A=As,B=XT,X0=Pi,Y0=None,
                                             tol=1e-8,verbose=False)
    else:
        print('ls Pie')
        Pi = np.linalg.lstsq(a=As,b=XT)[0]

    if verbose:
        print('MSE reconstruction for A^m Pi - X_m', np.mean( (As.dot(Pi) - XT)**2 ))

    #assert not fail

    return Pi

def id_Pi(X,A,idx_grp,co_obs,Pi=None):

    return Pi


def track_correlations(Qs_full, p, n, Om, C, A, Pi, X, 
    linearity='False'):

    corrs = np.zeros(2)

    if not Qs_full is None:
        kl_ = len(Qs_full)  # covariances for online tracking of fitting
        k,l = kl_-1, 1      # process over unobserved covariances

        H_true = yy_Hankel_cov_mat_Qs(Qs_full,np.arange(p),k,l,n,Om=Om)
        if linearity=='True':
            H_est = yy_Hankel_cov_mat(C,A,Pi,k,l,Om=Om,linear=True)
        else:
            H_est  = yy_Hankel_cov_mat(C,X,None,k,l,Om=Om,linear=False)        

        corrs[0]= np.corrcoef(H_true[np.invert(np.isnan(H_true))], 
                                          H_est[np.invert(np.isnan(H_est))])[0,1]

        H_true = yy_Hankel_cov_mat_Qs(Qs_full,np.arange(p),k,l,n,Om=~Om)
        if linearity=='True':
            H_est = yy_Hankel_cov_mat(C,A,Pi,k,l,Om=~Om,linear=True)
        else:
            H_est  = yy_Hankel_cov_mat(C, X,None,k,l,Om=~Om,linear=False) 

        corrs[1] = np.corrcoef(H_true[np.invert(np.isnan(H_true))], 
                H_est[np.invert(np.isnan(H_est))])[0,1]

    else:
        corrs *= np.nan
    return corrs





###########################################################################
# Utility, semi-scripts, plotting
###########################################################################


def test_run(p,n,Ts=(np.inf,),k=None,l=None,batch_size=None,sub_pops=None,reps=1,
             nr=None, eig_m_r=0.8, eig_M_r=0.99, eig_m_c=0.8, eig_M_c=0.99,
             a=0.001, b1=0.9, b2=0.99, e=1e-8, max_iter_nl = 100,
             linearity=False, stable=False,init='default',save_file=None,
             verbose=False,get_subpop_stats=None, draw_sys=None, sim_data=None):

    # finish setting defaults:
    nr = n//2 if nr is None else nr
    if k is None and l is None:
        k,l = n,n
    batch_size = p if batch_size is None else batch_size
    # default subpops: fully observed
    sub_pops = (np.arange(0,p), np.arange(0,p)) if sub_pops is None else sub_pops

    Ts = np.array(Ts)
    calc_stats = True if np.max(Ts) == np.inf else False    
    draw_data  = True if np.any(np.isfinite(Ts)) else False
    Tmax = int(np.max(Ts[np.isfinite(Ts)]))
    Ts = np.sort(Ts) if np.max(Ts) != np.inf else np.hstack((np.inf, np.sort(Ts[np.isfinite(Ts)])))

    if get_subpop_stats is None:
        raise Exception(('ehem. Will need to properly install modules in the future. ',
                'For now, code requires utility.get_subpop_stats as provided input'))
    if draw_sys is None:
        raise Exception(('ehem. Will need to properly install modules in the future. ',
                'For now, code requires utility.draw_sys as provided input'))
    if sim_data is None:
        raise Exception(('ehem. Will need to properly install modules in the future. ',
                'For now, code requires ssm_scripts.sim_data as provided input'))

    if save_file is None:
        save_file = 'p' + str(p) + 'n' + str(n) + 'r' + str(len(sub_pops))
        save_file = save_file + 'k' + str(k) + 'l' + str(l)


    # draw data, run simulation

    obs_idx, idx_grp, co_obs, overlaps, overlap_grp, idx_overlap, Om, Ovw, Ovc = \
        get_subpop_stats(sub_pops=sub_pops, p=p, verbose=False)

    # draw system matrices    
    ev_r = np.linspace(eig_m_r, eig_M_r, nr)
    ev_c = np.exp(2 * 1j * np.pi * np.random.uniform(size= (n - nr)//2))
    ev_c = np.linspace(eig_m_c, eig_M_c, (n - nr)//2) * ev_c

    pars_true, Qs, Qs_full = draw_sys(p=p,n=n,k=k,l=l,Om=Om, nr=nr, ev_r=ev_r,ev_c=ev_c,calc_stats=calc_stats)
    pars_true['d'], pars_true['mu0'], pars_true['V0'] = np.zeros(p), np.zeros(n), pars_true['Pi'].copy()
    if not draw_data:
        x,y = np.zeros((n,0)), np.zeros((p,0))
    else:
        print(Tmax)
        x,y,_ = sim_data(pars=pars_true, t_tot= Tmax ) 
        x,y = x[:,:,0], y[:,:,0]

    Om_mask = np.asarray(Om, dtype=np.float)
    Om_mask[~Om] = np.nan
    #plt.imshow(Om_mask, interpolation='none')
    #plt.show()

    for T in Ts:        

        T = int(T) if np.isfinite(T) else T
        print('(p,n,T,k,l) = ', (p,n,T,k,l))
        print('max_iter = ', max_iter_nl)
        if p < 100:
            print('sub_pops = ', sub_pops)
        else:
            print('# of sub_pops = ', len(sub_pops))

        if np.isfinite(T):
            for m in range(k+l):
                #for i in range(len(idx_grp)):
                    #slm = np.ix_(idx_grp[i],idx_grp[i])
                    #slt = np.ix_(idx_grp[i], range(m,T+m-(k+l)))
                    #slt = np.ix_(idx_grp[i], range(m,T-(k+l)]))
                    #Qs_full[m][slm] = np.cov(y[slt], y[sls])[:p,p:]
                    #Qs[m] = Qs_full[m] * Om_mask
                Qs_full[m] = np.cov(y[:,m:T+m-(k+l)], y[:,:T-(k+l)])[:p,p:]
                Qs[m] = Qs_full[m] * Om_mask
                #print('Qs', np.mean(np.isfinite(Qs[m])))  
                #print('Qs_full', np.mean(np.isfinite(Qs_full[m])))  
        
        linearity = 'False'
        stable = True
        pars_init, pars_est, traces = run_bad(k=k,l=l,n=n,Qs=Qs,Om=Om,Qs_full=Qs_full,
                                              sub_pops=sub_pops,idx_grp=idx_grp,co_obs=co_obs,obs_idx=obs_idx,
                                              linearity=linearity,stable=stable,init=init,
                                              a=a,b1=b1,b2=b2,e=e,max_iter=max_iter_nl,batch_size=batch_size,
                                              verbose=verbose)
        os.chdir('../fits/nonlinear_cluster')

        """
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
                       'Ts': Ts,
                       'a': a, 'b1': b1, 'b2': b2, 'e': e, 'max_iter_nl': max_iter_nl,
                       'r': len(sub_pops),
                       'sub_pops': sub_pops,
                       'batch_size': batch_size,
                       'y': y,
                       'x': x,
                       'Qs': Qs, 
                       'Qs_full': Qs_full}

        savemat(save_file,save_file_m) # does the actual saving
        """
        pars_true_vec = np.hstack((pars_true['A'].reshape(n*n,),
                            pars_true['Pi'].reshape(n*n,),
                            pars_true['C'].reshape(p*n,)))
        pars_init_vec = np.hstack((pars_init['A'].reshape(n*n,),
                            pars_init['Pi'].reshape(n*n,),
                            pars_init['C'].reshape(p*n,)))
        pars_est_vec  = np.hstack((pars_est['A'].reshape(n*n,),
                            pars_est['Pi'].reshape(n*n,),
                            pars_est['C'].reshape(p*n,)))

        np.savez(save_file + 'T' + str(T), 
                 y=y[:,:T] if np.isfinite(T) else None,
                 x=x[:,:T] if np.isfinite(T) else None,
                 Ts=Ts,
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

        os.chdir('../../dev')


    options = {}


def plot_outputs_l2_gradient_test(pars_true, pars_init, pars_est, k, l, Qs, 
                                       Qs_full, Om, Ovc, Ovw, f_i, g_i, traces=None,
                                       linearity = 'True', idx_grp = None, co_obs = None, 
                                       if_flip = False, m = 1):


    p,n = pars_true['C'].shape

    def plot_mats(thresh=500):
        return p * max((k,l)) <= thresh

    pars_init = set_none_mats(pars_init, p, n)
    f_l2_Hankel_lin = f_l2_Hankel if 'B' in pars_init.keys() else f_l2_Hankel_Pi

    parsv_true = l2_system_mats_to_vec(pars_true['A'],pars_true['Pi'],pars_true['C'])
    parsv_est  = l2_system_mats_to_vec(pars_est['A'], pars_est['Pi'], pars_est['C'])
    parsv_init = l2_system_mats_to_vec(pars_init['A'],pars_init['Pi'],pars_init['C'])

    if linearity=='False': 
        X = s_A_l2_Hankel_bad_sis(pars_est['C'],k,l,Qs,idx_grp,co_obs, linear=False)[1]
    elif linearity == 'first_order':
        X = s_A_l2_Hankel_bad_sis(pars_est['C'],k,l,Qs,idx_grp,co_obs, 
            linear=False, linearise_X=True)[1]


    def f(Om):
        if linearity=='True':
            return f_l2_Hankel_lin(parsv_est,k,l,n,Qs_full, Om) # Qs_full in case ~Om
        else:
            return f_l2_Hankel_nl(pars_est['C'],X,k,l,n,Qs_full,Om)

    print('final squared error on observed parts:', f(Om)) 
    print('final squared error on overlapping parts:', f(Ovw))
    print('final squared error on cross-overlapping parts:', f(Ovc))
    print('final squared error on stitched parts:', f(~Om))
    # this currently is a bit dirty:
    if if_flip:
        C_flip = pars_est['C'].copy()
        C_flip[Om[:,0],:] *= -1
        try:
            parsv_fest  = l2_system_mats_to_vec(pars_est['A'],pars_est['B'],C_flip)
        except:
            parsv_fest  = l2_system_mats_to_vec(pars_est['A'],pars_est['Pi'],C_flip)
        H_flip = yy_Hankel_cov_mat(C_flip,pars_est['A'],pars_est['Pi'],k,l)
        if linearity=='True':
            f_flip = f_l2_Hankel_lin(parsv_fest,k,l,n,Qs,Om)
        else: 
            f_flip = f_l2_Hankel_nl(C_flip,X,k,l,n,Qs,Om)
        print('final squared error on stitched parts (C over first subpop sign-flipped):',
          f_l2_Hankel_lin(parsv_fest,k,l,n,Qs_full,~Om))

    H_true = yy_Hankel_cov_mat_Qs(Qs_full,np.arange(p),k,l,n,Om=None)
    H_0    = yy_Hankel_cov_mat(pars_init['C'],pars_init['A'],pars_init['Pi'],k,l)

    H_obs = yy_Hankel_cov_mat_Qs(Qs,np.arange(p),k,l,n,Om= Om)
    H_obs[np.where(H_obs==0)] = np.nan
    H_sti = yy_Hankel_cov_mat_Qs(Qs_full,np.arange(p),k,l,n,Om=~Om)

    if linearity=='True':
        H_est = yy_Hankel_cov_mat(pars_est['C'],pars_est['A'],
            pars_est['Pi'],k,l)
    else:
        H_est = yy_Hankel_cov_mat(pars_est['C'],X,
            pars_est['Pi'],k,l,linear=False)

    if plot_mats():
        if if_flip:
            plt.figure(figsize=(16,18))
        else:
            plt.figure(figsize=(16,12))

        n_rows = 2
        #n_rows = 3 if if_flip else 2 
        plt.subplot(n_rows,2,1)
        plt.imshow(H_obs,interpolation='none')
        plt.title('Given data matrix (A_true, masked)')
        plt.subplot(n_rows,2,2)
        plt.imshow(H_true,interpolation='none')
        plt.title('True  matrix (A_true)')    
        plt.subplot(n_rows,2,3)
        plt.imshow(H_0,interpolation='none')
        plt.title('Initial matrix (A_0)')
        plt.subplot(n_rows,2,4)
        plt.imshow(H_est,interpolation='none')
        plt.title('Estimated matrix (A_est)')
        #if if_flip:
        #    plt.subplot(n_rows,2,6)
        #    plt.imshow(H_est,interpolation='none')
        #    plt.title('Estimated matrix (A_est, C[\rho_2] sign-flipped)')

        plt.show()
    
    """
    print('plotting true and recovered dynamics matrices')
    plt.figure(figsize=(16,12))
    plt.subplot(1,3,1)
    plt.imshow(pars_init['A'],interpolation='none')
    plt.title('A init')
    plt.subplot(1,3,2)
    plt.imshow(pars_est['A'],interpolation='none')
    plt.title('A est')
    plt.subplot(1,3,3)
    plt.imshow(pars_true['A'],interpolation='none')
    plt.title('A true')
    plt.show()
    """

    print('\n observed covariance entries')
    H_true = yy_Hankel_cov_mat_Qs(Qs_full,np.arange(p),k,l,n,Om=Om)
    if linearity=='True':
        H_est = yy_Hankel_cov_mat(pars_est['C'],pars_est['A'],pars_est['Pi'],
            k,l,Om=Om,linear=True)
    else:
        X = s_A_l2_Hankel_bad_sis(pars_est['C'],k,l,Qs,idx_grp,co_obs, linear=False)[1]
        H_est  = yy_Hankel_cov_mat(pars_est['C'],X,None,k,l,Om=Om,linear=False)        
    if plot_mats():
        plt.figure(figsize=(20,20))
        plt.subplot(3,3,1)
        plt.imshow(H_true, interpolation='none')
        plt.title('observed chunks of true Hankel matrix')
        plt.subplot(3,3,2)
        plt.imshow(H_est, interpolation='none')
        plt.title('observed chunks of rec. Hankel matrix')
        plt.subplot(3,3,3)
        plt.plot(H_true.reshape(-1,), H_est.reshape(-1,), 'k.')
        plt.title('true vs. reconstruction')
        plt.xlabel('H_kl true')
        plt.ylabel('H_kl rec.')
        plt.axis('equal')
    else: 
        plt.figure(figsize=(20,6))
        plt.subplot(1,3,1)
        plt.plot(H_true.reshape(-1,), H_est.reshape(-1,), 'k.')
        plt.title('true vs. reconstr. observed Hankel entries')
        plt.xlabel('H_kl true')
        plt.ylabel('H_kl rec.')
        plt.axis('equal')

    print('correlation:', np.corrcoef(H_true[np.invert(np.isnan(H_true))], 
                                      H_est[np.invert(np.isnan(H_est))])[0,1])
    print('obs. MSE:', np.mean( (H_true[np.invert(np.isnan(H_true))] - 
                                      H_est[np.invert(np.isnan(H_est))])**2 ) )

    print('\n stitched covariance entries')
    H_true = yy_Hankel_cov_mat_Qs(Qs_full,np.arange(p),k,l,n,Om=~Om)
    if linearity=='True':
        H_est = yy_Hankel_cov_mat(pars_est['C'],pars_est['A'],pars_est['Pi'],
            k,l,Om=~Om,linear=True)
    else:
        X = s_A_l2_Hankel_bad_sis(pars_est['C'],k,l,Qs,idx_grp,co_obs, linear=False)[1]
        H_est  = yy_Hankel_cov_mat(pars_est['C'], X,None,k,l,Om=~Om,linear=False)   
    if plot_mats():
        plt.subplot(3,3,4)
        plt.imshow(H_true, interpolation='none')
        plt.title('unobserved chunks of rec. Hankel matrix')
        plt.subplot(3,3,5)
        plt.imshow(H_est, interpolation='none')
        plt.title('stitched chunks of rec. Hankel matrix')
        plt.subplot(3,3,6)
        plt.plot(H_true.reshape(-1,), H_est.reshape(-1,), 'k.')
        plt.title('true vs. reconstruction')
        plt.xlabel('H_kl true')
        plt.ylabel('H_kl rec.')
        plt.axis('equal')
    else:
        plt.subplot(1,3,2)
        plt.plot(H_true.reshape(-1,), H_est.reshape(-1,), 'k.')
        plt.title('true vs. reconstr. un-observed Hankel entries')
        plt.xlabel('H_kl true')
        plt.ylabel('H_kl rec.')
        plt.axis('equal')


    print('correlation:', np.corrcoef(H_true[np.invert(np.isnan(H_true))], 
                                      H_est[np.invert(np.isnan(H_est))])[0,1])

    print('\n full time-lagged covariances, for time-lag m = ', m)
    H_true = pars_true['C'].dot( np.linalg.matrix_power(pars_true['A'],m).dot(pars_true['Pi']) ).dot(pars_true['C'].T)
    if linearity=='True':
        H_est = pars_est['C'].dot( np.linalg.matrix_power(pars_est['A'],m).dot(pars_est['Pi']) ).dot(pars_est['C'].T)
    else:
        H_est = pars_est['C'].dot(X[:,m-1].reshape(n,n).dot(pars_est['C'].T))
    if plot_mats():
        plt.subplot(3,3,7)
        plt.imshow(H_true, interpolation='none')
        plt.title('true time-lagged covariance matrix')
        plt.subplot(3,3,8)
        plt.imshow(H_est, interpolation='none')
        plt.title('rec. time-lagged covariance matrix')
        plt.subplot(3,3,9)
        plt.plot(H_true.reshape(-1,), H_est.reshape(-1,), 'k.')
        plt.title('true vs. reconstruction')
        plt.xlabel('H_kl true')
        plt.ylabel('H_kl rec.')
        plt.axis('equal')
    else:
        plt.subplot(1,3,3)
        plt.plot(H_true.reshape(-1,), H_est.reshape(-1,), 'k.')
        plt.title('true vs. reconstr. un-observed Hankel entries')
        plt.xlabel('H_kl true')
        plt.ylabel('H_kl rec.')
        plt.axis('equal')

    print('correlation:', np.corrcoef(H_true[np.invert(np.isnan(H_true))], 
                                      H_est[np.invert(np.isnan(H_est))])[0,1])
    plt.show()

    if not traces is None:
        if isinstance(traces, np.ndarray):
            fs, len_traces = traces, 1
        elif isinstance(traces, tuple):
            fs, len_traces = traces[0], len(traces)  
            try:
                corrs = traces[1]
                len_corrs = corrs.shape[1]
                plot_corrs = True
            except:
                plot_corrs = False
                pass

        plt.figure(figsize=(20,8))
        plt.subplot(len_traces,1,1)
        plt.plot(fs)
        plt.xlabel('iterations')
        plt.ylabel('target error')
        plt.title('target function vs. iteration count')
        plt.show()

        if plot_corrs:        
            plt.figure(figsize=(20,8))
            plt.subplot(len_traces,1,1)
            plt.plot(np.linspace(0,100, corrs.shape[1]), corrs.T)
            plt.xlabel('percent of total iterations')
            plt.ylabel('correlations of est. and true covariances')
            plt.title('recovered correlations vs. iteration percentage')
            plt.legend(('observed', 'non-observed'))
            plt.show()

def ssidSVD(SIGfp,SIGyy,n, pi_method='proper'):
    
    minVar    = 1e-5
    minVarPi  = 1e-5   

    p = np.size(SIGyy,0)
    UU,SS,VV = np.linalg.svd(SIGfp) # SIGfp = UU.dot(diag(SS).dot(VV)

    UU,SS,VV = UU[:,:n], np.diag(SS[:n]), VV.T[:,:n]   
    Obs = np.dot(UU,SS)

    A = np.linalg.lstsq(Obs[:-p,:],Obs[p:,:])[0]
    C = Obs[:p,:]
    Chat = VV[:p,:n]
    if pi_method=='proper':

        try:
            Pi,_,_ = control.matlab.dare(A=A.T,B=-C.T,Q=np.zeros((n,n)),R=-SIGyy,
                S=Chat.T, E=np.eye(n))    
        except:
            Pi = np.linalg.lstsq(A,np.dot(Chat.T,np.linalg.pinv(C.T)))[0]

    else:
        #warnings.warn(('Will not solve DARE, using heuristics; this might '
        #    'lead to poor estimates of Q and V0'))  
        Pi = np.linalg.lstsq(A,np.dot(Chat.T,np.linalg.pinv(C.T)))[0]

    D, V = np.linalg.eig(Pi)
    if np.any(D < minVarPi):
        D[D < minVarPi] = minVarPi
        Pi = V.dot(np.diag(D)).dot(V.T)
    Pi = np.real( (Pi + Pi.T) / 2 )

    Q = Pi - np.dot(np.dot(A,Pi),A.T)
    D, V = np.linalg.eig(Q); 
    D[D<minVar] = minVar

    Q = np.dot(np.dot(V,np.diag(D)),V.T)
    Q = (Q + Q.T) / 2
    
    # Getting R is too expensive for now
    #R = np.diag(SIGyy-np.dot(np.dot(C,Pi),C.T))
    #R.flags.writeable = True
    #R[R<minVar] = minVar
    #R = np.diag(R)   
    
    return {'A':A, 'Q': Q, 'C': C, 'R': None, 'Pi': Pi}

def set_none_mats(pars, p, n, val=np.nan):

    if pars['A'] is None:
        pars['A'] = val * np.ones((n,n))

    try:
        if pars['B'] is None:
            pars['B'] = val * np.ones((n,n))        
    except: 
        if pars['Pi'] is None:
            pars['Pi'] = val * np.ones((n,n))

    if pars['C'] is None:
        pars['C'] =  val * np.ones((p,n))

    return pars


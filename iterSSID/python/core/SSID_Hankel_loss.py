import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import warnings
import control
from scipy.linalg import solve_discrete_lyapunov as dlyap
from numpy.lib.stride_tricks import as_strided
import cvxopt
import SDLS

def mat(X):
    return cvxopt.matrix(X, tc='d')

###########################################################################
# iterative SSID for stitching via L2 loss on Hankel covariance matrix
###########################################################################

def yy_Hankel_cov_mat(C,A,Pi,k,l,Om=None,linear=True):
    "matrix with blocks cov(y_t+m, y_t) m = 1, ..., k+l-1 on the anti-diagonal"
    
    p,n = C.shape
    if linear:
        assert n == A.shape[1] and n == Pi.shape[0] and n == Pi.shape[1]
    else:
        assert n*n == A.shape[0] and k+l-1 <= A.shape[1]
        
    assert (Om is None) or (Om.shape == (p,p))
    
    H = np.zeros((k*p, l*p))
    
    for kl_ in range(k+l-1):        
        AmPi = np.linalg.matrix_power(A,kl_+1).dot(Pi) if linear else A[:,kl_].reshape(n,n)
        lamK = C.dot(AmPi).dot(C.T)
        
        lamK = lamK if Om is None else lamK * np.asarray( Om, dtype=float) 
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
    
    pe, p = idx.size, Qs[0].shape[0]
        
    assert (Om is None) or (Om.shape == (p,p))
    if not Om is None:
        Om_idx = np.asarray(Om, dtype=float)[np.ix_(idx,idx)]
    
    H = np.zeros((k*pe, l*pe))
    
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
        Pi,_,_ = control.matlab.dare(A=A.T,B=-C.T,Q=np.zeros((n,n)),R=-SIGyy,
            S=Chat.T, E=np.eye(n))    
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

def plot_outputs_l2_gradient_test(pars_true, pars_init, pars_est, k, l, Qs, 
                                       Qs_full, Om, Ovc, Ovw, f_i, g_i, traces=None,
                                       linear = True, idx_grp = None, co_obs = None, 
                                       if_flip = False, m = 1):


    p,n = pars_true['C'].shape

    def plot_mats(thresh=200):
        return p * max((k,l)) <= thresh

    pars_init = set_none_mats(pars_init, p, n)
    f_l2_Hankel_lin = f_l2_Hankel if 'B' in pars_init.keys() else f_l2_Hankel_Pi

    parsv_true = l2_system_mats_to_vec(pars_true['A'],pars_true['Pi'],pars_true['C'])
    parsv_est  = l2_system_mats_to_vec(pars_est['A'], pars_est['Pi'], pars_est['C'])
    parsv_init = l2_system_mats_to_vec(pars_init['A'],pars_init['Pi'],pars_init['C'])

    if not linear: 
        X = s_A_l2_Hankel_bad_sis(pars_est['C'],k,l,Qs,idx_grp,co_obs, linear=linear)[1]


    def f(Om):
        if linear:
            return f_l2_Hankel_lin(parsv_est,k,l,n,Qs_full, Om)
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
        f_flip = f_l2_Hankel_lin(parsv_fest,k,l,n,Qs,Om) if linear else f_l2_Hankel_nl(C_flip,X,k,l,n,Qs,Om)
        print('final squared error on stitched parts (C over first subpop sign-flipped):',
          f_l2_Hankel_lin(parsv_fest,k,l,n,Qs_full,~Om))

    H_true = yy_Hankel_cov_mat(pars_true['C'],pars_true['A'],
        pars_true['Pi'],k,l)
    H_0    = yy_Hankel_cov_mat(pars_init['C'],pars_init['A'],
        pars_init['Pi'],k,l)

    H_obs = yy_Hankel_cov_mat( pars_true['C'],pars_true['A'],
        pars_true['Pi'],k,l, Om)
    H_obs[np.where(H_obs==0)] = np.nan
    H_sti = yy_Hankel_cov_mat( pars_true['C'],pars_true['A'],
        pars_true['Pi'],k,l,~Om)

    if linear:
        H_est = yy_Hankel_cov_mat(pars_est['C'],pars_est['A'],
            pars_est['Pi'],k,l)
    else:
        H_est = yy_Hankel_cov_mat(pars_est['C'],X,
            pars_est['Pi'],k,l,linear=linear)

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
    H_true = yy_Hankel_cov_mat(pars_true['C'],pars_true['A'],pars_true['Pi'],
        k,l,Om=Om,linear=True)
    if linear:
        H_est = yy_Hankel_cov_mat(pars_est['C'],pars_est['A'],pars_est['Pi'],
            k,l,Om=Om,linear=linear)
    else:
        X = s_A_l2_Hankel_bad_sis(pars_est['C'],
            k,l,Qs,idx_grp,co_obs, linear=linear)[1]
        H_est  = yy_Hankel_cov_mat(pars_est['C'], X,None,k,l,Om=Om,linear=linear)        
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

    print('correlation:', np.corrcoef(H_true.reshape(-1,), H_est.reshape(-1,))[0,1])

    print('\n stitched covariance entries')
    H_true = yy_Hankel_cov_mat(pars_true['C'],pars_true['A'],pars_true['Pi'],
        k,l,Om=~Om,linear=True)
    if linear:
        H_est = yy_Hankel_cov_mat(pars_est['C'],pars_est['A'],pars_est['Pi'],
            k,l,Om=~Om,linear=linear)
    else:
        X = s_A_l2_Hankel_bad_sis(pars_est['C'],
            k,l,Qs,idx_grp,co_obs, linear=linear)[1]
        H_est  = yy_Hankel_cov_mat(pars_est['C'], X,None,k,l,Om=~Om,linear=linear)   
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


    print('correlation:', np.corrcoef(H_true.reshape(-1,), H_est.reshape(-1,))[0,1])

    print('\n full time-lagged covariances, for time-lag m = ', m)
    H_true = pars_true['C'].dot( np.linalg.matrix_power(pars_true['A'],m).dot(pars_true['Pi']) ).dot(pars_true['C'].T)
    if linear:
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

    print('correlation:', np.corrcoef(H_true.reshape(-1,), H_est.reshape(-1,))[0,1])
    plt.show()

    if not traces is None:
        if isinstance(traces, np.ndarray):
            fs, len_traces = traces, 1
        elif isinstance(traces, tuple):
            fs, len_traces = traces[0], len(traces)  

        plt.figure(figsize=(20,8))
        plt.subplot(len_traces,1,1)
        plt.plot(fs)
        plt.xlabel('iterations')
        plt.ylabel('target error')
        plt.title('target function vs. iteration count')
        plt.show()

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


###########################################################################
# basic variant: following gradients w.r.t. C, A, B = sqrt(Pi)
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

    p = Qs[0].shape[0]
    A,B,C = l2_vec_to_system_mats(parsv,p,n)
    Pi = B.dot(B.T)

    err = 0.
    for m in range(1,k+l-1):
        APi = np.linalg.matrix_power(A, m).dot(Pi)  
        err += f_l2_block(C,APi,Qs[m],Om)
            
    return err/(k*l)
    
def f_l2_block(C,AmPi,Q,Om):
    "Hankel reconstruction error on an individual Hankel block"

    v = (C.dot(AmPi.dot(C.T)))[Om] - Q[Om] # this is not efficient for spars Om

    return v.dot(v)/(2*np.sum(Om))


def g_l2_Hankel(parsv,k,l,n,Qs,idx_grp,co_obs):
    "returns overall l2 Hankel reconstruction gradient w.r.t. A, B, C"

    p = Qs[0].shape[0]
    A,B,C = l2_vec_to_system_mats(parsv,p,n)
    Pi = B.dot(B.T)

    Aexpm = np.zeros((n,n,k+l))
    Aexpm[:,:,0]= np.eye(n)
    for m in range(1,k+l):
        Aexpm[:,:,m] = A.dot(Aexpm[:,:,m-1])

    grad_A, grad_B, grad_C = np.zeros((n,n)), np.zeros((n,n)), np.zeros((p,n))
    for m in range(1,k+l-1):

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

###########################################################################
# Stochastic gradient descent: following gradients w.r.t. C, A, B = sqrt(Pi)
###########################################################################

def adam_zip(f,g,theta_0,a,b1,b2,e,max_iter,
                converged,Om,idx_grp,co_obs,batch_size=None):
    
    N = theta_0.size
    p = Om.shape[0]
    
    if batch_size is None:
        print('doing full gradients - switching to plain gradient descent')
        b1, b2, e, v_0 = 0, 1.0, 0, np.ones(N)
    elif batch_size == 1:
        print('using size-1 mini-batches')
        v_0 = np.zeros(N)        
    elif batch_size == p:
        print('using size-p mini-batches (coviarance columms)')
        v_0 = np.zeros(N)
    else: 
        raise Exception('cannot handle selected batch size')


    # setting up the stitching context
    is_, js_ = np.where(Om)
    
    # setting up Adam
    t_iter, t, t_zip = 0, 0, 0
    m, v = np.zeros(N), v_0.copy()
    theta, theta_old = theta_0.copy(), np.inf * np.ones(N)

    # setting up the stochastic batch selection:
    batch_draw = l2_sis_draw(p, batch_size, idx_grp, co_obs, is_,js_)

    def g_i(theta, use, co, i):
        
        if batch_size is None:  # eventually pull if-statements out of function def
            return g(theta,idx_use,idx_co)
        elif batch_size == 1:
            return g(theta,(np.array((use[i],)),),(np.array((co[i],)),))
        elif batch_size == p:
            a,b = (co_obs[idx_co[idx_zip]],), (np.array((idx_use[idx_zip],)),)
            return g(theta,a, b)

    # trace function values
    fun = np.empty(max_iter)    
    
    while not converged(theta_old, theta, e, t_iter):

        theta_old = theta.copy()

        t_iter += 1
        idx_use, idx_co = batch_draw()
        if batch_size is None:
            zip_size = 1
        elif batch_size == 1:
            zip_size = idx_use.size
        elif batch_size == p:
            zip_size = len(idx_use)

        for idx_zip in range(zip_size):
            t += 1

            # get data point(s) and corresponding gradients:                    
            grad = g_i(theta,idx_use,idx_co, idx_zip)
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

            theta = theta - a * mh/(np.sqrt(vh) + e)
        
        if t_iter <= max_iter:
            fun[t_iter-1] = f(theta)
            
        if np.mod(t_iter,max_iter//10) == 2:
            print('finished %', t_iter/max_iter)
            print('f = ', fun[t_iter-1])
            
    print('total iterations: ', t)
        
    return theta, fun


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


    # setting up the stitching context
    is_, js_ = np.where(Om)
    
    # setting up Adam
    t_iter, t, t_zip = 0, 0, 0
    m, v = np.zeros(NnA), v_0.copy()
    theta, theta_old = theta_0.copy(), np.inf * np.ones(NnA)

    # setting up the stochastic batch selection:
    batch_draw = l2_sis_draw(p, batch_size, idx_grp, co_obs, is_,js_)

    def g_i(theta, use, co, i):
        
        if batch_size is None:  # eventually pull if-statements out of function def
            return g(theta,idx_use,idx_co)
        elif batch_size == 1:
            return g(theta,(np.array((use[i],)),),(np.array((co[i],)),))
        elif batch_size == p:
            a,b = (co_obs[idx_co[idx_zip]],), (np.array((idx_use[idx_zip],)),)
            return g(theta,a, b)

    # trace function values
    fun = np.empty(max_iter)    
    sig = np.empty(max_iter)    


    
    while not converged(theta_old, theta, e, t_iter):

        theta_old = theta.copy()

        t_iter += 1
        idx_use, idx_co = batch_draw()
        if batch_size is None:
                zip_size = 1
        elif batch_size == 1:
            zip_size = idx_use.size
        elif batch_size == p:
            zip_size = len(idx_use)
        
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
        #print('\n num alpha-resisings: ', c)
        #print('eigvals \n: ', np.sort(np.abs(np.linalg.eigvals(A)))[::-1])
        #print('singular vals \n: ', np.sort(np.linalg.svd(A)[1])[::-1])
        #print('barrier: ', np.linalg.slogdet(np.eye(n)-A.dot(A.T))[1])
        #print('log-barrier: ', - np.log( np.linalg.det(np.eye(n)-A.dot(A.T)) ))
        #print('s(A):', s(A))
        theta[:n*n] = A.reshape(n*n,).copy()


        if t_iter <= max_iter:          # outcomment this eventually - really expensive!
            fun[t_iter-1] = f(theta)
            sig[t_iter-1] = np.max(np.max(np.linalg.svd(A)[1]))
            
        if np.mod(t_iter,max_iter//10) == 2:
            print('f = ', fun[t_iter-1])
            
    print('total iterations: ', t)
        
    return theta, (fun,sig)    

def l2_sis_draw(p, batch_size, idx_grp, co_obs, is_,js_):
    "returns sequence of indices for sets of neuron pairs for SGD"

    if batch_size is None:
        def batch_draw():
            return idx_grp, co_obs    
    elif batch_size == p:
        def batch_draw():
            idx_perm = np.random.permutation(np.arange(p))
            idx_co = np.hstack([ idx * np.ones(idx_grp[idx].size,dtype=np.int32) for idx in range(len(idx_grp)) ])[idx_perm]
            idx_use = np.hstack(idx_grp)[idx_perm]
            return idx_use, idx_co 
    elif batch_size == 1:
        def batch_draw():
            idx = np.random.permutation(len(is_))
            return is_[idx], js_[idx]       

    return batch_draw

########################
# Subsampling in space #
########################

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
    
def g_l2_Hankel_sis(parsv,k,l,n,Qs,idx_grp,co_obs):
    "returns l2 Hankel reconstr. stochastic gradient w.r.t. A, B, C"

    # sis: subsampled/sparse in space
    
    p = Qs[0].shape[0]
    A,B,C = l2_vec_to_system_mats(parsv,p,n)
    Pi = B.dot(B.T)

    Aexpm = np.zeros((n,n,k+l))
    Aexpm[:,:,0]= np.eye(n)
    for m in range(1,k+l):
        Aexpm[:,:,m] = A.dot(Aexpm[:,:,m-1])

    grad_A, grad_B, grad_C = np.zeros((n,n)), np.zeros((n,n)), np.zeros((p,n))
    for m in range(1,k+l-1):

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


###########################################################################
# block coordinate descent: following gradients w.r.t. C, cov(x_{t+m}, x_t)
###########################################################################
# (since we might as well only update C and get A^m*Pi in closed form)

def iter_X_m(CdQCdT_obs, C, Cd, p, n, idx_grp, not_co_obs, X_m):
    "'fast', because only recomputing the non-observed parts in each iteration"

    CdQCdT_stitch = np.zeros(X_m.shape)
    for i in range(len(idx_grp)):
        a, b = idx_grp[i], not_co_obs[i]
        CdQCdT_stitch += (Cd[:,a].dot(C[a,:])).dot(X_m).dot(Cd[:,b].dot(C[b,:]).T)
    return CdQCdT_obs + CdQCdT_stitch

def yy_Hankel_cov_mat_coord_asc(C, Qs, k, l, Om, idx_grp, co_obs, not_co_obs, 
        max_iter=50, idx_init = None):
    "compute model Hankel mat from given C (needs to compute optimal A^m Pi)"

    p,n = C.shape
    Cd = np.linalg.pinv(C)
    Cdi = None if idx_init is None else np.linalg.pinv(C[idx_init,:])

    assert (Om is None) or (Om.shape == (p,p))
    not_Om = np.zeros((p,p),dtype=bool) if Om is None else np.invert(Om)
    Om = np.ones((p,p), dtype=bool) if Om is None else Om

    H = np.zeros((k*p, l*p))
    
    for kl_ in range(k+l-1):        
        Q = Qs[kl_+1]

        if idx_init is None:
            X_m = np.eye(n)
        else:
            X_m = Cdi.dot(Q[np.ix_(idx_init,idx_init)]).dot(Cdi.T)

        if max_iter > 0:
            CdQCdT_obs = np.zeros((n,n))
            for i in range(len(idx_grp)):
                a, b = idx_grp[i], co_obs[i]
                CdQCdT_obs += Cd[:,a].dot(Q[np.ix_(a, b)]).dot(Cd[:,b].T)

            for i in range(max_iter):
                X_m = iter_X_m(CdQCdT_obs,C,Cd,p,n,idx_grp,not_co_obs,X_m)

        Q_est = np.full(Q.shape, np.nan)        
        #Q_est[Om] = (C.dot(X_m).dot(C.T))[Om]        
        for i in range(len(idx_grp)):
            a, b = idx_grp[i], not_co_obs[i]
            Q_est[np.ix_(a,b)] = C[a,:].dot(X_m).dot(C[b,:].T)

        if kl_ < k-0.5:     
            for l_ in range(0, min(kl_ + 1,l)):
                offset0, offset1 = (kl_-l_)*p, l_*p
                H[offset0:offset0+p, offset1:offset1+p] = Q_est
                
        else:
            for l_ in range(0, min(k+l - kl_ -1, l, k)):
                offset0, offset1 = (k - l_ - 1)*p, ( l_ + kl_ + 1 - k)*p
                H[offset0:offset0+p,offset1:offset1+p] = Q_est
            
    return H

def l2_coord_asc_setup(p,idx_grp,obs_idx):

    def co_observed(x, i):
        for idx in obs_idx:
            if x in idx and i in idx:
                return True
        return False        

    co_obs, not_co_obs, full_pop = [], [], np.arange(p)
    for i in range(len(idx_grp)):    
        co_obs.append([idx_grp[x] for x in np.arange(len(idx_grp)) \
            if co_observed(x,i)])
        co_obs[i] = np.sort(np.hstack(co_obs[i]))
        not_co_obs.append( np.setdiff1d(full_pop, co_obs[i]) )

    return co_obs, not_co_obs

def f_l2_Hankel_coord_asc(C, Qs, k, l, n, Om, idx_grp,co_obs,not_co_obs,
        max_iter=50, idx_init = None):

    p = Qs[0].shape[0]
    C = C.reshape(p,n)
    Cd = np.linalg.pinv(C)
    Cdi = None if idx_init is None else np.linalg.pinv(C[idx_init,:])

    err = 0.
    for m in range(1,k+l):
        err += f_l2_coord_asc_block(C,Cd,Qs[m],p,n,Om,idx_grp,co_obs,not_co_obs,
            max_iter,Cdi,idx_init)

    return err/(k*l)

def f_l2_coord_asc_block(C,Cd,Q, p,n, Om,idx_grp,co_obs,not_co_obs,
        max_iter=50,Cdi=None,idx_init=None):

    if Cdi is None or idx_init is None:
        X_m = np.eye(n)
    else:
        X_m = Cdi.dot(Q[np.ix_(idx_init,idx_init)]).dot(Cdi.T)

    if max_iter > 0:
        CdQCdT_obs = np.zeros((n,n))
        for i in range(len(idx_grp)):
            a, b = idx_grp[i], co_obs[i]
            CdQCdT_obs += Cd[:,a].dot(Q[np.ix_(a, b)]).dot(Cd[:,b].T)

        for i in range(max_iter):
            X_m = iter_X_m(CdQCdT_obs,C,Cd,p,n,idx_grp,not_co_obs,X_m)

    Q_est = np.zeros(Q.shape) # this could be reduced to only the needed parts
    for i in range(len(idx_grp)):
        a, b = idx_grp[i], co_obs[i]
        Q_est[np.ix_(a,b)] += (C[a,:].dot(X_m).dot(C[b,:].T))

    v = Q_est[Om] - Q[Om]

    return v.dot(v)/(2*np.sum(Om))

def g_l2_coord_asc(C, Qs, k,l,n, idx_grp,co_obs,not_co_obs, 
        max_iter=50, idx_init=None):

    C = C.reshape(-1, n)
    p, Cd = C.shape[0], np.linalg.pinv(C)

    Cdi = None if idx_init is None else np.linalg.pinv(C[idx_init,:])

    grad = np.zeros((p,n))
    for m in range(1,k+l):
        grad += g_l2_coord_asc_block(C,Cd,Qs[m].copy(),p,n,idx_grp,co_obs,not_co_obs,
            max_iter,Cdi,idx_init)
        
    return ((C.dot(Cd) - np.eye(p)).dot(grad)).reshape(p*n,)

def g_l2_coord_asc_block(C,Cd,Q, p,n, idx_grp,co_obs,not_co_obs,
        max_iter=50,Cdi=None,idx_init=None):

    if Cdi is None or idx_init is None:
        X_m = np.eye(n)
    else:
        X_m = Cdi.dot(Q[np.ix_(idx_init,idx_init)]).dot(Cdi.T)

    if max_iter > 0:
        CdQCdT_obs = np.zeros((n,n))
        for i in range(len(idx_grp)):
            a, b = idx_grp[i], co_obs[i]
            CdQCdT_obs += Cd[:,a].dot(Q[np.ix_(a, b)]).dot(Cd[:,b].T)

        for i in range(max_iter):
            X_m = iter_X_m(CdQCdT_obs,C,Cd,p,n,idx_grp,not_co_obs,X_m)

    for i in range(len(idx_grp)):
        a, b = idx_grp[i], not_co_obs[i]
        Q[np.ix_(a,b)] += (C[a,:].dot(X_m).dot(C[b,:].T))

    QC, QTC = Q.dot(C), Q.T.dot(C)

    return QC.dot(X_m.T) + QTC.dot(X_m)

def g_l2_coord_asc_sgd(C,Cd,Q, p,n, idx_grp,co_obs,not_co_obs,X_ms):

    C = C.reshape(-1, n)
    p, Cd = C.shape[0], np.linalg.pinv(C)

    Cdi = None if idx_init is None else np.linalg.pinv(C[idx_init,:])

    grad = np.zeros((p,n))
    for m in range(1,k+l):
        grad += g_l2_coord_asc_block(C,Cd,Qs[m].copy(),p,n,idx_grp,co_obs,not_co_obs,X_ms[m])
        
    return ((C.dot(Cd) - np.eye(p)).dot(grad)).reshape(p*n,)


########################
# Subsampling in space #
########################

def run_bad(k,l,n,Qs,Om,
            sub_pops,idx_grp,co_obs,obs_idx,
            linear=False,stable=False,init='SSID',
            a=0.001, b1=0.9, b2=0.99, e=1e-8, max_iter=100,batch_size=1):

    p = Qs[0].shape[0]

    if isinstance(init, dict):
        if linear:
            assert np.all([key in init.keys() for key in ('A', 'Pi', 'C')])
        else:
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

    f_i, g_C, g_A, g_Pi = l2_bad_sis_setup(k=k,l=l,n=n,Qs=Qs,
                                           Om=Om,idx_grp=idx_grp,obs_idx=obs_idx,
                                           linear=linear,stable=stable)
    print('starting descent')    
    def converged(theta_old, theta, e, t):
        return True if t >= max_iter else False
    pars_est, fs = adam_zip_bad_stable(f=f_i,g_C=g_C,g_A=g_A,g_Pi=g_Pi,
                                       pars_0=pars_init,
                                       a=a,b1=b1,b2=b2,e=e,
                                       max_iter=max_iter,converged=converged,
                                       Om=Om,idx_grp=idx_grp,co_obs=co_obs,
                                       batch_size=batch_size,linear=linear)                 

    return pars_init, pars_est, (fs,)


def l2_bad_sis_setup(k,l,n,Qs,Om,idx_grp,obs_idx,linearity='True',stable=False):
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

    if linearity == 'True':
        linear_C, linear_A = True, True
    elif linearity == 'False':
        linear_C, linear_A = False, False
    elif linearity == 'first_order':
        linear_C, linear_A = False, True

    def g_C(C,A,Pi,idx_grp,co_obs):
        return g_C_l2_Hankel_bad_sis(C,A,Pi,k,l,Qs,idx_grp,co_obs,linear_C)

    def g_A(C,idx_grp,co_obs,A=None):
        return s_A_l2_Hankel_bad_sis(C,k,l,Qs,idx_grp,co_obs,linear_C,stable,A)

    def g_Pi(X,A,idx_grp,co_obs,Pi=None):
        return s_Pi_l2_Hankel_bad_sis(X,A,k,l,Qs,Pi)

    if linearity == 'True':
        def f(parsv):                        
            return f_l2_Hankel_Pi(parsv,k,l,n,Qs,Om)*np.sum(Om)*(k*l)
    else:
        def f(C,X):
            return f_l2_Hankel_nl(C,X,k,l,n,Qs,Om)*np.sum(Om)*(k*l)

    return f,g_C,g_A,g_Pi

def f_l2_Hankel_nl(C,X,k,l,n,Qs,Om):
    "returns overall l2 Hankel reconstruction error"

    p,n = C.shape

    err = 0.
    for m in range(1,k+l-1):
        err += f_l2_block(C,X[:,m-1].reshape(n,n),Qs[m],Om)
            
    return err/(k*l)

def f_l2_Hankel_Pi(parsv,k,l,n,Qs,Om):
    "returns overall l2 Hankel reconstruction error"

    p = Qs[0].shape[0]
    A,Pi,C = l2_vec_to_system_mats(parsv,p,n)

    err = 0.
    for m in range(1,k+l-1):
        APi = np.linalg.matrix_power(A, m).dot(Pi)  
        err += f_l2_block(C,APi,Qs[m],Om)
            
    return err/(k*l)    
    


def adam_zip_bad_stable(f,g_C,g_A,g_Pi,pars_0,a,b1,b2,e,max_iter,
                converged,Om,idx_grp,co_obs,batch_size=None,linear=True):
    
    if isinstance(pars_0, dict):
        C = pars_0['C'].copy()
        Pi, A = pars_0['Pi'], pars_0['A']
    else:
        N = pars_0.size
        p = Om.shape[0]
        n = np.int(np.round( np.sqrt( p**2/16 + N/2 ) - p/4 ))
        A = pars_0[:n*n].reshape(n,n).copy()
        Pi = pars_0[n*n:2*n*n].reshape(n,n).copy()
        C = pars_0[-p*n:].reshape(p,n).copy()


    if A is None:
        A,X = g_A(C,idx_grp,co_obs,A)
        Pi = g_Pi(X,A,idx_grp,co_obs,Pi)
    else:
        A = A if linear else g_A(C,idx_grp,co_obs)[0] # A is either A or X
        Pi = Pi.copy()
    

    p, n = C.shape

    #print('A init:' , A)

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



    # setting up the stitching context
    is_, js_ = np.where(Om)
    
    # setting up Adam
    t_iter, t, t_zip = 0, 0, 0
    m, v = np.zeros((p,n)), v_0.copy()

    # setting up the stochastic batch selection:
    batch_draw = l2_sis_draw(p, batch_size, idx_grp, co_obs, is_,js_)

    def g_sis(C, A, Pi, use, co, i):

        if batch_size is None:  # eventually pull if-statements out of function def
            return g_C(C, A, Pi, idx_use,idx_co)
        elif batch_size == 1:
            return g_C(C, A, Pi, (np.array((use[i],)),),(np.array((co[i],)),))
        elif batch_size == p:
            a,b = (co_obs[idx_co[idx_zip]],), (np.array((idx_use[idx_zip],)),)
            return g_C(C, A, Pi, a, b)
    
    # trace function values
    fun = np.empty(max_iter)    
    
    C_old = np.inf * np.ones((p,n))
    while not converged(C_old, C, e, t_iter):

        C_old = C.copy()

        idx_use, idx_co = batch_draw()
        if batch_size is None:
            zip_size = 1
        elif batch_size == 1:
            zip_size = idx_use.size
        elif batch_size == p:
            zip_size = len(idx_use)        

        e_frac = 0.1 
        #Cm = np.zeros((p,n)) 
        for idx_zip in range(zip_size):
            t += 1

            # get data point(s) and corresponding gradients: 
            grad = g_sis(C,A,Pi,idx_use,idx_co, idx_zip)
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

            #if idx_zip > (zip_size * (1 -e_frac)):
            #    Cm += C/(zip_size * e_frac)

        A,X =  g_A(C,idx_grp,co_obs,A)

        Pi = g_Pi(X,A,idx_grp,co_obs,Pi) if linear else Pi

        if t_iter < max_iter:          # outcomment this eventually - really expensive!
            theta = np.zeros( (p + 2*n)*n )
            if linear:
                theta[:n*n] = A.reshape(-1,)
                theta[n*n:2*n*n] = Pi.reshape(-1,)
            theta[-p*n:] = C.reshape(-1,)

            fun[t_iter] = f(theta) if linear else f(C,X)
            
        if np.mod(t_iter,max_iter//10) == 0:
            print('finished %', 100*t_iter/max_iter)
            print('f = ', fun[t_iter])

        t_iter += 1

            
    print('total iterations: ', t)

    pars_out = {'C' : C }
    if linear: 
        pars_out['A'], pars_out['Pi'] = A, Pi
    else:
        pars_out['A'], pars_out['Pi'] = np.zeros((n,n)), np.zeros((n,n))

    #print('A final:' , A)

    return pars_out, fun    

def g_C_l2_Hankel_bad_sis(C,A,Pi,k,l,Qs,idx_grp,co_obs,linear=True):
    "returns l2 Hankel reconstr. stochastic gradient w.r.t. C"

    # sis: subsampled/sparse in space
    
    p,n = C.shape
    AmPi = Pi.copy() if linear else None

    grad_C = np.zeros((p,n))
    for m in range(1,k+l-1):

        AmPi = A.dot(AmPi) if linear else A[:,m-1].reshape(n,n)      

        CTC = np.zeros((n,n))
        for i in range(len(idx_grp)):
            a,b = idx_grp[i],co_obs[i]
            C_a, C_b  = C[a,:], C[b,:]

            ix_ab = np.ix_(a,b)
            Ci  = C_a.dot(AmPi.dot(  C_b.T.dot(C_b))) - Qs[m][ix_ab].dot(C_b)
            CiT = C_a.dot(AmPi.T.dot(C_b.T.dot(C_b))) - Qs[m].T[ix_ab].dot(C_b)

            grad_C[a,:] += g_C_l2_idxgrp(Ci,CiT,AmPi)
            
    return grad_C

def id_C(C, A, Pi, idx_use,idx_co):

    return C

def s_A_l2_Hankel_bad_sis(C,k,l,Qs,idx_grp,co_obs, linear=True,stable=False,
                            A_old=None,verbose=False):
    "returns l2 Hankel reconstr. solution for A given C and the covariances Qs"

    # sis: subsampled/sparse in space
    
    p,n = C.shape
    
    M = np.zeros((n**2, n**2))
    c = np.zeros((n**2, k+l-1))
    for i in range(len(idx_grp)):
        a,b = idx_grp[i], co_obs[i]
        M += np.kron(C[a,:].T.dot(C[a,:]), C[b,:].T.dot(C[b,:]))
        Mab = np.kron(C[a,:], C[b,:])
        for m_ in range(1,k+l):
            c[:,m_-1] +=  Mab.T.dot(Qs[m_][np.ix_(a,b)].reshape(-1,))
    X = np.linalg.solve(M,c)
    A = X


    if linear:
        cvxopt.solvers.options['show_progress'] = False
        X1,X2 = np.zeros((n, n*(k+l-2))), np.zeros((n, n*(k+l-2)))
        for m in range(k+l-2):
            X1[:,m*n:(m+1)*n] = X[:,m].reshape(n,n)
            X2[:,m*n:(m+1)*n] = X[:,m+1].reshape(n,n)
            
        P = cvxopt.matrix( np.kron(np.eye(n), X1.dot(X1.T)), tc='d')
        q = cvxopt.matrix( - (X2.dot(X1.T)).reshape(n**2,), tc='d')


        sol = cvxopt.solvers.qp(P=P,q=q)
        assert sol['status'] == 'optimal'

        r = np.trace(X2.T.dot(X2))/2
        A = np.asarray(sol['x']).reshape(n,n)

        if verbose:
            print('MSE reconstruction for A^m X1 - X2',  np.mean( (A.dot(X1) - X2)**2 ))
            print('MSE X1', np.mean(X1**2))
            print('MSE X2', np.mean(X2**2))

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

        #print(lam0)
        if G.shape[0] >= 1000:
            print('Warning! CG failed to guarantee stable A within iteration max')

    return A, X

def id_A(C,idx_grp,co_obs,A):

    return A

def s_Pi_l2_Hankel_bad_sis(X,A,k,l,Qs,Pi=None,verbose=False):    

    # requires solution of semidefinite least-squares problem (no solver known for Python):
    # minimize || [A;A^2;A^3] * Pi - [X1; X2; X3] ||
    # subt. to   Pi is symmetric and psd.

    n = A.shape[0]

    As = np.empty((n*(k+l-2),n))
    XT = np.empty((n*(k+l-2),n))
    for m in range(k+l-2):
        XT[m*n:(m+1)*n,:] = X[:,m].reshape(n,n)
        As[m*n:(m+1)*n,:] = np.linalg.matrix_power(A,m+1)
        
    Pi,_,norm_res,muv,r,fail = SDLS.sdls(A=As,B=XT,X0=Pi,Y0=None,tol=1e-8,verbose=False)

    if verbose:
        print('MSE reconstruction for A^m Pi - X_m', np.max(np.abs(As.dot(Pi) - XT)))

    #assert not fail

    return Pi

def id_Pi(X,A,idx_grp,co_obs,Pi=None):

    return Pi

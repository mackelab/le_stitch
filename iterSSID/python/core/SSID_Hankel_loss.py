import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import warnings
import control
from scipy.linalg import solve_discrete_lyapunov as dlyap
from numpy.lib.stride_tricks import as_strided
import cvxopt

def mat(X):
    return cvxopt.matrix(X, tc='d')

###########################################################################
# iterative SSID for stitching via L2 loss on Hankel covariance matrix
###########################################################################

def yy_Hankel_cov_mat(C,A,Pi,k,l,Om=None):
    "matrix with blocks cov(y_t+m, y_t) m = 1, ..., k+l-1 on the anti-diagonal"
    
    p,n = C.shape
    assert n == A.shape[1] and n == Pi.shape[0] and n == Pi.shape[1]
    
    assert (Om is None) or (Om.shape == (p,p))
    
    H = np.zeros((k*p, l*p))
    
    for kl_ in range(k+l-1):        
        lamK = (C.dot(np.linalg.matrix_power(A,kl_+1).dot(Pi))).dot(C.T)
        
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
    
def f_l2_block(C,A,Q,Om):
    "Hankel reconstruction error on an individual Hankel block"

    v = (C.dot(A.dot(C.T)))[Om] - Q[Om] # this is not efficient for spars Om

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

def plot_outputs_l2_gradient_test(pars_true, pars_init, pars_est, k, l, Qs, 
                                       Qs_full, Om, Ovc, Ovw, f_i, g_i,
                                       if_flip = False):

    n = pars_true['A'].shape[0]
    parsv_true = l2_system_mats_to_vec(pars_true['A'],
        pars_true['B'],pars_true['C'])
    parsv_est  = l2_system_mats_to_vec(pars_est['A'], 
        pars_est['B'], pars_est['C'])
    parsv_init = l2_system_mats_to_vec(pars_init['A'],
        pars_init['B'],pars_init['C'])
    print('final squared error on observed parts:', 
          f_l2_Hankel(parsv_est,k,l,n,Qs, Om))
    print('final squared error on overlapping parts:', 
          f_l2_Hankel(parsv_est,k,l,n,Qs,Ovw))
    print('final squared error on cross-overlapping parts:',
          f_l2_Hankel(parsv_est,k,l,n,Qs,Ovc))
    print('final squared error on stitched parts:',
          f_l2_Hankel(parsv_est,k,l,n,Qs_full,~Om))
    # this currently is a bit dirty:
    if if_flip:
        C_flip = pars_est['C'].copy()
        C_flip[Om[:,0],:] *= -1
        parsv_fest  = l2_system_mats_to_vec(pars_est['A'],pars_est['B'],C_flip)
        H_flip = yy_Hankel_cov_mat(C_flip,pars_est['A'],pars_est['Pi'],k,l)
        print('final squared error on stitched parts (C over first subpop sign-flipped):',
          f_l2_Hankel(parsv_fest,k,l,n,Qs_full,~Om))

    H_true = yy_Hankel_cov_mat(pars_true['C'],pars_true['A'],
        pars_true['Pi'],k,l)
    H_0    = yy_Hankel_cov_mat(pars_init['C'],pars_init['A'],
        pars_init['Pi'],k,l)

    H_obs = yy_Hankel_cov_mat( pars_true['C'],pars_true['A'],
        pars_true['Pi'],k,l, Om)
    H_obs[np.where(H_obs==0)] = np.nan
    H_sti = yy_Hankel_cov_mat( pars_true['C'],pars_true['A'],
        pars_true['Pi'],k,l,~Om)
    H_est = yy_Hankel_cov_mat(pars_est['C'],pars_est['A'],
        pars_est['Pi'],k,l)


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


###########################################################################
# Stochastic gradient descent: following gradients w.r.t. C, A, B = sqrt(Pi)
###########################################################################

def adam_zip(f,g,theta_0,a,b1,b2,e,max_iter,
                converged,Om,idx_grp,co_obs,batch_size=None):
    
    N = theta_0.size
    p = Om.shape[0]
    
    if batch_size is None:
        print('doing full gradients - switching to plain gradient descent')
        b1, b2, e, v_0 = 0, 0, 0, np.ones(N)
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
        
        zip_size = idx_use.size if isinstance(idx_use, np.ndarray) else len(idx_use)        
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
        b1, b2, e, v_0 = 0, 0, 0, np.ones(NnA)
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
    
    while not converged(theta_old, theta, e, t_iter):

        theta_old = theta.copy()

        t_iter += 1
        idx_use, idx_co = batch_draw()
        
        zip_size = idx_use.size if isinstance(idx_use, np.ndarray) else len(idx_use)        
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
        A, a_tmp = (theta[:n*n] - a_A/p * grad[:n*n]).reshape(n,n), a_A
        c = 0
        while not s(A) and c < 10000:
            c += 1
            a_tmp = tau * a_tmp
            A = (theta[:n*n] - a_tmp * grad[:n*n]).reshape(n,n)
        if c == 10000:
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
            
        if np.mod(t_iter,max_iter//10) == 2:
            print('f = ', fun[t_iter-1])
            
    print('total iterations: ', t)
        
    return theta, fun    

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

def l2_bad_sis_setup(k,l,n,Qs,Om,idx_grp,obs_idx,stable):
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
    def g_C(C,A,Pi,idx_grp,co_obs):
        return g_C_l2_Hankel_bad_sis(C,A,Pi,k,l,Qs,idx_grp,co_obs)

    def g_A(C,idx_grp,co_obs,A=None):
        return s_A_l2_Hankel_bad_sis(C,k,l,Qs,idx_grp,co_obs,stable,A)

    def f(parsv):                        
        return f_l2_Hankel(parsv,k,l,n,Qs,Om)*np.sum(Om)*(k*l)

    return f,g_C,g_A

def adam_zip_bad_stable(f,g_C,g_A,pars_0,a,b1,b2,e,max_iter,
                converged,Om,idx_grp,co_obs,batch_size=None):
    
    if isinstance(pars_0, dict):
        C, Pi, B, A = pars_0['C'].copy(), pars_0['Pi'].copy(), pars_0['B'].copy(), pars_0['A'].copy()
    else:
        N = pars_0.size
        p = Om.shape[0]
        n = np.int(np.round( np.sqrt( p**2/16 + N/2 ) - p/4 ))
        A = pars_0[:n*n].reshape(n,n).copy()
        B = pars_0[n*n:2*n*n].reshape(n,n).copy()
        C = pars_0[-p*n:].reshape(p,n).copy()
        Pi = B.dot(B.T)

    p, n = C.shape

    #print('A init:' , A)

    if batch_size is None:
        print('doing full gradients - switching to plain gradient descent')
        b1, b2, e, v_0 = 0, 1., 0, np.ones((p,n))
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

        t_iter += 1
        idx_use, idx_co = batch_draw()
        
        zip_size = idx_use.size if isinstance(idx_use, np.ndarray) else len(idx_use)    
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

        A =  g_A(C,idx_grp,co_obs,A)

        B  = B.copy()
        Pi = Pi.copy() #s_Pi_l2_Hankel_sis(C,A,k,l,Qs,idx_grp,co_obs)

        if t_iter <= max_iter:          # outcomment this eventually - really expensive!
            theta = np.zeros(C.size + A.size + Pi.size)
            theta[:n*n] = A.reshape(-1,)
            theta[n*n:2*n*n] = B.reshape(-1,)
            theta[-p*n:] = C.reshape(-1,)
            fun[t_iter-1] = f(theta)
            
        if np.mod(t_iter,max_iter//10) == 2:
            print('f = ', fun[t_iter-1])
            
    print('total iterations: ', t)

    theta = np.zeros(C.size + A.size + Pi.size)
    theta[:n*n] = A.reshape(-1,)
    theta[n*n:2*n*n] = B.reshape(-1,)
    theta[-p*n:] = C.reshape(-1,)

    #print('A final:' , A)

    return theta, fun    

def g_C_l2_Hankel_bad_sis(C,A,Pi,k,l,Qs,idx_grp,co_obs):
    "returns l2 Hankel reconstr. stochastic gradient w.r.t. C"

    # sis: subsampled/sparse in space
    
    p,n = C.shape
    AmPi = Pi.copy()

    grad_C = np.zeros((p,n))
    for m in range(1,k+l-1):

        AmPi = A.dot(AmPi)            

        CTC = np.zeros((n,n))
        for i in range(len(idx_grp)):
            a,b = idx_grp[i],co_obs[i]
            C_a, C_b  = C[a,:], C[b,:]

            ix_ab = np.ix_(a,b)
            Ci  = C_a.dot(AmPi.dot(  C_b.T.dot(C_b))) - Qs[m][ix_ab].dot(C_b)
            CiT = C_a.dot(AmPi.T.dot(C_b.T.dot(C_b))) - Qs[m].T[ix_ab].dot(C_b)

            grad_C[a,:] += g_C_l2_idxgrp(Ci,CiT,AmPi)
            
    return grad_C

def s_A_l2_Hankel_bad_sis(C,k,l,Qs,idx_grp,co_obs, stable=False,A_old=None):
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


    cvxopt.solvers.options['show_progress'] = False
    X1,X2 = np.zeros((n, n*(k+l-2))), np.zeros((n, n*(k+l-2)))
    for m in range(k+l-2):
        X1[:,(m)*n:(m+1)*n] = X[:,m].reshape(n,n)
        X2[:,(m)*n:(m+1)*n] = X[:,m+1].reshape(n,n)
        
    P = cvxopt.matrix( np.kron(np.eye(n), X1.dot(X1.T)), tc='d')
    q = cvxopt.matrix( - (X2.dot(X1.T)).reshape(n**2,), tc='d')


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

    #print(lam0)
    if G.shape[0] >= 1000:
        print('Warning! B.Boots failed to guarantee stable A within iteration max')
    return A

def id_A(C,idx_grp,co_obs,A):

    return A

def s_Pi_l2_Hankel_bad_sis(C,A,k,l,Qs,idx_grp,co_obs):

    return Pi

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import warnings
#import control
from scipy.linalg import solve_discrete_lyapunov as dlyap
from numpy.lib.stride_tricks import as_strided
from text import progprint_xrange
import cvxopt
import SDLS
import os
from utility import chunking_blocks, yy_Hankel_cov_mat, yy_Hankel_cov_mat_Qs


###########################################################################
# block coordinate descent: following gradients w.r.t. C, cov(x_{t+m}, x_t)
###########################################################################

def run_bad(k,l,n,Qs,
            Om,sub_pops,idx_grp,co_obs,obs_idx,
            linearity='False',stable=False,init='SSID',
            alpha=0.001, b1=0.9, b2=0.99, e=1e-8, 
            max_iter=100, max_zip_size=np.inf, batch_size=1,
            verbose=False, Qs_full=None, sym_psd=True, lag_range = None):

    if not Om is None:
        p = Om.shape[0]
    else:
        p = Qs_full[1].shape[0] if Qs_full[0] is None else Qs_full[0].shape[0] 

    if isinstance(init, dict):
        assert 'C' in init.keys()

        pars_init = init.copy()

    elif init =='SSID':

        print('getting initial param. values (SSID on largest subpopulation)')
        sub_pop_sizes = [ len(sub_pops[i]) for i in range(len(sub_pops))]
        idx = sub_pops[np.argmax(sub_pop_sizes)]
        H_kl = yy_Hankel_cov_mat_Qs(Qs=Qs,idx=idx,k=k,l=l,n=n,Om=Om)
        pars_ssid = ssidSVD(H_kl,Qs[0][np.ix_(idx,idx)],n,pi_method='proper')
        U,S,_ = np.linalg.svd(pars_ssid['Pi'])
        M = np.diag(1/np.sqrt(S)).dot(U.T)    
        pars_init = {'A'  : M.dot(pars_ssid['A']).dot(np.linalg.inv(M)),
             'Pi' : M.dot(pars_ssid['Pi']).dot(M.T),
             'B'  : np.eye(n), 
             'C'  : np.random.normal(size=(p,n)),
             'R'  : 10e-5 * np.ones(p)} #pars_ssid['C'].dot(np.linalg.inv(M))}   

    elif init =='default':
        pars_init = {'A'  : np.diag(np.linspace(0.89, 0.91, n)),
             'Pi' : np.eye(n),
             'B'  : np.eye(n), 
             'C'  : np.random.normal(size=(p,n)),
             'R'  : np.zeros(p)} #pars_ssid['C'].dot(np.linalg.inv(M))}   


    f_i, g_C, g_A, g_Pi, g_R, s_R, track_corrs = l2_bad_sis_setup(k=k,l=l,n=n,
                                           Qs=Qs,Qs_full=Qs_full,Om=Om,
                                           idx_grp=idx_grp,obs_idx=obs_idx,
                                           linearity=linearity,stable=stable,
                                           verbose=verbose,sym_psd=sym_psd,
                                           batch_size=batch_size)
    print('starting descent')    
    def converged(theta_old, theta, e, t):
        return True if t >= max_iter else False
    pars_est, traces = adam_zip_bad_stable(f=f_i,g_C=g_C,g_A=g_A,g_Pi=g_Pi,
                                        g_R=g_R,s_R=s_R,
                                        pars_0=pars_init,linearity=linearity,
                                        alpha=alpha,b1=b1,b2=b2,e=e,
                                        max_zip_size=max_zip_size,
                                        max_iter=max_iter,converged=converged,
                                        Om=Om,idx_grp=idx_grp,co_obs=co_obs,
                                        batch_size=batch_size,
                                        lag_range=lag_range,
                                        track_corrs=track_corrs)

    return pars_init, pars_est, traces



# decorations

def l2_bad_sis_setup(k,l,n,Qs,Om,idx_grp,obs_idx,Qs_full=None, 
                        linearity='True', stable=False, sym_psd=True, W=None,
                        verbose=False, batch_size=None):
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
        def g_C(C,X,R,idx_grp,co_obs,lag_range=None):
            return g_C_l2_Hankel_bad_sis(C,X,R,k,l,Qs,
                idx_grp,co_obs,lag_range,linear=True)

    elif linearity == 'first_order':
        linear_A = True
        linearise_X = True
        def g_C(C,X,R,idx_grp,co_obs,lag_range=None):
            return g_C_l2_Hankel_bad_sis(C,X,R,k,l,Qs,
                idx_grp,co_obs,lag_range,linear=False)

    elif linearity == 'False':
        linear_A = False
        linearise_X = False
        def g_C(C,X,R,idx_grp,co_obs,lag_range=None):
            return g_C_l2_Hankel_bad_sis(C,X,R,k,l,Qs,
                idx_grp,co_obs,lag_range,linear=False)


    def g_A(C,R,idx_grp,co_obs,A=None):
        return s_A_l2_Hankel_bad_sis(C,R,k,l,Qs,idx_grp,co_obs,
            linear=linear_A,linearise_X=linearise_X,stable=stable,A_old=A,
            verbose=verbose)

    if batch_size == None:
        def g_R(R, C, Pi, a, b):
            return s_R_l2_Hankel_bad_sis_block(R,C,Pi,Qs[0], a, b)
    else:
        def g_R(R, C, Pi, a, b=None):
            return s_R_l2_Hankel_bad_sis(R,C,Pi,Qs[0], a)

    def s_R(R,C,Pi):
        return s_R_l2_Hankel_bad_sis_block(R,C,Pi,Qs[0], idx_grp, co_obs)

    def g_Pi(X,A,idx_grp,co_obs,Pi=None):
        return s_Pi_l2_Hankel_bad_sis(X,A,k,l,Qs,Pi,
            verbose=verbose,sym_psd=sym_psd)

    if linearity == 'True':
        def f(C,A,Pi,R):                        
            return f_l2_Hankel_ln(C,A,Pi,R,k,l,Qs,idx_grp,co_obs)
    else:
        def f(C,X,R):
            return f_l2_Hankel_nl(C,X,R,k,l,Qs,idx_grp,co_obs)

    def track_corrs(C, A, Pi, X) :
         return track_correlations(Qs_full, p, n, Om, C, A, Pi, X, 
            linearity=linearity)

    return f,g_C,g_A,g_Pi,g_R,s_R,track_corrs

def l2_sis_draw(p, batch_size, idx_grp, co_obs, g_C, g_R, Om=None):
    "returns sequence of indices for sets of neuron pairs for SGD"

    if batch_size is None:

        def batch_draw():
            return idx_grp, co_obs    
        def g_sis_C(C, X, R, a, b, i, lag_range=None):
            return g_C(C, X, R, a, b, lag_range)
        def g_sis_R(R, C, X0, a, b, i):
            return g_R(R, C, X0, a, b)

    elif batch_size == p:

        def batch_draw():
            b = np.empty(p, dtype=np.uint8)
            start = 0
            for idx in range(len(idx_grp)):
                end = start + idx_grp[idx].size
                b[start:end] = idx
                start = end
            a = np.hstack(idx_grp)
            idx_perm = np.random.permutation(np.arange(p))
            return a[idx_perm], b[idx_perm]
        def g_sis_C(C, X, R, a, b, i, lag_range=None):
            a,b = (co_obs[b[i]],), (np.array((a[i],)),)
            return g_C(C, X, R, a, b, lag_range)
        def g_sis_R(R, C, X0, a, b, i):
            return g_R(R, C, X0, a[i], None)

    elif batch_size == 1:

        is_,js_ = np.where(Om)
        def batch_draw():
            idx = np.random.permutation(len(is_))
            return is_[idx], js_[idx]       
        def g_sis_C(C, X, R, a, b, i, lag_range=None):
            return g_C(C, X, R, 
                       (np.array((a[i],)),),(np.array((b[i],)),),
                       lag_range)
        def g_sis_R(R, C, X0, a, b, i):
            return g_R(R, C, X0, a[i], None)
    return batch_draw, g_sis_C, g_sis_R



# main optimiser

def adam_zip_bad_stable(f,g_C,g_A,g_Pi,g_R,s_R,track_corrs,pars_0,
                alpha,b1,b2,e,max_iter,
                converged,batch_size,lag_range,max_zip_size,
                Om,idx_grp,co_obs,linearity='False'):

    # initialise pars
    C,A,X,Pi,R = set_adam_init(pars_0,g_A,g_Pi,g_R,idx_grp,co_obs,linearity)
    p, n = C.shape

    # setting up Adam
    b1,b2,e,v_0 = set_adam_pars(batch_size,p,n,b1,b2,e)
    t_iter, t, t_zip = 0, 0, 0
    ct_iter, corrs  = 0, np.zeros((2, 11))
    m, v = np.zeros((p,n)), v_0.copy()
    def zip_range(zip_size):
        if p > 1000:
            return progprint_xrange(zip_size, perline=100)
        else:
            return range(zip_size)

    # setting up the stochastic batch selection:
    batch_draw, g_sis_C, g_sis_R = l2_sis_draw(p, batch_size, idx_grp, co_obs, 
                                                g_C, g_R, Om)

    # trace function values
    fun = np.empty(max_iter)    
    
    C_old = np.inf * np.ones((p,n))
    while not converged(C_old, C, e, t_iter):

        # updating C: full SGD pass over data
        C_old = C.copy()
        a, b = batch_draw()        
        zip_size = get_zip_size(batch_size, p, a, max_zip_size)
        for idx_zip in zip_range(zip_size):
            t += 1

            # get data point(s) and corresponding gradients: 
            grad, kl_ = g_sis_C(C,X,R,a,b, idx_zip, lag_range)
            #grad *= 0

            #print('||g(C)||', np.sum(grad**2))

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

            if 0 in kl_:
                R = g_sis_R(R,C,X[:,0].reshape(n,n), a, b, idx_zip)

            C -= alpha * mh/(np.sqrt(vh) + e)


        # updating A: solving linear least squares
        A,X,X1 =  g_A(C,R,idx_grp,co_obs,A)

        # updating Pi: sovling (constraint) linear least squares
        Pi = g_Pi(X,A,idx_grp,co_obs,Pi) if linearity=='True' else Pi
        linearise_latent_covs(A, X, Pi, X1, linearity) 

        if t_iter < max_iter:          # really expensive!
            if linearity=='True':
                fun[t_iter] = f(C,A,Pi,R) 
            else:
                fun[t_iter] = f(C,X,R)

        if np.mod(t_iter,max_iter//10) == 0:
            print('finished %', 100*t_iter/max_iter)
            print('f = ', fun[t_iter])
            corrs[:,ct_iter] = track_corrs(C, A, Pi, X) 
            ct_iter += 1

        t_iter += 1

    # final round over R (we before only updated R_ii just-in-time)
    R = s_R(R, C, X[:,0].reshape(n,n))

    corrs[:,ct_iter] = track_corrs(C, A, Pi, X)

    print('total iterations: ', t)

    pars_out = {'C' : C, 'X': X, 'R' : R }
    if linearity=='True': 
        pars_out['A'], pars_out['Pi'] = A, Pi
    elif linearity=='first_order':
        pars_out['A'], pars_out['Pi'] = A, np.zeros((n,n))
    if linearity=='False': 
        pars_out['A'], pars_out['Pi'] = np.zeros((n,n)), np.zeros((n,n))

    return pars_out, (fun,corrs)    



# setup

def set_adam_pars(batch_size,p,n,b1,b2,e):

    if batch_size is None:
        print('doing batch gradients - switching to plain gradient descent')
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

def set_adam_init(pars_0, g_A, g_Pi, g_R, idx_grp, co_obs, linearity):

    C, Pi = pars_0['C'].copy(), None
    p,n = C.shape
    R = pars_0['R'].copy() if 'R' in pars_0.keys() else np.zeros(p)
    A,X,_ = g_A(C,R,idx_grp,co_obs,None)

    if linearity=='True':
        if 'A' in pars_0.keys() and not pars_0['A'] is None:
            A,Pi = pars_0['A'].copy(), pars_0['Pi'].copy()
        else: 
            print('getting initial Pi now')
            Pi = g_Pi(X,A,idx_grp,co_obs,None)
            pars_0['A'], pars_0['Pi'] = A.copy(), Pi.copy()

    return C,A,X,Pi,R

def get_zip_size(batch_size, p=None, a=None, max_zip_size=np.inf):

    if batch_size is None:
        zip_size = 1
    elif batch_size == 1:
        zip_size = a.size
    elif batch_size == p:
        zip_size = len(a)  

    return int(np.min((zip_size, max_zip_size)))      



# evaluation of target loss function

def f_l2_Hankel_nl(C,X,R,k,l,Qs,idx_grp,co_obs):
    "returns overall l2 Hankel reconstruction error"

    p,n = C.shape
    err = f_l2_inst(C,X[:,0].reshape(n,n),R,Qs[0],idx_grp,co_obs)
    for m in range(1,k+l):
        err += f_l2_block(C,X[:,m].reshape(n,n),Qs[m],idx_grp,co_obs)
            
    return err/(k*l)

def f_l2_Hankel_ln(C,A,Pi,R,k,l,Qs,idx_grp,co_obs):
    "returns overall l2 Hankel reconstruction error"

    err = f_l2_inst(C,Pi,R,Qs[0],idx_grp,co_obs)
    for m in range(1,k+l):
        APi = np.linalg.matrix_power(A, m).dot(Pi)  
        err += f_l2_block(C,APi,Qs[m],idx_grp,co_obs)
            
    return err/(k*l)    
    
def f_l2_block(C,AmPi,Q,idx_grp,co_obs):
    "Hankel reconstruction error on an individual Hankel block"

    err = 0.
    for i in range(len(idx_grp)):
        err_ab = 0.
        a,b = idx_grp[i],co_obs[i]
        C_a, C_b  = C[a,:], C[b,:]

        def f(idx_i, idx_j, i=None, j=None):
            v = (C_a[idx_i,:].dot(AmPi).dot(C_b[idx_j,:].T) - \
                 Q[np.ix_(a[idx_i],b[idx_j])]).reshape(-1,)
            return v.dot(v)

        err += chunking_blocks(f, a, b, 1000)/(2*a.size*b.size)

    return err

def f_l2_inst(C,Pi,R,Q,idx_grp,co_obs):
    "reconstruction error on the instantaneous covariance"

    err = 0.
    if not Q is None:
        for i in range(len(idx_grp)):

            a,b = idx_grp[i],co_obs[i]
            C_a, C_b  = C[a,:], C[b,:]

            def f(idx_i, idx_j, i, j):
                v = (C_a[idx_i,:].dot(Pi).dot(C_b[idx_j,:].T) - \
                     Q[np.ix_(a[idx_i],b[idx_j])])
                if i == j:
                    idx_R = np.where(np.in1d(a[idx_i],b[idx_j]))[0]
                    v[idx_R, np.arange(idx_R.size)] += R[a[idx_j][idx_R]]
                v = v.reshape(-1,)
                return v.dot(v)

            err += chunking_blocks(f, a, b, 1000)/(2*a.size*b.size)

    return err


# gradients (g_*) & solvers (s_*) for model parameters

def g_C_l2_Hankel_bad_sis(C,X,R,k,l,Qs,
                            idx_grp,co_obs,lag_range=None,
                            linear=True, W=None):
    "returns l2 Hankel reconstr. stochastic gradient w.r.t. C"

    # sis: subsampled/sparse in space
    
    grad_C = np.zeros(C.shape)
    n = grad_C.shape[1]

    if lag_range is None:
        lag_range = range(k+l)
    elif lag_range == 1:
        lag_range = (np.random.randint(k+l),)

    for i in range(len(idx_grp)):

        a,b = idx_grp[i],co_obs[i]
        C_a, C_b  = C[a,:], C[b,:]
        ix_ab = np.ix_(a,b)
        ix_ba = np.ix_(b,a)

        for m in lag_range:

            if m==0 and not Qs[0] is None:
                idx_R = np.where(np.in1d(a,b))[0]
                tmp  = X[:,0].reshape(n,n).dot(C_b.T)       # n-by-1 vector
                tmpQ = C_a.dot(tmp) - Qs[0][ix_ab]          # p-by-1 vector
                tmpQ[idx_R, np.arange(b.size)] -= R[b]
                grad_C[a,:] += tmpQ.dot(tmp.T)              # p-by-n matrix

                tmp  = C_b.dot(X[:,0].reshape(n,n))         # n-by-1 vector    
                tmpQ = C_a.dot(tmp.T) - Qs[0][ix_ba].T      # p-by-1 vector       
                tmpQ[idx_R, np.arange(b.size)] -= R[b]
                grad_C[a,:] += tmpQ.dot(tmp)                # p-by-n matrix                

            elif m > 0:
                tmp = X[:,m].reshape(n,n).dot(C_b.T)    # n-by-1 vector        
                grad_C[a,:] += ( C_a.dot(tmp)   - Qs[m][ix_ab]   ).dot(tmp.T)
                tmp = C_b.dot(X[:,m].reshape(n,n))      # n-by-1 vector            
                grad_C[a,:] += ( C_a.dot(tmp.T) - Qs[m][ix_ba].T ).dot(tmp)
            
    return grad_C, lag_range

def id_C(C, A, Pi, a, b):

    return C



def g_R_l2_Hankel_bad_sis(R, C, Pi, cov_y, a):

    p,n = C.shape

    g = np.zeros(p)
    if not cov_y is None:
        g[a] = R[a] + C[a,:].dot(Pi.dot(C[a,:].T)) - cov_y[a, a]

    R -= 0.01 * g

    return R

def g_R_l2_Hankel_bad_sis_block(R, C, Pi, cov_y, idx_grp, co_obs):

    p,n = C.shape
    g = np.zeros(p)

    if not cov_y is None:
        for i in range(len(idx_grp)):
            a,b = idx_grp[i], co_obs[i]
            ab = np.intersect1d(a,b)

            PiC = Pi.dot(C[ab,:].T)
            for s in range(ab.size):
                g[ab[s]] = R[ab[s]]+C[ab[s],:].dot(PiC[:,s])-cov_y[ab[s],ab[s]]
    R -= 0.01 * g

    return R

def s_R_l2_Hankel_bad_sis(R, C, Pi, cov_y, a):

    if not cov_y is None:
        R[a] = np.maximum(cov_y[a,a] - C[a,:].dot(Pi).dot(C[a,:].T), 0)

    return R

def s_R_l2_Hankel_bad_sis_block(R, C, Pi, cov_y, idx_grp, co_obs):

    if not cov_y is None:
        for i in range(len(idx_grp)):
            a,b = idx_grp[i], co_obs[i]
            ab = np.intersect1d(a,b)

            PiC = Pi.dot(C[ab,:].T)
            for s in range(ab.size):
                R[ab[s]] = cov_y[ab[s], ab[s]] - C[ab[s],:].dot(PiC[:,s])
            R[ab] = np.maximum(R[ab], 0)

    return R

def id_R(R, C, Pi, cov_y, a):

     return R



def mat(X):
    return cvxopt.matrix(X, tc='d')

def s_A_l2_Hankel_bad_sis(C,R,k,l,Qs,idx_grp,co_obs, 
                            linear=True, linearise_X=False, stable=False,
                            A_old=None,verbose=False):
    "returns l2 Hankel reconstr. solution for A given C and the covariances Qs"

    # sis: subsampled/sparse in space
    
    p,n = C.shape

    if p > 1000:
        print('starting extraction of A')
    
    X = s_X_l2_Hankel_fully_obs(C, R, Qs, k, l, idx_grp, co_obs)

    A, X1 = A_old, None


    if linear:
        cvxopt.solvers.options['show_progress'] = False
        X1,X2 = np.zeros((n, n*(k+l-1))), np.zeros((n, n*(k+l-1)))
        for m in range(k+l-1):
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
            
            sol =cvxopt.solvers.qp(P=P,q=q,G=mat(G),h=mat(h),initvals=initvals)

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
            print('MSE reconstruction for A^m X1 - X2', np.mean((AX1 - X2)**2))
            print('MSE X1', np.mean(X1**2))
            print('MSE X2', np.mean(X2**2))

        #print(lam0)
        if G.shape[0] >= 1000:
            print('Warning! CG failed to guarantee stable A within max iter')

    return A, X, X1

def linearise_latent_covs(A, X, Pi=None, X1=None, linearity='True'):

    n = int(np.sqrt(X.shape[0]))
    klm1 = X.shape[1]

    if linearity=='True':
        X[:,0] = Pi.reshape(-1,)
        for m in range(1,klm1): # leave X[:,0] = cov(x_{t+1},x_t) unchanged!
            X[:,m] = np.linalg.matrix_power(A,m).dot(Pi).reshape(-1,)


    elif linearity=='first_order':
        if X1 is None:
            X1 = np.zeros((n, n*(klm1-1)))
            for m in range(klm1-1):
                X1[:,m*n:(m+1)*n] = X[:,m].reshape(n,n)

        AX1 = A.dot(X1)
        X[:,0] = X1[:,:n].reshape(-1,)
        for m in range(1,klm1): # leave X[:,0] = cov(x_{t+1},x_t) unchanged!
            X[:,m] = AX1[:,(m-1)*n:m*n].reshape(-1,)

    elif linearity=='False':
        pass 
    return X

def id_A(C,idx_grp,co_obs,A):

    return A



def s_X_l2_Hankel_vec(C, R, Qs, k, l, idx_grp, co_obs):
    "solves min || C X C.T - Q || for X, using a naive vec() approach"

    p,n = C.shape

    M = np.zeros((n**2, n**2))
    c = np.zeros((n**2, k+l))
    for i in range(len(idx_grp)):
        a,b = idx_grp[i], co_obs[i]
        if not Qs[0] is None:
            idx_R = np.where(np.in1d(a,b))[0]

        M += np.kron(C[b,:].T.dot(C[b,:]), C[a,:].T.dot(C[a,:]))

        if a.size * b.size * n**2 > 10e6: # size of variable Mab (see below)
            for s in range(b.size):
                Mab = np.kron(C[b[s],:], C[a,:]).T
                if not Qs[0] is None:
                    tmpQ = Qs[0][a,b[s]].copy()
                    tmpQ[idx_R[s]] -= R[b[s]]
                    c[:,0] += Mab.dot(tmpQ)
                for m_ in range(1,k+l):
                    c[:,m_] += Mab.dot(Qs[m_][a,b[s]])
        else:  # switch to single row of cov mats (size < p * n^2)
            Mab = np.kron(C[b,:], C[a,:]).T
            if not Qs[0] is None:
                tmpQ = Qs[0][np.ix_(a,b)].copy()
                tmpQ[idx_R, np.arange(b.size)] -= R[b]
                c[:,0] += Mab.dot(tmpQ.T.reshape(-1,))
            for m_ in range(1,k+l):
                c[:,m_] +=  Mab.dot(Qs[m_][np.ix_(a,b)].T.reshape(-1,)) 

    X = np.linalg.solve(M,c)
    for m in range(k+l):
        X[:,m] = (X[:,m].reshape(n,n).T).reshape(-1,)

    return X

def s_X_l2_Hankel_fully_obs(C, R, Qs, k, l, idx_grp, co_obs, max_size=1000):
    "solves min || C X C.T - Q || for X in the fully observed case."

    assert len(idx_grp) == 1

    p,n = C.shape
    Cd = np.linalg.pinv(C)

    X = np.zeros((n**2, k+l))
    p_range = np.arange(p)
    if not Qs[0] is None:
        if p > 1000:
            print('extracting latent cov. matrix for time-lag m=', 0)

        def f(idx_i, idx_j, i, j):
            tmpQ = Qs[0][np.ix_(idx_i,idx_j)].copy()
            if i == j: # indicates that idx_i == idx_j
                tmpQ[np.diag_indices(len(idx_i))] -= R[idx_i]
            return (Cd[:,idx_i].dot(tmpQ).dot(Cd[:,idx_j].T)).reshape(-1,)

        X[:,0] += chunking_blocks(f, p_range, p_range, max_size)

    for m_ in range(1,k+l):
        if p > 1000:
            print('extracting latent cov. matrix for time-lag m=', m_)

        def f(idx_i, idx_j, i, j):
            return (Cd[:,idx_i].dot(Qs[m_][np.ix_(idx_i,idx_j)]).dot( \
                Cd[:,idx_j].T)).reshape(-1,)

        X[:,m_] += chunking_blocks(f, p_range, p_range, max_size)            

    return X



def s_Pi_l2_Hankel_bad_sis(X,A,k,l,Qs,Pi=None,verbose=False, sym_psd=True):    

    # requires solution of semidefinite least-squares problem 
    # (no solver known for Python):
    # minimize || [A;A^2;A^3] * Pi - [X1; X2; X3] ||
    # subt. to   Pi is symmetric and psd.

    n = A.shape[1]

    As = np.empty((n*(k+l),n))
    XT = np.empty((n*(k+l),n))
    for m in range(k+l):
        XT[m*n:(m+1)*n,:] = X[:,m].reshape(n,n)
        As[m*n:(m+1)*n,:] = np.linalg.matrix_power(A,m)
        
    if sym_psd:
        Pi,_,norm_res,muv,r,fail = SDLS.sdls(A=As,B=XT,X0=Pi,Y0=None,
                                             tol=1e-8,verbose=False)
    else:
        print('ls Pie')
        Pi = np.linalg.lstsq(a=As,b=XT)[0]

    if verbose:
        print('MSE reconstruction:  A^m Pi - X_m',np.mean((As.dot(Pi)-XT)**2))

    #assert not fail

    return Pi

def id_Pi(X,A,idx_grp,co_obs,Pi=None):

    return Pi


###########################################################################
# utility, semi-scripts, plotting
###########################################################################


def track_correlations(Qs_full, p, n, Om, C, A, Pi, X, 
    linearity='False'):

    corrs = np.nan * np.ones(2)

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

    return corrs

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
    if np.max(Ts) != np.inf:
        Ts = np.sort(Ts)
    else: 
        Ts = np.hstack((np.inf, np.sort(Ts[np.isfinite(Ts)])))

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
                                       Qs_full, Om, traces=None,
                                       linearity = 'True', idx_grp = None, co_obs = None, 
                                       if_flip = False, m = 1):


    p,n = pars_true['C'].shape

    def plot_mats(thresh=500):
        return p * max((k,l)) <= thresh

    pars_init = set_none_mats(pars_init, p, n)

    if linearity == 'True':
        def f(C,A,Pi,R):                        
            return f_l2_Hankel_ln(C,A,Pi,R,k,l,Qs,idx_grp,co_obs)
        print('final squared error on observed parts:', 
            f(pars_est['C'],pars_est['A'],pars_est['Pi'],pars_est['R'])) 
    else:
        def f(C,X,R):
            return f_l2_Hankel_nl(C,X,R,k,l,Qs,idx_grp,co_obs)
        print('final squared error on observed parts:', 
            f(pars_est['C'],pars_est['X'],pars_est['R'])) 

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
            None,k,l,linear=False)

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
        plt.show()
    
    print('\n observed covariance entries')
    H_true = yy_Hankel_cov_mat_Qs(Qs_full,np.arange(p),k,l,n,Om=Om)
    if linearity=='True':
        H_est = yy_Hankel_cov_mat(pars_est['C'],pars_est['A'],pars_est['Pi'],
            k,l,Om=Om,linear=True)
    else:
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
    if m == 0:
        H_true += np.diag(pars_true['R'])

    if linearity=='True':
        H_est = pars_est['C'].dot( np.linalg.matrix_power(pars_est['A'],m).dot(pars_est['Pi']) ).dot(pars_est['C'].T)
        if m == 0:
            H_est += np.diag(pars_est['R'])
    else:
        H_est = pars_est['C'].dot(X[:,m].reshape(n,n).dot(pars_est['C'].T)) 
        if m == 0:
            H_est += np.diag(pars_est['R'])
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
        if m == 0:
            plt.hold(True)
            plt.plot(np.diag(H_true), np.diag(H_est), 'r.')
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

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

def run_bad(lag_range,n,y,Qs,
            Om,sub_pops,idx_grp,co_obs,obs_idx,
            obs_pops=None, obs_time=None,
            idx_a=None,idx_b=None,
            linearity='False',stable=False,init='SSID',
            alpha=0.001, alpha_R = None, b1=0.9, b2=0.99, e=1e-8, 
            max_iter=100, max_zip_size=np.inf, batch_size=1,
            verbose=False, sym_psd=True, mmap=False, data_path=None):

    T,p = y.shape 
    kl = len(lag_range)
    assert np.all(lag_range == np.sort(lag_range))

    alpha_R = alpha if alpha_R is None else alpha_R

    if isinstance(init, dict):
        assert 'C' in init.keys()
        pars_init = init.copy()

    if init =='default':
        pars_init = {'A'  : np.diag(np.linspace(0.89, 0.91, n)),
             'Pi' : np.eye(n),
             'B'  : np.eye(n), 
             'C'  : np.random.normal(size=(p,n)),
             'R'  : np.zeros(p),
             'X'  : np.zeros(((kl)*n, n))} #pars_ssid['C'].dot(np.linalg.inv(M))}   

    f_i,g_C,g_X,g_R,s_R,batch_draw,track_corrs = l2_bad_sis_setup(
                                           lag_range=lag_range,n=n,T=T,
                                           y=y,Qs=Qs,Om=Om,
                                           idx_a=idx_a, idx_b=idx_b,
                                           idx_grp=idx_grp,obs_idx=obs_idx,
                                           obs_pops=obs_pops, obs_time=obs_time,
                                           sub_pops=sub_pops,
                                           linearity=linearity,stable=stable,
                                           verbose=verbose,sym_psd=sym_psd,
                                           batch_size=batch_size,
                                           mmap=mmap, data_path=data_path)
    print('starting descent')    
    def converged(theta_old, theta, e, t):
        return True if t >= max_iter else False
    pars_est, traces = adam_zip_bad_stable(f=f_i,g_C=g_C,g_X=g_X,
                                        g_R=g_R,s_R=s_R, batch_draw=batch_draw,
                                        pars_0=pars_init,linearity=linearity,
                                        alpha=alpha,b1=b1,b2=b2,e=e,alpha_R=alpha_R,
                                        max_zip_size=max_zip_size,
                                        max_iter=max_iter,converged=converged,
                                        Om=Om,idx_grp=idx_grp,co_obs=co_obs,
                                        batch_size=batch_size,
                                        track_corrs=track_corrs)

    return pars_init, pars_est, traces



# decorations

def l2_bad_sis_setup(lag_range,T,n,y,Qs,Om,idx_grp,obs_idx,obs_pops=None, obs_time=None,
                     sub_pops=None,idx_a=None, idx_b=None, linearity='True', 
                     stable=False, sym_psd=True, W=None, verbose=False, 
                     batch_size=None, mmap=False, data_path=None):
    "returns error function and gradient for use with gradient descent solvers"

    T,p = y.shape
    kl = len(lag_range)

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

    if obs_time is None or obs_pops is None or sub_pops is None:
        def get_observed(p, t):
            return range(p) 
    else:
        def get_observed(p, t):
            i = obs_pops[np.digitize(t, obs_time)]
            return sub_pops[i]


    def g_C(C, X, R, ts, ms):
        return g_C_l2_Hankel_bad_sit(C,X,R,y,lag_range,ts,ms,
            get_observed=get_observed,linear=False,W=None)

    def g_X(C, X, R, ts, ms):
        return g_X_l2_Hankel_fully_obs(C,X,R,y,lag_range,ts,ms,
            get_observed=get_observed)

    def g_R(C, X0, R, ts):
        return g_R_l2_Hankel_bad_sis_block(C, X0, R, y, ts,
            get_observed=get_observed)

    def s_R(R,C,Pi):
        return s_R_l2_Hankel_bad_sis_block(R,C,Pi,Qs[0], idx_grp, co_obs)

    if linearity == 'True':
        def f(C,A,Pi,R):
            return f_l2_Hankel_ln(C,A,Pi,R,lag_range,Qs,idx_grp,co_obs,idx_a,idx_b)
    else:
        def f(C,X,R):
            return f_l2_Hankel_nl(C,X,None,R,lag_range,Qs,idx_grp,co_obs,idx_a,idx_b)

    def track_corrs(C, A, Pi, X, R) :
         return track_correlations(Qs, p, n, lag_range, Om, C, A, Pi, X, R,  
                        idx_a, idx_b, 'False', mmap, data_path)


    # setting up the stochastic batch selection:
    batch_draw, g_sit_C, g_sit_X,g_sit_R = l2_sis_draw(p, T, lag_range, batch_size, 
                                            idx_grp, co_obs, 
                                            g_C=g_C, g_X=g_X, g_R=g_R, Om=Om)


    return f,g_sit_C,g_sit_X,g_sit_R,s_R,batch_draw,track_corrs

def l2_sis_draw(p, T, lag_range, batch_size, idx_grp, co_obs, g_C, g_X, g_R, Om=None):
    "returns sequence of indices for sets of neuron pairs for SGD"

    kl = len(lag_range)
    kl_ = np.max(lag_range)+1
    if batch_size is None:

        def batch_draw():
            ts = (np.random.permutation(np.arange(T - (kl_) )) , )
            ms = (np.random.randint(0, kl, size= T - (kl_) ) , )   
            return ts, ms
        def g_sis_C(C, X, R, ts, ms, i):
            return g_C(C, X, R, ts[i], ms[i])
        def g_sis_X(C, X, R, ts, ms, i):
            return g_X(C, X, R, ts[i], ms[i])
        def g_sis_R(C, X0, R, ts, i):
            return g_R(C, X0, R, ts[i])


    elif batch_size == 1:

        def batch_draw():
            ts = np.random.permutation(np.arange(T - (kl_) ))
            ms = np.random.randint(0, kl, size=(len(ts),))         
            return ts, ms
        def g_sis_C(C, X, R, ts, ms, i):
            return g_C(C, X, R, (ts[i],), (ms[i],))
        def g_sis_X(C, X, R, ts, ms, i):
            return g_X(C, X, R, (ts[i],), (ms[i],))
        def g_sis_R(C, X0, R, ts, i):
            return g_R(C, X0, R, (ts[i],))


    elif batch_size > 1:

        def batch_draw():
            ts = np.random.permutation(np.arange(T - (kl_) ))
            ms = np.random.randint(0, kl, size=(len(ts),))         
            return ([ts[i*batch_size:(i+1)*batch_size] for i in range(len(ts)//batch_size)], 
                    [ms[i*batch_size:(i+1)*batch_size] for i in range(len(ms)//batch_size)])
        def g_sis_C(C, X, R, ts, ms, i):
            return g_C(C, X, R, ts[i], ms[i])
        def g_sis_X(C, X, R, ts, ms, i):
            return g_X(C, X, R, ts[i], ms[i])
        def g_sis_R(C, X, R, ts, i):
            return g_R(C, X, R, ts[i])

    return batch_draw, g_sis_C, g_sis_X, g_sis_R



# main optimiser

def adam_zip_bad_stable(f,g_C,g_X,g_R,s_R,batch_draw,track_corrs,pars_0,
                alpha,b1,b2,e,max_iter,alpha_R,
                converged,batch_size,max_zip_size,
                Om,idx_grp,co_obs,linearity='False'):


    # initialise pars
    p, n = pars_0['C'].shape
    kl   = pars_0['X'].shape[0]//n
    C,X,R = set_adam_init(pars_0, p, n, kl)
    A, Pi = None, None

    # setting up Adam
    b1,b2,e,v_0 = set_adam_pars(batch_size,p,n,kl,b1,b2,e)
    t_iter, t, ct_iter = 0, 0, 0 
    corrs  = np.zeros((kl, 12))
    m, v = np.zeros((p+kl*n,n)), v_0.copy()
    mR, vR = np.zeros(p), np.zeros(p)

    def zip_range(zip_size):
        if p > 1000:
            return progprint_xrange(zip_size, perline=100)
        else:
            return range(zip_size)

    # trace function values
    fun = np.empty(max_iter)    

    corrs[:,ct_iter] = track_corrs(C, A, Pi, X, R) 
    ct_iter += 1

    C_old = np.inf * np.ones((p,n))
    while not converged(C_old, C, e, t_iter):

        # updating C: full SGD pass over data
        C_old = C.copy()
        ts, ms = batch_draw()        
        zip_size = get_zip_size(batch_size, p, ts, max_zip_size)
        for idx_zip in zip_range(zip_size):
            t += 1

            # get data point(s) and corresponding gradients: 

            grad = np.vstack((g_C(C,X,R,ts,ms,idx_zip),
                              g_X(C,X,R,ts,ms,idx_zip)))

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

            C -= alpha * mh[:p,:]/(np.sqrt(vh[:p,:]) + e)
            X -= alpha * mh[p:,:]/(np.sqrt(vh[p:,:]) + e)

            gradR = g_R(C, X[:n, :], R, ts, idx_zip)
            mR = (b1 * mR + (1-b1)* gradR)     
            vR = (b2 * vR + (1-b2)*(gradR**2)) 
            if b1 != 1.:                    # delete those eventually 
                mhR = mR / (1-b1**t)
            else:
                mhR = mR
            if b2 != 1.:
                vhR = vR / (1-b2**t)
            else:
                vhR = vR
            R -= alpha_R * mhR/(np.sqrt(vhR) + e)


        if t_iter < max_iter:          # really expensive!
            if linearity=='True':
                fun[t_iter] = f(C,A,Pi,R) 
            else:
                fun[t_iter] = f(C,X,R)

        if np.mod(t_iter,max_iter//10) == 0:
            print('finished %', 100*t_iter/max_iter+10)
            print('f = ', fun[t_iter])
            corrs[:,ct_iter] = track_corrs(C, A, Pi, X, R) 
            ct_iter += 1

        t_iter += 1

    # final round over R (we before only updated R_ii just-in-time)
    #R = s_R(R, C, X[:n,:].reshape(n,n))

    corrs[:,ct_iter] = track_corrs(C, A, Pi, X, R)

    print('total iterations: ', t)

    pars_out = {'C' : C, 'X': X, 'R' : R }
    if linearity=='True': 
        pars_out['A'], pars_out['Pi'] = A, Pi
    elif linearity=='first_order':
        pars_out['A'], pars_out['Pi'] = A, np.nan * np.ones((n,n))
    if linearity=='False': 
        pars_out['A'], pars_out['Pi'] = np.nan*np.ones((n,n)), np.nan*np.ones((n,n))

    return pars_out, (fun,corrs)    



# setup

def set_adam_pars(batch_size,p,n,kl,b1,b2,e):

    if batch_size is None:
        print('doing batch gradients - switching to plain gradient descent')
        b1, b2, e, v_0 = 0, 1.0, 0, np.ones((p+kl*n,n))
    elif batch_size >= 1:
        print('subsampling in time!')
        v_0 = np.zeros((p+kl*n,n))        
    else: 
        raise Exception('cannot handle selected batch size')

    return b1,b2,e,v_0

def set_adam_init(pars_0, p, n, kl):

    C = pars_0['C'].copy()
    R = pars_0['R'].copy() if 'R' in pars_0.keys() else np.zeros(p)
    X = pars_0['X'].copy() if 'X' in pars_0.keys() else np.zeros(n*kl, n)

    return C,X,R

def get_zip_size(batch_size, p=None, a=None, max_zip_size=np.inf):

    if batch_size is None:
        zip_size = len(a)
    elif batch_size >= 1:
        zip_size = len(a)
    
    return int(np.min((zip_size, max_zip_size)))      



# gradients (g_*) & solvers (s_*) for model parameters

def g_C_l2_Hankel_bad_sit(C,X,R,y,lag_range,ts,ms,get_observed,linear=False, W=None):
    "returns l2 Hankel reconstr. stochastic gradient w.r.t. C"

    p,n = C.shape
    grad = np.zeros((p,n))

    assert len(ts) == len(ms)

    for (t,m) in zip(ts, ms):
        m_ = lag_range[m]
        a = get_observed(p, t+m_)
        b = get_observed(p, t)
        #if len(b) > n:
        #    raise Exception(('Warning, code was written for |b| <= n, but provided |b| > n.'
        #                    'Outcomment this if annoyed'))
        g_C_l2_Hankel_vector_pair(grad, C, X[m*n:(m+1)*n, :], R, 
                                            a, b, y[t], y[t+m_])    
            
    return grad / len(ts)

def g_C_l2_Hankel_vector_pair(grad, C, Xm, R, a, b, yp, yf):

    CX  = C[b,:].dot(Xm)
    CXT = C[b,:].dot(Xm.T) # = CX if m == 0 ...

    grad[a,:] += C[a,:].dot(CX.T.dot(CX) + CXT.T.dot(CXT)) \
               - (np.outer(yp[a],yf[b].dot(CX)) + np.outer(yf[a], yp[b].dot(CXT)))
    
    if yf is yp: # i.e, if m = 0
        ab = np.intersect1d(a,b) # returns sorted(unique( .. )), might be overkill here
        grad[ab,:] += 2 * R[ab].reshape(-1,1) * C[ab,:].dot(Xm) # need better indexing here...



def g_X_l2_Hankel_fully_obs(C, X, R, y, lag_range, ts, ms, get_observed):
    "solves min || C X C.T - Q || for X in the fully observed case."

    p,n = C.shape
    grad = np.zeros(X.shape)

    assert len(ts) == len(ms)

    for (t,m) in zip(ts, ms):
        m_ = lag_range[m]
        a = get_observed(p, t+m_)
        b = get_observed(p, t)

        grad[m*n:(m+1)*n,:] += g_X_l2_vector_pair(C, X[m*n:(m+1)*n, :], R, 
                                        a, b, y[t], y[t+m_])

    return grad / len(ts)


def g_X_l2_vector_pair(C, Xm, R, a, b, yp, yf):

    # if |a|, |b| < n, we actually first want to compute CXC' rather than the C'C !
    CC_a = C[a,:].T.dot(C[a,:])
    CC_b = C[b,:].T.dot(C[b,:]) if not a is b else CC_a.copy()

    grad = CC_a.dot(Xm).dot(CC_b) - np.outer(yf[a].dot(C[a,:]), yp[b].dot(C[b,:]))

    return grad



def g_R_l2_Hankel_bad_sis_block(C, X0, R, y, ts, get_observed):

    p,n = C.shape
    grad = np.zeros(p)

    for t in ts:
        a = get_observed(p, t)
        XCT = X0.dot(C[a,:].T)
        y2 = y[t,a]**2

        for s in range(len(a)):
            grad[a[s]] += R[a[s]] + C[a[s],:].dot(XCT[:,s]) - y2[s]

    return grad / len(ts)

def s_R_l2_Hankel_bad_sis_block(R, C, Pi, cov_y, idx_grp, co_obs):

    if not cov_y is None:
        for i in range(len(idx_grp)):
            a,b = idx_grp[i], co_obs[i]
            ab = np.intersect1d(a,b)

            PiC = Pi.dot(C[ab,:].T)
            for s in range(len(ab)):
                R[ab[s]] = cov_y[ab[s], ab[s]] - C[ab[s],:].dot(PiC[:,s])
            R[ab] = np.maximum(R[ab], 0)

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
            X1[:,m*n:(m+1)*n] = X[m*n:(m+1)*n, :].reshape(n,n)
            X2[:,m*n:(m+1)*n] = X[(m+1)*n:(m+2)*n, :].reshape(n,n)
            
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
        X[:n, :] = Pi.reshape(-1,)
        for m in range(1,klm1): # leave X[:,0] = cov(x_{t+1},x_t) unchanged!
            X[m*n:(m+1)*n, :] = np.linalg.matrix_power(A,m).dot(Pi).reshape(-1,)


    elif linearity=='first_order':
        if X1 is None:
            X1 = np.zeros((n, n*(klm1-1)))
            for m in range(klm1-1):
                X1[:,m*n:(m+1)*n] = X[:,m].reshape(n,n)
        AX1 = A.dot(X1)
        X[:n, :] = X1[:,:n].reshape(-1,)
        for m in range(1,klm1): # leave X[:,0] = cov(x_{t+1},x_t) unchanged!
            X[m*n:(m+1)*n, :] = AX1[:,(m-1)*n:m*n].reshape(-1,)

    elif linearity=='False':
        pass 
    return X

def id_A(C,idx_grp,co_obs,A):

    return A



def s_X_l2_Hankel_vec(C, R, Qs, lag_range, idx_grp, co_obs):
    "solves min || C X C.T - Q || for X, using a naive vec() approach"

    p,n = C.shape
    kl = len(lag_range)

    M = np.zeros((n**2, n**2))
    c = np.zeros((n**2, kl))
    for i in range(len(idx_grp)):
        a,b = idx_grp[i], co_obs[i]
        if not Qs[0] is None:
            idx_R = np.where(np.in1d(a,b))[0]

        M += np.kron(C[b,:].T.dot(C[b,:]), C[a,:].T.dot(C[a,:]))

        if len(a) * len(b) * n**2 > 10e6: # size of variable Mab (see below)
            for s in range(len(b)):
                Mab = np.kron(C[b[s],:], C[a,:]).T
                if not Qs[0] is None:
                    tmpQ = Qs[0][a,b[s]].copy()
                    tmpQ[idx_R[s]] -= R[b[s]]
                    c[:,0] += Mab.dot(tmpQ)
                for m_ in range(1,kl):
                    c[:,m_] += Mab.dot(Qs[m_][a,b[s]])
        else:  # switch to single row of cov mats (size < p * n^2)
            Mab = np.kron(C[b,:], C[a,:]).T
            if not Qs[0] is None:
                tmpQ = Qs[0][np.ix_(a,b)].copy()
                tmpQ[idx_R, np.arange(len(b))] -= R[b]
                c[:,0] += Mab.dot(tmpQ.T.reshape(-1,))
            for m_ in range(1,kl):
                c[:,m_] +=  Mab.dot(Qs[m_][np.ix_(a,b)].T.reshape(-1,)) 

    X = np.linalg.solve(M,c)
    for m in range(kl):
        X[:,m] = (X[:,m].reshape(n,n).T).reshape(-1,)

    return X

def s_X_l2_Hankel_fully_obs(C, R, Qs, lag_range, idx_grp, co_obs, max_size=1000):
    "solves min || C X C.T - Q || for X in the fully observed case."

    assert len(idx_grp) == 1
    kl = len(lag_range)

    p,n = C.shape
    Cd = np.linalg.pinv(C)

    X = np.zeros((n**2, kl))
    p_range = np.arange(p)
    if not Qs[0] is None:
        if p > 1000:
            print('extracting latent cov. matrix for time-lag m=', 0)

        def f(idx_i, idx_j, i, j):
            tmpQ = Qs[0][np.ix_(idx_i,idx_j)].copy()
            if i == j: # indicates that idx_i == idx_j
                tmpQ[np.diag_indices(len(idx_i))] -= R[idx_i]
            return (Cd[:,idx_i].dot(tmpQ).dot(Cd[:,idx_j].T)).reshape(-1,)

        X[:,0] = Cd[:,idx_i].dot(tmpQ).dot(Cd[:,idx_j].T)

    for m_ in range(1,kl):
        if p > 1000:
            print('extracting latent cov. matrix for time-lag m=', m_)

        def f(idx_i, idx_j, i, j):
            return (Cd[:,idx_i].dot(Qs[m_][np.ix_(idx_i,idx_j)]).dot( \
                Cd[:,idx_j].T)).reshape(-1,)

        X[:,m_] += chunking_blocks(f, p_range, p_range, max_size)            

    return X



def s_Pi_l2_Hankel_bad_sis(X,A,lag_range,Qs,Pi=None,verbose=False, sym_psd=True):    

    # requires solution of semidefinite least-squares problem 
    # (no solver known for Python):
    # minimize || [A;A^2;A^3] * Pi - [X1; X2; X3] ||
    # subt. to   Pi is symmetric and psd.

    kl = len(lag_range)
    n = A.shape[1]

    As = np.empty((n*(kl),n))
    XT = np.empty((n*(kl),n))
    for m in range(kl):
        XT[m*n:(m+1)*n,:] = X[:,m].reshape(n,n)
        As[m*n:(m+1)*n,:] = np.linalg.matrix_power(A,lag_range[m])
        
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



# evaluation of target loss function

def f_blank(C,A,Pi,R,lag_range,Qs,idx_grp,co_obs,idx_a,idx_b):

    return 0.


def f_l2_Hankel_nl(C,X,Pi,R,lag_range,Qs,idx_grp,co_obs,idx_a=None,idx_b=None):
    "returns overall l2 Hankel reconstruction error"

    kl = len(lag_range)
    p,n = C.shape
    idx_a = np.arange(p) if idx_a is None else idx_a
    idx_b = idx_a if idx_b is None else idx_b
    assert (len(idx_a), len(idx_b)) == Qs[0].shape

    err = f_l2_inst(C,X[:n, :],R,Qs[0],idx_grp,co_obs,idx_a,idx_b)
    for m in range(1,kl):
        err += f_l2_block(C,X[m*n:(m+1)*n, :],Qs[m],idx_grp,co_obs,idx_a,idx_b)
            
    return err/(kl)

def f_l2_Hankel_ln(C,A,Pi,R,lag_range,Qs,idx_grp,co_obs,idx_a=None,idx_b=None):
    "returns overall l2 Hankel reconstruction error"

    kl = len(lag_range)
    p,n = C.shape
    idx_a = np.arange(p) if idx_a is None else idx_a
    idx_b = idx_a if idx_b is None else idx_b
    assert (len(idx_a), len(idx_b)) == Qs[0].shape

    err = f_l2_inst(C,Pi,R,Qs[0],idx_grp,co_obs,idx_a,idx_b)
    for m in range(1,kl):
        APi = np.linalg.matrix_power(A, m).dot(Pi)  
        err += f_l2_block(C,APi,Qs[m],idx_grp,co_obs,idx_a,idx_b)
            
    return err/(kl)    
    
def f_l2_block(C,AmPi,Q,idx_grp,co_obs,idx_a,idx_b):
    "Hankel reconstruction error on an individual Hankel block"

    err = 0.
    for i in range(len(idx_grp)):
        err_ab = 0.
        a = np.intersect1d(idx_grp[i],idx_a)
        b = np.intersect1d(co_obs[i], idx_b)
        a_Q = np.in1d(idx_a, idx_grp[i])
        b_Q = np.in1d(idx_b, co_obs[i])

        v = (C[a,:].dot(AmPi).dot(C[b,:].T) - Q[np.ix_(a_Q,b_Q)])
        v = v.reshape(-1,)

        err += v.dot(v)

    return err

def f_l2_inst(C,Pi,R,Q,idx_grp,co_obs,idx_a,idx_b):
    "reconstruction error on the instantaneous covariance"

    err = 0.
    if not Q is None:
        for i in range(len(idx_grp)):

            a = np.intersect1d(idx_grp[i],idx_a)
            b = np.intersect1d(co_obs[i], idx_b)
            a_Q = np.in1d(idx_a, idx_grp[i])
            b_Q = np.in1d(idx_b, co_obs[i])

            v = (C[a,:].dot(Pi).dot(C[b,:].T) - Q[np.ix_(a_Q,b_Q)])
            idx_R = np.where(np.in1d(a,b))[0]
            v[idx_R, np.arange(len(idx_R))] += R[a[idx_R]]
            v = v.reshape(-1,)

            err += v.dot(v)

    return err

###########################################################################
# utility, semi-scripts, plotting
###########################################################################


def track_correlations(Qs, p, n, lag_range, Om, C, A, Pi, X, R,
    idx_a=None, idx_b=None, linearity='False', mmap = False, data_path=None):

    kl = len(lag_range)
    corrs = np.nan * np.ones(kl)
    if not Qs is None:

        idx_a = np.arange(p) if idx_a is None else idx_a
        idx_b = idx_a if idx_b is None else idx_b
        idx_ab = np.intersect1d(idx_a, idx_b)
        idx_a_ab = np.where(np.in1d(idx_a, idx_ab))[0]
        idx_b_ab = np.where(np.in1d(idx_b, idx_ab))[0]

        assert (len(idx_a), len(idx_b)) == Qs[0].shape
        for m in range(kl):
            m_ = lag_range[m] 
            Qrec = C[idx_a,:].dot(X[m*n:(m+1)*n, :]).dot(C[idx_b,:].T) 
            if m_==0:
                Qrec[np.ix_(idx_a_ab, idx_b_ab)] += np.diag(R[idx_ab])
            
            if mmap:
                Q = np.memmap(data_path+'Qs_'+str(m_), dtype=np.float, mode='r', shape=(pa,pb))
            else:
                Q = Qs[m]
                
            corrs[m] = np.corrcoef( Qrec.reshape(-1), Q.reshape(-1) )[0,1]
            
            if mmap:
                del Q                

    return corrs

def plot_slim(Qs,lag_range,pars,idx_a,idx_b,traces,mmap,data_path):

    kl = len(lag_range)
    p,n = pars['C'].shape
    pa, pb = idx_a.size, idx_b.size
    idx_ab = np.intersect1d(idx_a, idx_b)
    idx_a_ab = np.where(np.in1d(idx_a, idx_ab))[0]
    idx_b_ab = np.where(np.in1d(idx_b, idx_ab))[0]
    plt.figure(figsize=(20,10*np.ceil( (kl)/2)))
    for m in range(kl):
        m_ = lag_range[m] 
        Qrec = pars['C'][idx_a,:].dot(pars['X'][m*n:(m+1)*n, :]).dot(pars['C'][idx_b,:].T) 
        if m_ == 0:
            Qrec[np.ix_(idx_a_ab, idx_b_ab)] += np.diag(pars['R'][idx_ab])
        plt.subplot(np.ceil( (kl)/2 ), 2, m+1, adjustable='box-forced')
        if mmap:
            Q = np.memmap(data_path+'Qs_'+str(m_), dtype=np.float, mode='r', shape=(pa,pb))
        else:
            Q = Qs[m]
        plt.plot(Q.reshape(-1), Qrec.reshape(-1), '.')
        plt.title( ('m = ' + str(m_) + ', corr = ' + 
        str(np.corrcoef( Qrec.reshape(-1), (Qs[m]).reshape(-1) )[0,1])))
        if mmap:
            del Q
        plt.xlabel('true covs')
        plt.ylabel('est. covs')
    plt.show()
    plt.figure(figsize=(20,10))
    plt.plot(traces[0])
    plt.xlabel('iteration count')
    plt.ylabel('target loss')
    plt.title('loss function vs. iterations')
    plt.show()


def print_slim(Qs,lag_range,pars,idx_a,idx_b,traces,mmap,data_path):

    kl = len(lag_range)
    p,n = pars['C'].shape
    pa, pb = idx_a.size, idx_b.size
    idx_ab = np.intersect1d(idx_a, idx_b)
    idx_a_ab = np.where(np.in1d(idx_a, idx_ab))[0]
    idx_b_ab = np.where(np.in1d(idx_b, idx_ab))[0]
    for m in range(kl): 
        m_ = lag_range[m] 
        Qrec = pars['C'][idx_a,:].dot(pars['X'][m*n:(m+1)*n, :]).dot(pars['C'][idx_b,:].T) 
        if m_ == 0:
            Qrec[np.ix_(idx_a_ab, idx_b_ab)] += np.diag(pars['R'][idx_ab])
        if mmap:
            Q = np.memmap(data_path+'Qs_'+str(m_), dtype=np.float, mode='r', shape=(pa,pb))
        else:
            Q = Qs[m]
        print('m = ' + str(m_) + ', corr = ' + 
        str(np.corrcoef( Qrec.reshape(-1), (Qs[m]).reshape(-1) )[0,1]))
        if mmap:
            del Q

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

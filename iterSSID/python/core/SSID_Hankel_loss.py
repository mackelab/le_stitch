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

def run_bad(k,l,n,y,Qs,
            Om,sub_pops,idx_grp,co_obs,obs_idx,idx_a=None,idx_b=None,
            linearity='False',stable=False,init='SSID',
            alpha=0.001, b1=0.9, b2=0.99, e=1e-8, 
            max_iter=100, max_zip_size=np.inf, batch_size=1,
            verbose=False, sym_psd=True, mmap=False, data_path=None):

    T,p = y.shape 

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
             'R'  : 10e-5 * np.ones(p),
             'X'  : np.zeros(((k+l)*n, n))} #pars_ssid['C'].dot(np.linalg.inv(M))}   

    elif init =='default':
        pars_init = {'A'  : np.diag(np.linspace(0.89, 0.91, n)),
             'Pi' : np.eye(n),
             'B'  : np.eye(n), 
             'C'  : np.random.normal(size=(p,n)),
             'R'  : np.zeros(p),
             'X'  : np.zeros(((k+l)*n, n))} #pars_ssid['C'].dot(np.linalg.inv(M))}   

    f_i,g_C,g_X,g_R,s_R,batch_draw,track_corrs = l2_bad_sis_setup(
                                           k=k,l=l,n=n,T=T,
                                           y=y,Qs=Qs,Om=Om,
                                           idx_a=idx_a, idx_b=idx_b,
                                           idx_grp=idx_grp,obs_idx=obs_idx,
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
                                        alpha=alpha,b1=b1,b2=b2,e=e,
                                        max_zip_size=max_zip_size,
                                        max_iter=max_iter,converged=converged,
                                        Om=Om,idx_grp=idx_grp,co_obs=co_obs,
                                        batch_size=batch_size,
                                        track_corrs=track_corrs)

    return pars_init, pars_est, traces



# decorations

def l2_bad_sis_setup(k,l,T,n,y,Qs,Om,idx_grp,obs_idx,idx_a=None, idx_b=None,
                        linearity='True', stable=False, sym_psd=True, W=None,
                        verbose=False, batch_size=None, 
                        mmap=False, data_path=None):
    "returns error function and gradient for use with gradient descent solvers"

    T,p = y.shape

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

    def get_observed(p, t):
        return range(p) # just sad right now         

    def g_C(C, X, R, ts, ms):
        return g_C_l2_Hankel_bad_sit(C,X,R,y,ts,ms,
            get_observed=get_observed,linear=False,W=None)

    def g_X(C, X, R, ts, ms):
        return g_X_l2_Hankel_fully_obs(C,X,R,y,ts,ms,
            get_observed=get_observed)

    def g_R(C, X0, R, ts, ms):
        return g_R_l2_Hankel_bad_sis_block(C, X0, R, y, ts, ms,
            get_observed=get_observed)

    def s_R(R,C,Pi):
        return s_R_l2_Hankel_bad_sis_block(R,C,Pi,Qs[0], idx_grp, co_obs)

    if linearity == 'True':
        def f(C,A,Pi,R):
            return f_l2_Hankel_ln(C,A,Pi,R,k,l,Qs,idx_grp,co_obs,idx_a,idx_b)
    else:
        def f(C,X,R):
            return f_l2_Hankel_nl(C,X,None,R,k,l,Qs,idx_grp,co_obs,idx_a,idx_b)

    def track_corrs(C, A, Pi, X, R) :
         return track_correlations(Qs, p, n, k, l, Om, C, A, Pi, X, R,  
                        idx_a, idx_b, 'False', mmap, data_path)


    # setting up the stochastic batch selection:
    batch_draw, g_sit_C, g_sit_X,g_sit_R = l2_sis_draw(p, T, k, l, batch_size, 
                                            idx_grp, co_obs, 
                                            g_C=g_C, g_X=g_X, g_R=g_R, Om=Om)


    return f,g_sit_C,g_sit_X,g_sit_R,s_R,batch_draw,track_corrs

def l2_sis_draw(p, T, k, l, batch_size, idx_grp, co_obs, g_C, g_X, g_R, Om=None):
    "returns sequence of indices for sets of neuron pairs for SGD"

    if batch_size is None:

        def batch_draw():
            ts = (np.random.permutation(np.arange(T - (k+l) )) , )
            ms = (np.random.randint(0, k+l, size= T - (k+l) ) , )   
            return ts, ms
        def g_sis_C(C, X, R, ts, ms, i):
            return g_C(C, X, R, ts[i], ms[i])
        def g_sis_X(C, X, R, ts, ms, i):
            return g_X(C, X, R, ts[i], ms[i])
        def g_sis_R(C, X0, R, ts, ms, i):
            return g_R(C, X0, R, ts[i], ms[i])


    elif batch_size == 1:

        def batch_draw():
            ts = np.random.permutation(np.arange(T - (k+l) ))
            ms = np.random.randint(0, k+l, size=(len(ts),))         
            return ts, ms
        def g_sis_C(C, X, R, ts, ms, i):
            return g_C(C, X, R, (ts[i],), (ms[i],))
        def g_sis_X(C, X, R, ts, ms, i):
            return g_X(C, X, R, (ts[i],), (ms[i],))
        def g_sis_R(C, X0, R, ts, ms, i):
            return g_R(C, X0, R, (ts[i],), (ms[i],))


    elif batch_size > 1:

        def batch_draw():
            ts = np.random.permutation(np.arange(T - (k+l) ))
            ms = np.random.randint(0, k+l, size=(len(ts),))         
            return ([ts[i*batch_size:(i+1)*batch_size] for i in range(len(ts)//batch_size)], 
                    [ms[i*batch_size:(i+1)*batch_size] for i in range(len(ms)//batch_size)])
        def g_sis_C(C, X, R, ts, ms, i):
            return g_C(C, X, R, ts[i], ms[i])
        def g_sis_X(C, X, R, ts, ms, i):
            return g_X(C, X, R, ts[i], ms[i])
        def g_sis_R(C, X, R, ts, ms, i):
            return g_R(C, X, R, ts[i], ms[i])

    return batch_draw, g_sis_C, g_sis_X, g_sis_R



# main optimiser

def adam_zip_bad_stable(f,g_C,g_X,g_R,s_R,batch_draw,track_corrs,pars_0,
                alpha,b1,b2,e,max_iter,
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

            gradR = g_R(C, X[:n, :], R, ts, ms, idx_zip)
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
            R -= alpha*10000 * mhR/(np.sqrt(vhR) + e)


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

def g_C_l2_Hankel_bad_sit(C,X,R,y,ts,ms,get_observed,linear=False, W=None):
    "returns l2 Hankel reconstr. stochastic gradient w.r.t. C"

    p,n = C.shape
    grad = np.zeros((p,n))

    assert len(ts) == len(ms)

    for (t,m) in zip(ts, ms):
        a = get_observed(p, t)
        b = get_observed(p, t+m)
        #if len(b) > n:
        #    raise Exception(('Warning, code was written for |b| <= n, but provided |b| > n.'
        #                    'Outcomment this if annoyed'))
        g_C_l2_Hankel_vector_pair(grad, C, X[m*n:(m+1)*n, :], R, 
                                            a, b, y[t], y[t+m])    
            
    return grad / len(ts)

def g_C_l2_Hankel_vector_pair(grad, C, Xm, R, a, b, yp, yf):

    CX  = C[b,:].dot(Xm)
    CXT = C[b,:].dot(Xm.T) # = CX if m == 0 ...

    grad[a,:] += C[a,:].dot(CX.T.dot(CX) + CXT.T.dot(CXT)) \
               - (np.outer(yp[a],yf[b].dot(CX)) + np.outer(yf[a], yp[b].dot(CXT)))
    
    if yf is yp: # i.e, if m = 0
        ab = np.intersect1d(a,b) # returns sorted(unique( .. )), might be overkill here
        grad[ab,:] += 2 * R[ab].reshape(-1,1) * C[ab,:].dot(Xm) # need better indexing here...



def g_X_l2_Hankel_fully_obs(C, X, R, y, ts, ms, get_observed):
    "solves min || C X C.T - Q || for X in the fully observed case."

    p,n = C.shape
    grad = np.zeros(X.shape)

    assert len(ts) == len(ms)

    for (t,m) in zip(ts, ms):
        a = get_observed(p, t+m)
        b = get_observed(p, t)

        grad[m*n:(m+1)*n,:] += g_X_l2_vector_pair(C, X[m*n:(m+1)*n, :], R, 
                                        a, b, y[t], y[t+m])

    return grad / len(ts)


def g_X_l2_vector_pair(C, Xm, R, a, b, yp, yf):

    # if |a|, |b| < n, we actually first want to compute CXC' rather than the C'C !
    CC_a = C[a,:].T.dot(C[a,:])
    CC_b = C[b,:].T.dot(C[b,:]) if not a is b else CC_a.copy()

    grad = CC_a.dot(Xm).dot(CC_b) - np.outer(yf[a].dot(C[a,:]), yp[b].dot(C[b,:]))

    return grad



def g_R_l2_Hankel_bad_sis_block(C, X0, R, y, ts, ms, get_observed):

    p,n = C.shape
    grad = np.zeros(p)

    assert len(ts) == len(ms)

    for (t,m) in zip(ts, ms):
        if m == 0:
            a = get_observed(p, t+m)
            b = get_observed(p, t)
            ab = np.intersect1d(a,b)

            XCT = X0.dot(C[ab,:].T)
            for s in range(len(ab)):
                grad[ab[s]] += R[ab[s]] + C[ab[s],:].dot(XCT[:,s]) - y[t,ab[s]]**2

    return grad / np.max( (1,np.sum(ts==0)) )

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

        if len(a) * len(b) * n**2 > 10e6: # size of variable Mab (see below)
            for s in range(len(b)):
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
                tmpQ[idx_R, np.arange(len(b))] -= R[b]
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

        X[:,0] = Cd[:,idx_i].dot(tmpQ).dot(Cd[:,idx_j].T)

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



# evaluation of target loss function

def f_blank(C,A,Pi,R,k,l,Qs,idx_grp,co_obs,idx_a,idx_b):

    return 0.


def f_l2_Hankel_nl(C,X,Pi,R,k,l,Qs,idx_grp,co_obs,idx_a=None,idx_b=None):
    "returns overall l2 Hankel reconstruction error"

    p,n = C.shape
    idx_a = np.arange(p) if idx_a is None else idx_a
    idx_b = idx_a if idx_b is None else idx_b
    assert (len(idx_a), len(idx_b)) == Qs[0].shape

    err = f_l2_inst(C,X[:n, :],R,Qs[0],idx_grp,co_obs,idx_a,idx_b)
    for m in range(1,k+l):
        err += f_l2_block(C,X[m*n:(m+1)*n, :],Qs[m],idx_grp,co_obs,idx_a,idx_b)
            
    return err/(k*l)

def f_l2_Hankel_ln(C,A,Pi,R,k,l,Qs,idx_grp,co_obs,idx_a=None,idx_b=None):
    "returns overall l2 Hankel reconstruction error"

    p,n = C.shape
    idx_a = np.arange(p) if idx_a is None else idx_a
    idx_b = idx_a if idx_b is None else idx_b
    assert (len(idx_a), len(idx_b)) == Qs[0].shape

    err = f_l2_inst(C,Pi,R,Qs[0],idx_grp,co_obs,idx_a,idx_b)
    for m in range(1,k+l):
        APi = np.linalg.matrix_power(A, m).dot(Pi)  
        err += f_l2_block(C,APi,Qs[m],idx_grp,co_obs,idx_a,idx_b)
            
    return err/(k*l)    
    
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


def track_correlations(Qs, p, n, k, l, Om, C, A, Pi, X, R,
    idx_a=None, idx_b=None, linearity='False', mmap = False, data_path=None):

    corrs = np.nan * np.ones(k+l)

    if not Qs is None:

        idx_a = np.arange(p) if idx_a is None else idx_a
        idx_b = idx_a if idx_b is None else idx_b
        assert (len(idx_a), len(idx_b)) == Qs[0].shape
        for m in range(0,k+l): 
            Qrec = C[idx_a,:].dot(X[m*n:(m+1)*n, :]).dot(C[idx_b,:].T) 
            Qrec = Qrec + np.diag(R)[np.ix_(idx_a,idx_b)] if m==0 else Qrec
            
            if mmap:
                Q = np.memmap(data_path+'Qs_'+str(m), dtype=np.float, mode='r', shape=(pa,pb))
            else:
                Q = Qs[m]
                
            corrs[m] = np.corrcoef( Qrec.reshape(-1), Q.reshape(-1) )[0,1]
            
            if mmap:
                del Q                

    return corrs


def plot_outputs_l2_gradient_test(pars_true, pars_init, pars_est, k, l, Qs, 
                                       Om, traces=None, idx_a=None, idx_b=None,
                                       linearity = 'True', idx_grp = None, co_obs = None, 
                                       if_flip = False, m = 1):

    print(m)
    p,n = pars_true['C'].shape

    def plot_mats(thresh=500):
        return p * max((k,l)) <= thresh

    pars_init = set_none_mats(pars_init, p, n)

    idx_a = np.arange(p) if idx_a is None else idx_a
    idx_b = idx_a if idx_b is None else idx_b
    if linearity == 'True':
        def f(C,A,Pi,R):     
            return f_l2_Hankel_ln(C,A,Pi,R,k,l,Qs,idx_grp,co_obs,idx_a,idx_b)
        print('final squared error on observed parts:', 
            f(pars_est['C'],pars_est['A'],pars_est['Pi'],pars_est['R'])) 
    else:
        def f(C,X,R):                 
            return f_l2_Hankel_nl(C,X,Pi,R,k,l,Qs,idx_grp,co_obs,idx_a,idx_b)
            print('final squared error on observed parts:', 
            f(pars_est['C'],pars_est['X'],pars_est['R'])) 


    if plot_mats():
        H_true = yy_Hankel_cov_mat_Qs(Qs,np.arange(p),k,l,n,Om=None)
        H_0    = yy_Hankel_cov_mat(pars_init['C'],pars_init['A'],pars_init['Pi'],k,l)

        H_obs = yy_Hankel_cov_mat_Qs(Qs,np.arange(p),k,l,n,Om= Om)
        H_obs[np.where(H_obs==0)] = np.nan
        H_sti = yy_Hankel_cov_mat_Qs(Qs,np.arange(p),k,l,n,Om=~Om)

        if linearity=='True':
            H_est = yy_Hankel_cov_mat(pars_est['C'],pars_est['A'],
                pars_est['Pi'],k,l)
        else:
            H_est = yy_Hankel_cov_mat(pars_est['C'],pars_est['X'],
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
    H_true = yy_Hankel_cov_mat_Qs(Qs,np.arange(p),k,l,n,Om=Om)
    if linearity=='True':
        H_est = yy_Hankel_cov_mat(pars_est['C'],pars_est['A'],pars_est['Pi'],
            k,l,Om=Om,linear=True)
    else:
        H_est  = yy_Hankel_cov_mat(pars_est['C'],pars_est['X'],None,k,l,Om=Om,linear=False)        
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
    H_true = yy_Hankel_cov_mat_Qs(Qs,np.arange(p),k,l,n,Om=~Om)
    if linearity=='True':
        H_est = yy_Hankel_cov_mat(pars_est['C'],pars_est['A'],pars_est['Pi'],
            k,l,Om=~Om,linear=True)
    else:
        H_est  = yy_Hankel_cov_mat(pars_est['C'], pars_est['X'], None,k,l,Om=~Om,linear=False)   
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
        H_est = pars_est['C'].dot(pars_est['X'][m*n:(m+1)*n, :].dot(pars_est['C'].T)) 
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
        if m == 0:
            plt.hold(True)
            plt.plot(np.diag(H_true), np.diag(H_est), 'r.')
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

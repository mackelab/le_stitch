import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

###########################################################################
# stochastic gradient descent: gradients w.r.t. C,R,A,B,X=cov(x_{t+m},x_t)
###########################################################################

def run_bad(lag_range,n,y,obs_scheme,
            Qs=None, Om=None,
            parametrization='nl',
            sso=False, W=None, 
            idx_a=None,idx_b=None,
            init='default',
            alpha=0.001, b1=0.9, b2=0.99, e=1e-8, 
            max_iter=100, max_epoch_size=np.inf, eps_conv=0.99999,
            batch_size=1,
            verbose=False, mmap=False, data_path=None, pars_track=None):

    assert parametrization in ['nl', 'ln']    
    num_pars = 3 if parametrization=='nl' else 4

    if W is None:
        W = obs_scheme.comp_coocurrence_weights(lag_range=lag_range, 
                                                sso=sso,
                                                idx_a=idx_a,
                                                idx_b=idx_b) 

    if Qs is None or Om is None:
        Qs, Om = f_l2_Hankel_comp_Q_Om(n=n,y=y,lag_range=lag_range,
                                obs_scheme=obs_scheme,idx_a=idx_a,idx_b=idx_b,W=W)


    idx_a = np.arange(p) if idx_a is None else idx_a
    idx_b = idx_a if idx_b is None else idx_b
    assert np.all(idx_a == np.sort(idx_a))
    assert np.all(idx_a == np.sort(idx_a))

    alpha = alpha * np.ones(num_pars) if np.asarray(alpha).size==1 else alpha
    b1    =  b1   * np.ones(num_pars) if np.asarray(  b1 ).size==1 else b1
    b2    =  b2   * np.ones(num_pars) if np.asarray(  b2 ).size==1 else b2
    e     =   e   * np.ones(num_pars) if np.asarray(  e  ).size==1 else e

    T,p = y.shape 
    kl = len(lag_range)
    assert np.all(lag_range == np.sort(lag_range))

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

    f,g,batch_draw,track_corrs,converged = l2_bad_sis_setup(
                                           lag_range=lag_range,n=n,T=T,
                                           parametrization=parametrization,
                                           sso=sso,
                                           y=y,Qs=Qs,Om=Om,
                                           idx_a=idx_a, idx_b=idx_b,
                                           obs_scheme=obs_scheme, W=W,
                                           batch_size=batch_size,
                                           mmap=mmap, data_path=data_path,
                                           max_iter=max_iter, eps_conv=eps_conv)
    if verbose:
        print('starting descent')    

    t_desc = time.time()
    pars_est, traces = adam_main(f=f,g=g,pars_0=pars_init,num_pars=num_pars,kl=kl,
                                 alpha=alpha,b1=b1,b2=b2,e=e,
                                 batch_draw=batch_draw,converged=converged,
                                 track_corrs=track_corrs,max_iter=max_iter,
                                 max_epoch_size=max_epoch_size,batch_size=batch_size,
                                 verbose=verbose,pars_track=pars_track)
    t_desc = time.time() - t_desc

    if parametrization=='nl':

        pars_est = {'C': pars_est[0], 
                    'X': pars_est[1], 
                    'R': pars_est[2], 
                    'A': None, 'B': None}

    elif parametrization=='ln':

        X, Pi = np.zeros((len(lag_range)*n, n)), pars_est[2].dot(pars_est[2].T)
        for m in lag_range:
            X[m*n:(m+1)*n,:] = np.linalg.matrix_power(pars_est[1],m).dot(Pi)   

        pars_est = {'C': pars_est[0], 
                    'A': pars_est[1], 
                    'B': pars_est[2], 
                    'R': pars_est[3], 
                    'Pi': Pi,
                    'X': X}


    return pars_init, pars_est, traces, Qs, Om, t_desc


# decorations

def l2_bad_sis_setup(lag_range,T,n,y,Qs,Om,obs_scheme,
                     idx_a=None, idx_b=None, W=None,
                     parametrization='nl', sso=False, 
                     max_iter=np.inf, eps_conv=0.99999,
                     batch_size=None, mmap=False, data_path=None):
    "returns error function and gradient for use with gradient descent solvers"

    sub_pops= obs_scheme.sub_pops
    idx_grp = obs_scheme.idx_grp
    obs_idx = obs_scheme.obs_idx
    obs_pops= obs_scheme.obs_pops
    obs_time= obs_scheme.obs_time

    T,p = y.shape
    kl = len(lag_range)
    kl_ = np.max(lag_range)+1

    if sso:
        g_l2_nl, g_l2_ln = g_l2_Hankel_sgd_nl_sso, g_l2_Hankel_sgd_ln_sso 
    else:
        g_l2_nl, g_l2_ln = g_l2_Hankel_sgd_nl, g_l2_Hankel_sgd_ln

    if parametrization=='nl':


        #def g(pars, ts, ms):
        #    return g_l2_Hankel_sgd_nl_sso(C=pars[0],X=pars[1],R=pars[2],y=y,
        #        lag_range=lag_range,ts=ts,ms=ms,obs_scheme=obs_scheme,W=W)

        def g(pars, ts, ms):
            return  g_l2_nl(C=pars[0],X=pars[1],R=pars[2],y=y,
                lag_range=lag_range,ts=ts,ms=ms,obs_scheme=obs_scheme,W=W)

        def f(pars):
            return f_l2_Hankel_nl(C=pars[0],X=pars[1],R=pars[2],Qs=Qs,Om=Om,
                lag_range=lag_range, ms=range(kl),idx_a=idx_a,idx_b=idx_b)

        def track_corrs(pars) :
            return track_correlations(C=pars[0],A=None,B=None,X=pars[1],R=pars[2],
                            Qs=Qs, p=p, n=n, lag_range=lag_range,
                            idx_a=idx_a, idx_b=idx_b, mmap=mmap, data_path=data_path)


    elif parametrization=='ln':

        def g(pars, ts, ms):
            return  g_l2_ln(C=pars[0],A=pars[1],B=pars[2],R=pars[3],y=y, 
                lag_range=lag_range,ts=ts,ms=ms,obs_scheme=obs_scheme,W=W)

        def f(pars):
            X, Pi = np.zeros((kl_*n,n)), pars[2].dot(pars[2].T)
            for m in lag_range:
                X[m*n:(m+1)*n,:] = np.linalg.matrix_power(pars[1],m).dot(Pi)                
            return f_l2_Hankel_nl(C=pars[0],X=X,R=pars[3],Qs=Qs,Om=Om,
                lag_range=lag_range, ms=range(kl),idx_a=idx_a,idx_b=idx_b)

        def track_corrs(pars) :
            return track_correlations(C=pars[0],A=pars[1],B=pars[2],X=None,R=pars[3],
                            Qs=Qs, p=p, n=n, lag_range=lag_range,
                            idx_a=idx_a, idx_b=idx_b, mmap=mmap, data_path=data_path)


    # setting up the stochastic batch selection:
    batch_draw, g_sgd = l2_sgd_draw(p, T, lag_range, batch_size, g)

    # set convergence criterion 
    if batch_size is None:
        #print('bach sizes, stopping if loss <'+str(eps_conv)+' previous loss')
        def converged(t, fun):
            if t>= max_iter:
                return True
            elif t > 99 and fun[t-1] > eps_conv * np.min(fun[t-100:t-1]):
                #print(fun[t-1]  / np.min(fun[t-4:t-1]))
                return True
            else:
                return False

    else:
        def converged(t, fun):
            return True if t >= max_iter else False

    return f,g_sgd,batch_draw,track_corrs,converged

def l2_sgd_draw(p, T, lag_range, batch_size, g):
    "returns sequence of indices for sets of neuron pairs for SGD"

    kl = len(lag_range)
    kl_ = np.max(lag_range)+1
    if batch_size is None:

        def batch_draw():
            ts = (np.random.permutation(np.arange(T - kl_)) , )
            ms = (lag_range, )   
            return ts, ms
        def g_sgd(pars, ts, ms, i):
            return g(pars, ts[i], ms[i])


    elif batch_size == 1:

        def batch_draw():
            ts = np.random.permutation(np.arange(T - kl_))
            ms = np.random.randint(0, kl, size=(len(ts),))         
            return ts, ms
        def g_sgd(pars, ts, ms, i):
            return g(pars, (ts[i],), (ms[i],))


    elif batch_size > 1:

        def batch_draw():
            ts = np.random.permutation(np.arange(T - (kl_) ))
            ms = np.random.randint(0,kl,size=len(ts)//batch_size) # one lag per gradient!
            return ([ts[i*batch_size:(i+1)*batch_size] for i in range(len(ts)//batch_size)],
                     ms)

        def g_sgd(pars, ts, ms, i):
            return g(pars, ts[i], (ms[i],))

    return batch_draw, g_sgd



# main optimiser

def adam_main(f,g,pars_0,num_pars,kl,alpha,b1,b2,e,
            batch_draw,track_corrs,converged,
            max_iter,batch_size,max_epoch_size,
            verbose,pars_track):

    # initialise pars
    p, n = pars_0['C'].shape
    pars = init_adam(pars_0, p, n, kl, num_pars)

    # setting up Adam
    b1,b2,e,vp,mp = pars_adam(batch_size,p,n,kl,num_pars,b1,b2,e)
    t_iter, t, ct_iter = 0, 0, 0 
    corrs  = np.zeros((kl, 12))

    def epoch_range(epoch_size):
        if p > 1000:
            return progprint_xrange(epoch_size, perline=100)
        else:
            return range(epoch_size)

    # trace function values
    fun = np.empty(max_iter)    

    corrs[:,ct_iter] = track_corrs(pars) 
    ct_iter += 1

    while not converged(t_iter, fun):

        ts, ms = batch_draw()        
        epoch_size = get_epoch_size(batch_size, p, ts, max_epoch_size)

        for idx_epoch in epoch_range(epoch_size):
            t += 1

            grads = g(pars, ts, ms, idx_epoch)
            pars, mp, vp = adam_step(pars,grads,mp,vp,alpha,b1,b2,e,t_iter+1)

        if t_iter < max_iter:          # really expensive!
            fun[t_iter] = f(pars)
            if not pars_track is None:
                pars_track(pars,t_iter)

        if np.mod(t_iter,max_iter//10) == 0:
            if verbose:
                print('finished %', 100*t_iter/max_iter+10)
                print('f = ', fun[t_iter])
            corrs[:,ct_iter] = track_corrs(pars) 
            ct_iter += 1

        t_iter += 1

    # final round over R (we before only updated R_ii just-in-time)
    #R = s_R(R, C, X[:n,:].reshape(n,n))

    corrs[:,ct_iter] = track_corrs(pars)
    fun = fun[:t_iter]

    if verbose:
        print('total iterations: ', t)

    return pars, (fun,corrs)    


def pars_adam(batch_size,p,n,kl,num_pars,b1,b2,e):

    if batch_size is None:
        print('using batch gradients - switching to plain gradient descent')
        b1, b2, e = np.zeros(num_pars), np.ones(num_pars), np.zeros(num_pars) 
    elif not batch_size >= 1: 
        raise Exception('cannot handle selected batch size')

    if num_pars == 3:     # (C, X, R)
        vp_0 = [np.ones((p,n)),  np.ones((kl*n,n)),  np.ones(p)]
        mp_0 = [np.zeros((p,n)), np.zeros((kl*n,n)), np.zeros(p)]
    elif num_pars == 4:   # (C, A, B, R)
        vp_0 = [np.ones((p,n)),  np.ones((n,n)),  np.ones((n,n)),  np.ones(p)]
        mp_0 = [np.zeros((p,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros(p)]

    return b1,b2,e,vp_0,mp_0


def adam_step(pars,g,m,v,a,b1,b2,e,t):
    for i in range(len(m)):

        m[i] = (b1[i] * m[i] + (1-b1[i]) * g[i])
        v[i] = (b2[i] * v[i] + (1-b2[i]) * g[i]**2)

        mh,vh = adam_norm(m[i],v[i],b1[i],b2[i],t)

        pars[i] -= a[i] * mh / (np.sqrt(vh) + e[i])

    return pars, m, v

def adam_norm(m, v, b1, b2,t):

    m = m / (1-b1**t) if b1 != 1 else m.copy()
    v = v / (1-b2**t) if b2 != 1 else v.copy()

    return m,v

def init_adam(pars_0, p, n, kl, num_pars):

    if num_pars == 3:
        C = pars_0['C'].copy()
        R = pars_0['R'].copy() if 'R' in pars_0.keys() else np.zeros(p)
        X = pars_0['X'].copy() if 'X' in pars_0.keys() else np.zeros((n*kl, n))
        pars = [C,X,R]
    elif num_pars == 4:
        C = pars_0['C'].copy()
        A = pars_0['A'].copy() if 'A' in pars_0.keys() else np.zeros((n, n))
        B = pars_0['B'].copy() if 'B' in pars_0.keys() else np.zeros((n, n))
        R = pars_0['R'].copy() if 'R' in pars_0.keys() else np.zeros(p)
        pars = [C,A,B,R]

    return pars

def get_epoch_size(batch_size, p=None, a=None, max_epoch_size=np.inf):

    if batch_size is None:
        epoch_size = len(a)
    elif batch_size >= 1:
        epoch_size = len(a)
    
    return int(np.min((epoch_size, max_epoch_size)))      

# gradients (g_*) & solvers (s_*) for model parameters

def g_l2_Hankel_sgd_nl(C,X,R,y,lag_range,ts,ms,obs_scheme,W):

    p,n = C.shape
    grad_C = np.zeros_like(C)
    grad_X = np.zeros_like(X)
    grad_R = np.zeros_like(R)

    get_observed = obs_scheme.gen_get_observed()

    for m in ms:
        m_ = lag_range[m]
        Xm = X[m*n:(m+1)*n, :]
        grad_Xm = grad_X[m*n:(m+1)*n, :]
        for t in ts:
            a = get_observed(t+m_)
            b = get_observed(t)
            anb = np.intersect1d(a,b)

            g_C_l2_vector_pair_rw(grad_C,  m_, C, Xm, R, a, b, anb, y[t], y[t+m_], W[m])
            g_X_l2_vector_pair_rw(grad_Xm, m_, C, Xm, R, a, b, anb, y[t], y[t+m_], W[m])

        if m_==0:
            g_R_l2_Hankel_sgd_rw(grad_R, C, Xm, R, y, ts, get_observed, W[m])

    return grad_C, grad_X, grad_R

def g_l2_Hankel_sgd_ln(C,A,B,R,y,lag_range,ts,ms,obs_scheme,W):

    p,n = C.shape
    kl_ = np.max(lag_range)+1 # d/dA often (but not always) needs all powers A^m

    get_observed = obs_scheme.gen_get_observed()

    grad_C = np.zeros_like(C)
    grad_A = np.zeros_like(A)
    grad_B = np.zeros_like(B)
    grad_R = np.zeros_like(R)

    Pi = B.dot(B.T)
    Aexpm = np.zeros((kl_*n,n))
    Aexpm[:n,:] = np.eye(n)
    for m in range(1,kl_):
        Aexpm[m*n:(m+1)*n,:] = A.dot(Aexpm[(m-1)*n:(m)*n,:])
    grad_X = np.zeros_like(Aexpm, dtype=A.dtype)

    for m in ms:

        m_ = lag_range[m]
        Xm = Aexpm[m*n:(m+1)*n,:].dot(Pi)

        grad_Bm  = np.zeros_like(B)
        grad_Xm = grad_X[m*n:(m+1)*n, :]

        for t in ts:
            a = get_observed(t+m_)
            b = get_observed(t)
            anb = np.intersect1d(a,b)

            g_C_l2_vector_pair_rw(grad_C,  m_, C, Xm, R, a, b, anb, y[t], y[t+m_], W[m])
            g_B_l2_vector_pair_rw(grad_Bm, m_, C, Aexpm[m*n:(m+1)*n,:], Pi, R, a, b, anb, y[t], y[t+m_], W[m])
            g_X_l2_vector_pair_rw(grad_Xm, m_, C, Xm, R, a, b, anb, y[t], y[t+m_], W[m])

        grad_B += (grad_Bm + grad_Bm.T)

        g_A_l2_block(grad_A, grad_Xm.dot(Pi), Aexpm,m) # grad_Xm.dot(Pi) possibly too costly

        if m_==0:
            g_R_l2_Hankel_sgd_rw(grad_R, C, Xm, R, y, ts, get_observed, W[m])

    grad_B = grad_B.dot(B)

    return grad_C, grad_A, grad_B, grad_R

def g_A_l2_block(grad, dXmPi, Aexpm, m):
    "returns l2 Hankel reconstr. gradient w.r.t. A for a single Hankel block"

    n = Aexpm.shape[1]

    for q in range(m):
        grad += Aexpm[q*n:(q+1)*n,:].T.dot(dXmPi.dot(Aexpm[(m-1-q)*n:(m-q)*n,:].T))

def g_C_l2_vector_pair_rw(grad, m_, C, Xm, R, a, b, anb, yp, yf, Wm):

    C___ = C.dot(Xm)   # mad-
    C_tr = C.dot(Xm.T) # ness        
        
    for k in a:
        WC = C_tr[b,:] * Wm[k,b].reshape(-1,1)
        grad[k,:] += C[k,:].dot( C_tr[b,:].T.dot(WC) ) 
        grad[k,:] -= yf[k] * yp[b].dot(WC)
        
    for k in b:
        WC = C___[a,:] * Wm[a,k].reshape(-1,1)
        grad[k,:] += C[k,:].dot( C___[a,:].T.dot(WC) ) 
        grad[k,:] -= yp[k] * yf[a].dot(WC)
        
    if m_ == 0:      
        grad[anb,:] += (R[anb]*Wm[anb,anb]).reshape(-1,1)*(C___[anb,:] + C_tr[anb,:])
           
def g_B_l2_vector_pair_rw(grad, m_, C, Am, Pi, R, a, b, anb, yp, yf, Wm):

    for k in a:        
        CAm_k = C[k,:].dot(Am)
        S_k = C[b,:].T.dot(C[b,:] * Wm[k,b].reshape(-1,1))
        grad += np.outer(CAm_k, CAm_k).dot(Pi).dot(S_k)
        S_k = yp[b].dot(C[b,:] * Wm[k,b].reshape(-1,1))
        grad -= np.outer(yf[k] * CAm_k, S_k)

    if m_ == 0:
        grad += C[anb,:].dot(Am).T.dot( (R[anb] * Wm[anb,anb]).reshape(-1,1)*C[anb,:]) 

def g_X_l2_vector_pair_rw(grad, m_, C, Xm, R, a, b, anb, yp, yf, Wm):

    for k in a:        
        S_k = C[b,:].T.dot(C[b,:] * Wm[k,b].reshape(-1,1))
        grad += np.outer(C[k,:], C[k,:]).dot(Xm).dot(S_k)
        
        S_k = yp[b].dot(C[b,:] * Wm[k,b].reshape(-1,1))
        grad -= np.outer(yf[k] * C[k,:], S_k)
        
    if m_ == 0:
        grad += C[anb,:].T.dot( (R[anb] * Wm[anb,anb]).reshape(-1,1)*C[anb,:]) 


def g_R_l2_Hankel_sgd_rw(grad, C, X0, R, y, ts, get_observed, W0):

    for t in ts:
        b = get_observed(t)         
        grad[b] += (R[b] + np.sum(C[b,:] * C[b,:].dot(X0.T),axis=1) - y[t,b]**2) * W0[b,b]






def g_l2_Hankel_sgd_ln_sso(C,A,B,R,y,lag_range,ts,ms,obs_scheme,W):

    p,n = C.shape
    kl_ = np.max(lag_range)+1 # d/dA often (but not always) needs all powers A^m

    get_idx_grp,idx_grp = obs_scheme.gen_get_idx_grp(), obs_scheme.idx_grp

    grad_C = np.zeros_like(C)
    grad_A = np.zeros_like(A)
    grad_B = np.zeros_like(B)
    grad_R = np.zeros_like(R)

    Pi = B.dot(B.T)
    Aexpm = np.zeros((kl_*n,n))
    Aexpm[:n,:] = np.eye(n)
    for m in range(1,kl_):
        Aexpm[m*n:(m+1)*n,:] = A.dot(Aexpm[(m-1)*n:(m)*n,:])
    grad_X = np.zeros_like(Aexpm, dtype=A.dtype)

    for m in ms:

        m_ = lag_range[m]
        Xm = Aexpm[m*n:(m+1)*n,:].dot(Pi)

        grad_Bm  = np.zeros_like(B)
        grad_Xm = grad_X[m*n:(m+1)*n, :]

        for t in ts:
            is_ = get_idx_grp(t+m_)
            js_ = get_idx_grp(t)
            inj = np.intersect1d(is_, js_)

            g_C_l2_vector_pair_sso(grad_C,  m_, C, Xm, R, idx_grp, is_, js_, inj, y[t], y[t+m_], W[m])
            g_B_l2_vector_pair_sso(grad_Bm, m_, C, Aexpm[m*n:(m+1)*n,:], Pi, R, idx_grp, is_, js_, inj, y[t], y[t+m_], W[m])
            g_X_l2_vector_pair_sso(grad_Xm, m_, C, Xm, R, idx_grp, is_, js_, inj, y[t], y[t+m_], W[m])
            if m_==0:
                g_R_l2_Hankel_sgd_sso(grad_R, C, Xm, R, idx_grp, inj, y[t], W[m])

        grad_B += (grad_Bm + grad_Bm.T)

        g_A_l2_block(grad_A, grad_Xm.dot(Pi), Aexpm,m) # grad_Xm.dot(Pi) possibly too costly

    grad_B = grad_B.dot(B)

    return grad_C, grad_A, grad_B, grad_R

def g_l2_Hankel_sgd_nl_sso(C,X,R,y,lag_range,ts,ms,obs_scheme,W):

    p,n = C.shape
    grad_C = np.zeros_like(C)
    grad_X = np.zeros_like(X)
    grad_R = np.zeros_like(R)

    get_idx_grp,idx_grp = obs_scheme.gen_get_idx_grp(), obs_scheme.idx_grp
    #CCs = [ C[idx_grp[i],:].T.dot(C[idx_grp[i],:]) for i in range(len(idx_grp)) ] 

    for m in ms:
        m_ = lag_range[m]
        Xm = X[m*n:(m+1)*n, :]
        grad_Xm = grad_X[m*n:(m+1)*n, :]
        for t in ts:
            is_ = get_idx_grp(t+m_)
            js_ = get_idx_grp(t)
            inj = np.intersect1d(is_, js_)

            g_C_l2_vector_pair_sso(grad_C,  m_, C, Xm, R, idx_grp, is_, js_, inj, y[t], y[t+m_], W[m])
            g_X_l2_vector_pair_sso(grad_Xm, m_, C, Xm, R, idx_grp, is_, js_, inj, y[t], y[t+m_], W[m])

            if m_ == 0:
                g_R_l2_Hankel_sgd_sso(grad_R, C, Xm, R, idx_grp, inj, y[t], W[m])

    return grad_C, grad_X, grad_R

def g_C_l2_vector_pair_sso(grad, m_, C, Xm, R, idx_grp, is_, js_, inj, yp, yf, Wm):

    p,n = C.shape

    C___ = C.dot(Xm)   # mad-
    C_tr = C.dot(Xm.T) # ness        

    for i in is_:
        a = idx_grp[i]
        SC, Sy = np.zeros((n,n),dtype=C.dtype), np.zeros(n,dtype=C.dtype)
        for j in js_:
            b = idx_grp[j]        
            SC += C_tr[b,:].T.dot(C_tr[b,:]) * Wm[i,j]
            Sy += yp[b].dot(C_tr[b,:]) * Wm[i,j]
        grad[a,:] += C[a,:].dot( SC ) - np.outer(yf[a], Sy)

    for j in js_:
        b = idx_grp[j]        
        SC, Sy = np.zeros((n,n),dtype=C.dtype), np.zeros(n,dtype=C.dtype)
        for i in is_:        
            a = idx_grp[i]        
            SC += C___[a,:].T.dot(C___[a,:]) * Wm[i,j]
            Sy += yf[a].dot(C___[a,:]) * Wm[i,j]
        grad[b,:] += C[b,:].dot( SC) - np.outer(yp[b], Sy)
        
    if m_ == 0:
        for i in inj:
            anb = idx_grp[i]
            grad[anb,:] += (R[anb]*Wm[i,i]).reshape(-1,1) * (C___[anb,:] + C_tr[anb,:])

def g_B_l2_vector_pair_sso(grad, m_, C, Am, Pi, R, idx_grp, is_, js_, inj, yp, yf, Wm):

    p,n = C.shape
    CAm = C.dot(Am)
    for i in is_:
        a = idx_grp[i]
        SC, Sy = np.zeros_like(Pi), np.zeros(n,dtype=C.dtype)
        for j in js_:
            b = idx_grp[j]
            SC += C[b,:].T.dot(C[b,:]) * Wm[i,j]
            Sy += yp[b].dot(C[b,:] * Wm[i,j])
        grad += CAm[a,:].T.dot(CAm[a,:]).dot(Pi).dot(SC)-np.outer(yf[a].dot(CAm[a,:]),Sy)

    if m_ == 0:
        for i in inj:
            anb = idx_grp[i]
            grad += CAm[anb,:].T.dot( (R[anb] * Wm[i,i]).reshape(-1,1) * C[anb,:] )

def g_X_l2_vector_pair_sso(grad, m_, C, Xm, R, idx_grp, is_, js_, inj, yp, yf, Wm):

    p,n = C.shape
    for i in is_:
        a = idx_grp[i]
        SC, Sy = np.zeros((n,n),dtype=C.dtype), np.zeros(n,dtype=C.dtype)
        for j in js_:
            b = idx_grp[j]
            SC += C[b,:].T.dot(C[b,:]) * Wm[i,j]
            Sy += yp[b].dot(C[b,:]) * Wm[i,j]
        grad += C[a,:].T.dot(C[a,:]).dot(Xm).dot(SC)-np.outer(yf[a].dot(C[a,:]),Sy)
        
    if m_ == 0:
        for i in inj:
            anb = idx_grp[i]
            grad += C[anb,:].T.dot( (R[anb] * Wm[i,i]).reshape(-1,1) * C[anb,:] )

def g_R_l2_Hankel_sgd_sso(grad, C, X0, R, idx_grp, inj, yp, W0):

    for i in inj:
        b = idx_grp[i]         
        grad[b] += (R[b] + np.sum(C[b,:] * C[b,:].dot(X0.T),axis=1) - yp[b]**2) * W0[i,i]

# evaluation of target loss function

def f_blank(C,A,Pi,R,lag_range,Qs,idx_grp,co_obs,idx_a,idx_b):

    return 0.

def f_l2_Hankel_comp_Q_Om(n,y,lag_range,obs_scheme,idx_a,idx_b,W,ts=None,ms=None):

    T,p = y.shape
    kl_ = np.max(lag_range)+1
    pa, pb = len(idx_a), len(idx_b)
    get_observed = obs_scheme.gen_get_observed()
    idx_grp = obs_scheme.idx_grp

    ts = range(T-kl_) if ts is None else ts
    ms = range(len(lag_range)) if ms is None else ms

    Qs = [np.zeros((pa,pb), dtype=y.dtype) for m in range(len(lag_range))]
    Om = [np.zeros((pa,pb), dtype=bool) for m in range(len(lag_range))]

    for m in ms:
        m_ = lag_range[m]
        for t in ts:
            a = np.intersect1d(get_observed(t+m_), idx_a)
            b = np.intersect1d(get_observed(t),    idx_b)
            a_Q = np.in1d(idx_a, a)
            b_Q = np.in1d(idx_b, b)

            Qs[m][np.ix_(a_Q, b_Q)] += np.outer(y[t+m_,a], y[t,b])
            Om[m][np.ix_(a_Q, b_Q)] = True

    if np.all(W[0].shape == (len(idx_grp), len(idx_grp))):
        for m in ms:
            for i in range(len(idx_grp)):
                for j in range(len(idx_grp)):
                    Qs[m][np.ix_(idx_grp[i], idx_grp[j])] *= W[m][i,j]

    elif np.all(W[0].shape == (p, p)):
        for m in ms:
            Qs[m] = Qs[m] * W[m][np.ix_(idx_a,idx_b)]

    else:
        raise Exception('shape misfit for weights W[m] at time-lag m=0')

    return Qs, Om    

def f_l2_Hankel_nl(C,X,R,Qs,Om,lag_range,ms,idx_a,idx_b):

    p,n = C.shape
    L = 0.

    for m in ms:
        CXC = C[idx_a,:].dot(X[m*n:(m+1)*n,:]).dot(C[idx_b,:].T)
        if lag_range[m]==0:
            idx_R = np.where(np.in1d(idx_b,idx_a))[0]
            CXC[np.arange(len(idx_R)), idx_R] += R[idx_a]
        L += np.sum( (CXC - Qs[m])[Om[m]]**2)

    return 0.5 * L


def f_l2_block(C,AmPi,Q,idx_grp,co_obs,idx_a,idx_b,W=None):
    "Hankel reconstruction error on an individual Hankel block"

    err = 0.
    for i in range(len(idx_grp)):
        err_ab = 0.
        a = np.intersect1d(idx_grp[i],idx_a)
        b = np.intersect1d(co_obs[i], idx_b)
        a_Q = np.in1d(idx_a, idx_grp[i])
        b_Q = np.in1d(idx_b, co_obs[i])

        v = (C[a,:].dot(AmPi).dot(C[b,:].T) - Q[np.ix_(a_Q,b_Q)])
        v = v.reshape(-1,) if  W is None else W.reshape(-1,) * v.reshape(-1,)

        err += v.dot(v)

    return err

def f_l2_inst(C,Pi,R,Q,idx_grp,co_obs,idx_a,idx_b,W=None):
    "reconstruction error on the instantaneous covariance"

    err = 0.
    if not Q is None:
        for i in range(len(idx_grp)):

            a = np.intersect1d(idx_grp[i],idx_a)
            b = np.intersect1d(co_obs[i], idx_b)
            a_Q = np.in1d(idx_a, idx_grp[i])
            b_Q = np.in1d(idx_b, co_obs[i])

            v = (C[a,:].dot(Pi).dot(C[b,:].T) - Q[np.ix_(a_Q,b_Q)])
            idx_R = np.where(np.in1d(b,a))[0]
            v[np.arange(len(idx_R)), idx_R] += R[a]
            v = v.reshape(-1,) if  W is None else W.reshape(-1,)*v.reshape(-1,)

            err += v.dot(v)

    return err

###########################################################################
# utility, semi-scripts, plotting
###########################################################################


def track_correlations(C,A,B,X,R,Qs, p, n, lag_range,
    idx_a=None, idx_b=None, mmap = False, data_path=None):

    Pi = None if B is None else B.dot(B.T) 

    if X is None:
        X = np.vstack([np.linalg.matrix_power(A,m) for m in lag_range]).dot(Pi)
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

def plot_slim(Qs,Om,lag_range,pars,idx_a,idx_b,traces,mmap,data_path):

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
        plt.plot(Q[Om[m]].reshape(-1), Qrec[Om[m]].reshape(-1), '.')
        plt.title( ('m = ' + str(m_) + ', corr = ' + 
        str(np.corrcoef( Qrec[Om[m]].reshape(-1), (Qs[m][Om[m]]).reshape(-1) )[0,1])))
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


def print_slim(Qs,Om,lag_range,pars,idx_a,idx_b,traces,mmap,data_path):

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
        str(np.corrcoef( Qrec[Om[m]].reshape(-1), (Qs[m][Om[m]]).reshape(-1) )[0,1]))
        if mmap:
            del Q

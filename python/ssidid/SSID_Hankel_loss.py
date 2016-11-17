import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

###########################################################################
# stochastic gradient descent: gradients w.r.t. C, R, cov(x_{t+m},x_t)
###########################################################################

def run_bad(lag_range,n,y,Qs,
            Om,sub_pops,idx_grp,co_obs,obs_idx,
            obs_pops=None, obs_time=None, W=None, 
            idx_a=None,idx_b=None,
            init='default',
            alpha_C=0.001, b1_C=0.9, b2_C=0.99, e_C=1e-8, 
            alpha_R=None, b1_R=None, b2_R=None, e_R=None, 
            alpha_X=None, b1_X=None, b2_X=None, e_X=None, 
            max_iter=100, max_zip_size=np.inf, eps_conv=0.99999,
            batch_size=1,
            verbose=False, mmap=False, data_path=None):

    alpha_R = alpha_C if alpha_R is None else alpha_R
    alpha_X = alpha_C if alpha_X is None else alpha_X

    b1_R = b1_C if b1_R is None else b1_R
    b1_X = b1_C if b1_X is None else b1_X

    b2_R = b2_C if b2_R is None else b2_R
    b2_X = b2_C if b2_X is None else b2_X

    e_R = e_C if e_R is None else e_R
    e_X = e_C if e_X is None else e_X


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

    f,g,batch_draw,track_corrs,converged = l2_bad_sis_setup(
                                           lag_range=lag_range,n=n,T=T,
                                           y=y,Qs=Qs,Om=Om,
                                           idx_a=idx_a, idx_b=idx_b,
                                           sub_pops=sub_pops, W=W,
                                           obs_pops=obs_pops, obs_time=obs_time,
                                           idx_grp=idx_grp,obs_idx=obs_idx,
                                           batch_size=batch_size,
                                           mmap=mmap, data_path=data_path,
                                           max_iter=max_iter, eps_conv=eps_conv)
    if verbose:
        print('starting descent')    
    pars_est, traces = adam_zip_bad(f=f,g=g,pars_0=pars_init,
                                    alpha_C=alpha_C,b1_C=b1_C,b2_C=b2_C,e_C=e_C,
                                    alpha_R=alpha_R,b1_R=b1_R,b2_R=b2_R,e_R=e_R,
                                    alpha_X=alpha_X,b1_X=b1_X,b2_X=b2_X,e_X=e_X,
                                    batch_draw=batch_draw,converged=converged,
                                    track_corrs=track_corrs,max_iter=max_iter,
                                    max_zip_size=max_zip_size,batch_size=batch_size,
                                    verbose=verbose)

    return pars_init, pars_est, traces



# decorations

def l2_bad_sis_setup(lag_range,T,n,y,Qs,Om,idx_grp,obs_idx,obs_pops=None, obs_time=None,
                     sub_pops=None,idx_a=None, idx_b=None, W=None,  
                     max_iter=np.inf, eps_conv=0.99999,
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

    def g(C, X, R, ts, m):
        return  g_l2_Hankel_sgd(C,X,R,y,lag_range,ts,m,
            get_observed=get_observed,linear=False, W=None)

    def f(C,X,R):
        return f_l2_Hankel_nl(C,X,None,R,lag_range,Qs,idx_grp,co_obs,
                idx_a=idx_a,idx_b=idx_b,W=W)

    def track_corrs(C, A, Pi, X, R) :
         return track_correlations(Qs, p, n, lag_range, Om, C, A, Pi, X, R,  
                        idx_a, idx_b, mmap, data_path)


    # setting up the stochastic batch selection:
    batch_draw, g_sgd = l2_sgd_draw(p, T, lag_range, batch_size, 
                                    idx_grp, co_obs, g)

    # set convergence criterion 
    if batch_size is None:
        #print('bach sizes, stopping if loss < ' + str(eps_conv) + ' previous loss')
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

def l2_sgd_draw(p, T, lag_range, batch_size, idx_grp, co_obs, g, Om=None):
    "returns sequence of indices for sets of neuron pairs for SGD"

    kl = len(lag_range)
    kl_ = np.max(lag_range)+1
    if batch_size is None:

        def batch_draw():
            ts = (np.random.permutation(np.arange(T - (kl_) )) , )
            ms = (lag_range, )   
            return ts, ms
        def g_sgd(C, X, R, ts, ms, i):
            return g(C, X, R, ts[i], ms[i])


    elif batch_size == 1:

        def batch_draw():
            ts = np.random.permutation(np.arange(T - (kl_) ))
            ms = np.random.randint(0, kl, size=(len(ts),))         
            return ts, ms
        def g_sgd(C, X, R, ts, ms, i):
            return g(C, X, R, (ts[i],), (ms[i],))


    elif batch_size > 1:

        def batch_draw():
            ts = np.random.permutation(np.arange(T - (kl_) ))
            ms = np.random.randint(0,kl,size=len(ts)//batch_size) # one lag per gradient!
            return ([ts[i*batch_size:(i+1)*batch_size] for i in range(len(ts)//batch_size)],
                     ms)

        def g_sgd(C, X, R, ts, ms, i):
            return g(C, X, R, ts[i], (ms[i],))

    return batch_draw, g_sgd



# main optimiser

def adam_zip_bad(f,g,pars_0,
                alpha_C,b1_C,b2_C,e_C,
                alpha_R,b1_R,b2_R,e_R,
                alpha_X,b1_X,b2_X,e_X,
                batch_draw,track_corrs,converged,
                max_iter,batch_size,max_zip_size,
                verbose):


    # initialise pars
    p, n = pars_0['C'].shape
    kl   = pars_0['X'].shape[0]//n
    C,X,R = set_adam_init(pars_0, p, n, kl)
    A, Pi = None, None

    # setting up Adam
    b1_C,b2_C,e_C,vC, b1_R,b2_R,e_R,vR, b1_X,b2_X,e_X,vX \
    = set_adam_pars(batch_size,p,n,kl,b1_C,b2_C,e_C, 
                                      b1_R,b2_R,e_R, 
                                      b1_X,b2_X,e_X)
    mC, mR, mX = np.zeros((p,n)), np.zeros(p), np.zeros((kl*n,n))
    t_iter, t, ct_iter = 0, 0, 0 
    corrs  = np.zeros((kl, 12))

    def zip_range(zip_size):
        if p > 1000:
            return progprint_xrange(zip_size, perline=100)
        else:
            return range(zip_size)

    # trace function values
    fun = np.empty(max_iter)    

    corrs[:,ct_iter] = track_corrs(C, A, Pi, X, R) 
    ct_iter += 1

    def adamize(m, v, b1, b2):
        if b1 != 1.:                    
            m = m / (1-b1**t)
        else:
            m = m.copy()
        if b2 != 1.:
            v = v / (1-b2**t)
        else:
            v = v.copy()

        return m,v


    while not converged(t_iter, fun):

        ts, ms = batch_draw()        
        zip_size = get_zip_size(batch_size, p, ts, max_zip_size)
        for idx_zip in zip_range(zip_size):
            t += 1

            # get data point(s) and corresponding gradients: 


            grad_C,grad_X,grad_R = g(C,X,R,ts,ms,idx_zip)

            C, mC, vC = adam(C,grad_C,mC,vC,alpha_C,b1_C,b2_C,e_C)
            X, mX, vX = adam(X,grad_X,mX,vX,alpha_X,b1_X,b2_X,e_X)
            R, mR, vR = adam(R,grad_R,mR,vR,alpha_R,b1_R,b2_R,e_R)


        if t_iter < max_iter:          # really expensive!
            fun[t_iter] = f(C,X,R)

        if np.mod(t_iter,max_iter//10) == 0:
            if verbose:
                print('finished %', 100*t_iter/max_iter+10)
                print('f = ', fun[t_iter])
            corrs[:,ct_iter] = track_corrs(C, A, Pi, X, R) 
            ct_iter += 1

        t_iter += 1

    # final round over R (we before only updated R_ii just-in-time)
    #R = s_R(R, C, X[:n,:].reshape(n,n))

    corrs[:,ct_iter] = track_corrs(C, A, Pi, X, R)
    fun = fun[:t_iter]

    if verbose:
        print('total iterations: ', t)

    pars_out = {'C' : C, 'X': X, 'R' : R }
    pars_out['A'], pars_out['Pi'] = np.nan*np.ones((n,n)), np.nan*np.ones((n,n))

    return pars_out, (fun,corrs)    

def set_adam_pars(batch_size,p,n,kl,b1_C,b2_C,e_C, 
                                    b1_R,b2_R,e_R, 
                                    b1_X,b2_X,e_X):

    if batch_size is None:
        print('using batch gradients - switching to plain gradient descent')
        b1_C, b2_C, e_C, v_0_C = 0, 1.0, 0, np.ones((p,n))
        b1_R, b2_R, e_R, v_0_R = 0, 1.0, 0, np.ones(p)
        b1_X, b2_X, e_X, v_0_X = 0, 1.0, 0, np.ones((kl*n,n))
    elif batch_size >= 1:
        v_0_C, v_0_R, v_0_X = np.zeros((p,n)),np.ones(p),np.ones((kl*n,n))
    else: 
        raise Exception('cannot handle selected batch size')

    return b1_C,b2_C,e_C,v_0_C, b1_R,b2_R,e_R,v_0_R, b1_X,b2_X,e_X,v_0_X


def adam(w,g,m,v,a,b1,b2,e):
    m = (b1 * m + (1-b1) * g)
    v = (b2 * v + (1-b2) * g**2)
    mh,vh = adamize(m,v,b1,b2)
    w -= a * mh / (np.sqrt(vh) + e)

    return w, m, v

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



def g_l2_Hankel_sgd(C,X,R,y,lag_range,ts,ms,get_observed,linear=False, W=None):
    p,n = C.shape
    grad_C = np.zeros((p,n))
    grad_X = np.zeros(X.shape)
    grad_R = np.zeros(p)
    idx_ct = np.zeros(p,dtype=np.int32)

    for m in ms:
        m_ = lag_range[m]
        Xm = X[m*n:(m+1)*n, :]
        X0 = X[:n,:]
        for t in ts:
            a = get_observed(p, t+m_)
            b = get_observed(p, t)

            idx_ct[a] += 1
            idx_ct[b] += 1

            CC_a = C[a,:].T.dot(C[a,:])
            CC_b = C[b,:].T.dot(C[b,:]) if not a is b else CC_a    

            g_C_l2_Hankel_vector_pair(grad_C, m_, C, Xm, R, a, b, CC_a, CC_b, y[t,b], y[t+m_,a])    
            grad_X[m*n:(m+1)*n,:] += g_X_l2_vector_pair(C, Xm, R, 
                                            a, b, CC_a, CC_b, y[t,b], y[t+m_,a])

            XCT = X0.dot(C[b,:].T)
            y2 = y[t,b]**2
            for s in range(len(b)):
                grad_R[b[s]] += R[b[s]] + C[b[s],:].dot(XCT[:,s]) - y2[s]

        idx_ct = np.maximum(idx_ct, 1)
        grad_X[m*n:(m+1)*n,:] /= len(ts)
    grad_C = grad_C / idx_ct.reshape(-1,1)
    grad_R = grad_R / idx_ct

    return grad_C, grad_X, grad_R

def g_C_l2_Hankel_vector_pair(grad, m_, C, Xm, R, a, b, CC_a, CC_b, yp, yf):

    if m_ == 0:
        grad[a,:] += ( C[a,:].dot(Xm.dot(CC_b))   + R[b].reshape(-1,1)*C[b,:] - np.outer(yf, yp.dot(C[b,:])) ).dot(Xm.T)
        grad[b,:] += ( C[b,:].dot(Xm.T.dot(CC_a)) + R[a].reshape(-1,1)*C[a,:]- np.outer(yp, yf.dot(C[a,:])) ).dot(Xm)

    else:
        grad[a,:] += ( C[a,:].dot(Xm.dot(CC_b))   - np.outer(yf, yp.dot(C[b,:])) ).dot(Xm.T)
        grad[b,:] += ( C[b,:].dot(Xm.T.dot(CC_a)) - np.outer(yp, yf.dot(C[a,:])) ).dot(Xm)


def g_X_l2_vector_pair(C, Xm, R, a, b, CC_a, CC_b, yp, yf):

    grad = CC_a.dot(Xm).dot(CC_b) - np.outer(yf.dot(C[a,:]), yp.dot(C[b,:]))

    return grad

def g_R_l2_Hankel_sgd(C, X0, R, y, ts, get_observed):

    p,n = C.shape
    grad = np.zeros(p)

    for t in ts:
        b = get_observed(p, t)
        XCT = X0.dot(C[b,:].T)
        y2 = y[t,b]**2

        for s in range(len(b)):
            grad[b[s]] += R[b[s]] + C[b[s],:].dot(XCT[:,s]) - y2[s]

    return grad / len(ts)

# solving some parameters with others fixed (mostly only in fully observed case)

def s_R_l2_Hankel_sgd(R, C, Pi, cov_y, idx_grp, co_obs):

    if not cov_y is None:
        for i in range(len(idx_grp)):
            a,b = idx_grp[i], co_obs[i]
            ab = np.intersect1d(a,b)

            PiC = Pi.dot(C[ab,:].T)
            for s in range(len(ab)):
                R[ab[s]] = cov_y[ab[s], ab[s]] - C[ab[s],:].dot(PiC[:,s])
            R[ab] = np.maximum(R[ab], 0)

    return R

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

# evaluation of target loss function

def f_blank(C,A,Pi,R,lag_range,Qs,idx_grp,co_obs,idx_a,idx_b):

    return 0.


def f_l2_Hankel_nl(C,X,Pi,R,lag_range,Qs,idx_grp,co_obs,
                   idx_a=None,idx_b=None,W=None):
    "returns overall l2 Hankel reconstruction error"

    kl = len(lag_range)
    p,n = C.shape
    idx_a = np.arange(p) if idx_a is None else idx_a
    idx_b = idx_a if idx_b is None else idx_b
    assert (len(idx_a), len(idx_b)) == Qs[0].shape

    err = f_l2_inst(C,X[:n, :],R,Qs[0],idx_grp,co_obs,idx_a,idx_b)
    for m in range(1,kl):
        err += f_l2_block(C,X[m*n:(m+1)*n, :],Qs[m],idx_grp,co_obs,idx_a,idx_b,W)
            
    return err/(kl)
    
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


def track_correlations(Qs, p, n, lag_range, Om, C, A, Pi, X, R,
    idx_a=None, idx_b=None, mmap = False, data_path=None):

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

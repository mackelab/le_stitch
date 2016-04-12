import numpy as np
import matplotlib.pyplot as plt
import warnings
import control
from scipy.linalg import solve_discrete_lyapunov as dlyap
from numpy.lib.stride_tricks import as_strided

def FitLDSParamsSSID(seq, n):

    T, p      = seq['y'].shape

    SIGfp, SIGyy = generateCovariancesFP(seq=seq, hS=n)
            
    params = post_process(ssidSVD(SIGfp=SIGfp,SIGyy=SIGyy,n=n));
    params = add_d(data=seq['y'], params=params)

    return params

def run_exp_iterSSID(seq, seq_f, n, num_iter=100,pi_method='proper', plot_flag=True):

    T,p = seq['y'].shape

    # use observation scheme informatoin to mask data
    for i in range(len(seq['obs_time'])):
        idx_t = np.arange(seq['obs_time'][0]) if i == 0 \
            else np.arange(seq['obs_time'][i-1], seq['obs_time'][i])
        idx_y = np.setdiff1d(np.arange(p), seq['sub_pops'][i])
        seq['y'][np.ix_(idx_t, idx_y)] = np.NaN
    
    # generate covs from full ground truth data for comparison
    SIGfp_f, SIGyy_f = generateCovariancesFP(seq=seq_f,hS=n)

    # fit model with iterative SSID
    params, SIGfp_new, SIGyy_new, lPSIGfp, lPSIGyy, broken = \
        iterSSID(seq=seq, n=n, num_iter=num_iter, 
            pi_method=pi_method, init=None, alpha=1)

    # reconstruct covs from final estimate 
    SIGfp, SIGyy = construct_hankel_cov(params=params[-1])

    # visualize results
    if plot_flag:
        idx_stitched = np.invert(lPSIGyy)
        m = np.min([SIGyy.min(), SIGyy_f.min()])
        M = np.max([SIGyy.max(), SIGyy_f.max()])
        plt.figure(figsize=(12,8))
        plt.subplot(2,3,1)
        plt.imshow(SIGyy_f, interpolation='none')
        plt.clim(m,M)
        plt.title('SIGyy true')
        plt.subplot(2,3,2)
        plt.imshow(SIGyy, interpolation='none')
        plt.clim(m,M)
        plt.title('SIGyy est.')
        plt.subplot(2,3,3)
        if np.any(idx_stitched):
            m = np.min([SIGyy[idx_stitched].min(), SIGyy_f[idx_stitched].min()])
            M = np.max([SIGyy[idx_stitched].max(), SIGyy_f[idx_stitched].max()])
            plt.plot([m,M], [m,M], 'k')
            plt.axis([m,M,m,M])
        plt.hold(True)
        plt.plot(SIGyy[np.invert(idx_stitched)], 
                 SIGyy_f[np.invert(idx_stitched)], 'b.')
        plt.plot(SIGyy[idx_stitched], SIGyy_f[idx_stitched], 'r.')
        plt.xlabel('est.')
        plt.ylabel('true')

        if n*p <= 2000:
            idx_stitched = np.invert(lPSIGfp)
            m = np.min([SIGfp.min(), SIGfp_f.min()])
            M = np.max([SIGfp.max(), SIGfp_f.max()])
            plt.subplot(2,3,4)
            plt.imshow(SIGfp_f, interpolation='none')
            plt.clim(m,M)
            plt.title('SIGfp true')
            plt.subplot(2,3,5)
            plt.imshow(SIGfp, interpolation='none')
            plt.clim(m,M)
            plt.title('SIGfp est.')
            plt.subplot(2,3,6)
            if np.any(idx_stitched):
                m = np.min([SIGfp[idx_stitched].min(), SIGfp_f[idx_stitched].min()])
                M = np.max([SIGfp[idx_stitched].max(), SIGfp_f[idx_stitched].max()])
                plt.plot([m,M], [m,M], 'k')
                plt.axis([m,M,m,M])
            plt.hold(True)
            plt.plot(SIGfp[np.invert(idx_stitched)], 
                     SIGfp_f[np.invert(idx_stitched)], 'b.')
            plt.plot(SIGfp[idx_stitched], SIGfp_f[idx_stitched], 'r.')
            plt.xlabel('est.')
            plt.ylabel('true')

        plt.show()    

    return params, SIGfp, SIGyy, lPSIGfp, lPSIGyy, SIGfp_f, SIGyy_f


def iterSSID(seq, n, num_iter=100, init=None, alpha=1.,pi_method='proper'):

    T,p = seq['y'].shape

    print(seq['y'].shape)

    params = []
    broken = False

    if p > 200:
        print('p = ', p)
        print('very high-dimensional data! Will take a while. Switching to verbose...')
        verbose=True
    else:
        verbose=False

    if verbose:
        print('generating data cov. mat')

    SIGfp_new, SIGyy_new = generateCovariancesFP(seq=seq, hS=n)

    lnPSIGfp, lnPSIGyy = np.isnan(SIGfp_new),np.isnan(SIGyy_new) # for indexing
    lPSIGfp, lPSIGyy = np.invert(lnPSIGfp), np.invert(lnPSIGyy) 
    PSIGfp, PSIGyy = SIGfp_new[lPSIGfp], SIGyy_new[lPSIGyy]  # observed parts


    if verbose:
        print('computing naive iterSSID')
    # 'zero'-th iteration: used ssidSVD on Hankel cov matrix with NaN set to 0:
    SIGfp_new[lnPSIGfp] = 0
    SIGyy_new[lnPSIGyy] = 0
    params.append(post_process(ssidSVD(SIGfp=SIGfp_new,
                                       SIGyy=SIGyy_new,
                                       n=n,
                                       pi_method=pi_method)))
    params[-1] = add_d(seq['y'], params[-1])

    # main iterations: fill in NAN's iteratively using missing-value SVD:
    SIGfp_old = SIGfp_new.copy() if init is None else init
    SIGyy_old = SIGyy_new.copy()

    if verbose:
        print('computing iterative SSID')
    for t in range(1,num_iter+1):

        if verbose:
            print(t)
        params.append(post_process(ssidSVD(SIGfp=SIGfp_old,
                                           SIGyy=SIGyy_old,
                                           n=n,
                                           pi_method=pi_method,
                                           params_old=params[-1])))

        params[-1] = add_d(seq['y'], params[-1])

        SIGfp_new, SIGyy_new = construct_hankel_cov(params[-1])

        if np.any(np.isnan(SIGfp_new)) or not np.all(np.isfinite(SIGfp_new)):
            print('reconstructed Hankel cov matrix contains NaNs or Infs!')
            print('cancelling iteration')

            broken = True

            break

        # make sure we keep the observed part right
        SIGfp_new[lPSIGfp], SIGyy_new[lPSIGyy] = PSIGfp, PSIGyy
        SIGfp_new[lnPSIGfp] = alpha*SIGfp_new[lnPSIGfp] \
                            +(1-alpha)*SIGfp_old[lnPSIGfp]
        SIGyy_new[lnPSIGyy] = alpha*SIGyy_new[lnPSIGyy] \
                            +(1-alpha)*SIGyy_old[lnPSIGyy]
        SIGyy_new = symmetrize(SIGyy_new)

        # hack: change halves of C if helpful
        params[-1] = flip_emission_signs(SIGyy_new, seq['sub_pops'], params[-1])


        SIGfp_old, SIGyy_old = SIGfp_new, SIGyy_new # shallow copies!


    if broken:
        broken = params[-1]
        params.pop()
        SIGfp_new, SIGyy_new = construct_hankel_cov(params[-1])
        SIGfp_new[lPSIGfp], SIGyy_new[lPSIGyy] = PSIGfp, PSIGyy
        SIGyy_new = symmetrize(SIGyy_new)

    return params, SIGfp_new, SIGyy_new, lPSIGfp, lPSIGyy, broken


def flip_emission_signs(SIGyy, sub_pops, params):

    # we drop this for now, as it hardly ever flips anything
    """
    if len(sub_pops) == 1:
        idx_overlap = sub_pops[0]
    elif len(sub_pops) == 2:
        idx_overlap = np.intersect1d(sub_pops[0], sub_pops[1])
    else:
        raise Exception('more than two subpopulations not yet implemented')

    cov_h = params['C'][idx_overlap,:].dot(params['Pi']).dot(params['C'].T) + \
        params['R'][idx_overlap,:]


    for i in range(len(sub_pops)):
        idx_sub = rem_overlap(sub_pops[i], idx_overlap)
        SIGp_est, SIGp_true = cov_parts(cov_h,SIGyy,idx_sub,idx_overlap)

        if check_flip(SIGp_est, SIGp_true):
            params['C'][idx_sub,:] *= -1
    """    
    return params

def check_flip(x_est, x_true, thresh=0.5):
    return np.mean( np.sum(x_est * x_true, 1) < 0 ) > thresh

def rem_overlap(idx_sub_pop, idx_overlap):
    return np.setdiff1d(idx_sub_pop, idx_overlap)

def cov_parts(SIG_est, SIG_true, idx_sub, idx_overlap):
    return SIG_est[:,idx_sub],SIG_true[np.ix_(idx_overlap,idx_sub)]


def construct_hankel_cov(params):

    p, hS = params['C'].shape
    SIGfp  = np.zeros((p*hS,p*hS))

    SIGyy = params['C'].dot(params['Pi'].dot(params['C'].T)) + params['R']

    covxx = params['Pi']
    for k in range(2*hS-1):
        
        covxx =  params['A'].dot(covxx)
        lamK = params['C'].dot(covxx.dot(params['C'].T))

        if k < hS-0.5:
            
            for kk in range(k + 1):
                offset0, offset1 = (k-kk)*p, kk*p
                SIGfp[offset0:offset0+p, offset1:offset1+p] = lamK
                
        else:

            for kk in range(2*hS - k -1):
                offset0, offset1 = (hS-kk-1)*p, (kk+k+1-hS)*p
                SIGfp[offset0:offset0+p,offset1:offset1+p] = lamK

    return SIGfp, SIGyy


def generateCovariancesFP(seq,hS):
    
    T, p = seq['y'].shape
    
    SIGfp  = np.zeros((p*hS,p*hS),dtype=np.float64)
    SIGyy  = np.zeros((p,p),dtype=np.float64)
    
    
    idxObs = np.invert(np.isnan(seq['y'])).astype(np.int)

    sYYobs = idxObs.T.dot(idxObs)
    nPSIG  = (sYYobs==0)
    sYYobs = np.maximum(sYYobs, 1)
    
    Ytot   = seq['y'] - np.nanmean(seq['y'],0)
    Ytot[np.isnan(Ytot)] = 0

    Yshift = Ytot        
    lamK   = Ytot.T.dot(Yshift)
    SIGyy  = lamK;            

    for k in range(2*hS-1):
        
        Yshift  = np.roll(Yshift, 1, axis = 0)
        lamK    = Ytot.T.dot(Yshift) / sYYobs
        lamK[nPSIG] = np.NaN

        if k < hS-0.5:
            
            for kk in range(k + 1):
                offset0, offset1 = (k-kk)*p, kk*p
                SIGfp[offset0:offset0+p, offset1:offset1+p] = lamK
                
        else:

            for kk in range(2*hS - k -1):
                offset0, offset1 = (hS-kk-1)*p, (kk+k+1-hS)*p
                SIGfp[offset0:offset0+p,offset1:offset1+p] = lamK
             
    SIGyy = symmetrize(SIGyy)/sYYobs  
    SIGyy[nPSIG] = np.NaN      
             
    return SIGfp, SIGyy

def ssidSVD(SIGfp,SIGyy,n, pi_method='proper', params_old = None):
    
    minVar    = 1e-5
    minVarPi  = 1e-5   

    p = np.size(SIGyy,0)

    if p > 200:
        print('p = ', p)
        print('very high-dimensional data! Will take a while. Switching to verbose...')
        verbose=True
    else:
        verbose=False
       
    
    if verbose:
        print('computing SVD:')
    UU,SS,VV = np.linalg.svd(SIGfp) # SIGfp = UU.dot(diag(SS).dot(VV)

    VV = VV.T
    SS = np.diag(SS[:n])
    UU = UU[:,:n]
    VV = VV[:,:n]
   
    Obs = np.dot(UU,SS)

    if verbose:
        print('computing A, C:')

    A = np.linalg.lstsq(Obs[:-p,:],Obs[p:,:])[0]
    C = Obs[:p,:]
    Chat = VV[:p,:n]
    
    if verbose:
        print('computing Pi:')

    # hack: abolish constant flipping of mathematical signs of latent states...
    if not params_old is None:
        flips = np.diag(2*(np.sum(C * params_old['C'],0) > 0).astype(np.float) - 1)
        A = flips.dot(A.dot(flips))
        C = C.dot(flips)
        Chat = Chat.dot(flips)


    if pi_method=='proper':
        Pi,_,_ = control.matlab.dare(A=A.T,B=-C.T,Q=np.zeros((n,n)),R=-SIGyy,
            S=Chat.T, E=np.eye(n))    

    else:
        #warnings.warn(('Will not solve DARE, using heuristics; this might '
        #    'lead to poor estimates of Q and V0'))  
        Pi = np.linalg.lstsq(A,np.dot(Chat.T,np.linalg.pinv(C.T)))[0]


    if verbose:
        print('computing Q, R:')
                   
    D, V = np.linalg.eig(Pi)
    D[D < minVarPi] = minVarPi
    Pi = V.dot(np.diag(D)).dot(V.T)
    Pi = np.real(symmetrize(Pi))
    Q = Pi - np.dot(np.dot(A,Pi),A.T)
    D, V = np.linalg.eig(Q); 
    D[D<minVar] = minVar

    Q = np.dot(np.dot(V,np.diag(D)),V.T)
    Q = symmetrize(Q)
    
    R = np.diag(SIGyy-np.dot(np.dot(C,Pi),C.T))
    R.flags.writeable = True
    R[R<minVar] = minVar
    R = np.diag(R)   
    
    return {'A':A, 'Q': Q, 'C': C, 'R': R, 'Pi': Pi}

def post_process(params):

    p, n = params['C'].shape

    #params = stabilize_A(params)

    if isinstance(params['Q'], complex):
        params['Q'] = np.real(params['Q'])
        
    params['Q'] = (params['Q'] + params['Q'].T)/2
    
    if np.min(np.linalg.eig(params['Q'])[0]) < 0:
        
        a,b = np.linalg.eig(params['Q'])
        params['Q'] = a*np.max(b,10**-10)*a.T
    
    if 'V0' not in params:
        params['V0'] = np.real(dlyap(params['A'],params['Q']))
    
    params.update({'mu0':np.zeros(n)}) 
    params['R']  = np.diag(np.diag(params['R']))


    return params

def stabilize_A(params):

    D, V = np.linalg.eig(params['A'])
    if np.any(np.abs(D) > 1):
        print(np.abs(D))
        warnings.warn(('Produced instable dynamics matrix A. Bluntly pushing '
                       'eigenvalues into the unit circle'))  
        D /= np.maximum(np.abs(D), 1)
        print(np.abs(D))
        params['A'] = np.real(V.dot(np.diag(D).dot(np.linalg.inv(V))))

    return params

def add_d(data, params):

    params['d'] = np.nanmean(data,axis = 0)

    return params

def symmetrize(A):
	
    return (A + A.T)/2

def blockarray(*args,**kwargs):
    "Taken from Matthew J. Johnson's 'pybasicbayes' code package"
    return np.array(np.bmat(*args,**kwargs),copy=False)

def observability_mat(pars, k):

    if isinstance(pars, dict):
        A, C = pars['A'], pars['C']
    elif isinstance(pars, tuple) and len(pars) == 2:
        A, C = pars[0], pars[1]

    if len(C.shape)<2:
        C = C.reshape(1,C.size).copy()
    
    return blockarray([[C.dot(np.linalg.matrix_power(A,i))] for i in range(k)])

def reachability_mat(pars, l):

    if isinstance(pars, dict):
        A, B = pars['A'], pars['B']
    elif isinstance(pars, tuple) and len(pars) == 2:
        A, B = pars[0], pars[1]  

    assert len(B.shape)>1

    return blockarray([np.linalg.matrix_power(A,i).dot(B) for i in range(l)])

def Hankel_data_mat(data, k, l=None, N=None):
    "returns the *transpose* of Y_l|l+k-1 (see notation of Katayama, 2004)"

    l = k if l is None else l
    assert k > 1 and l > 1

    data = np.asarray(data, order='C')
    T,p = data.shape
    assert T > N+k+l-2

    Y = as_strided(x=data[l:l+k+N-1,:], 
                   shape=(N,k,p), 
                   strides=(data.strides[0], data.strides[0], data.strides[1])
                   ).reshape(N,p*k)
    assert Y.shape == (N, k*p)

    return Y


def input_output_Hankel_mat(data, inputs, k, l=None, N=None):
    "W_l|l+k-1 = [ U_l|l+k-1, Y_l|l+k-1 ]"

    l = k if l is None else l
    assert k > 1 and l > 1

    N =  data.shape[0]-k-l+1 if N is None else N
    assert N > 0

    p,m = data.shape[1], inputs.shape[1]

    W = np.empty((N, k*(p+m)))
    W[:, :k*m] = Hankel_data_mat(inputs, k, l, N)
    W[:, k*m:] = Hankel_data_mat(data, k, l, N)

    return W

def soft_impute(X, n, eps = 0, max_iter = 500, P_sig=None):

    nan_m = np.asarray(np.isnan(X),dtype=float)
    P_sig_X = X.copy()
    P_sig_X[np.isnan(P_sig_X)] = 0

    Z = np.zeros(X.shape)
    for i in range(max_iter):
        U,s,V = np.linalg.svd(P_sig_X + (Z * nan_m))
        Z = U[:,:n].dot(np.diag(s[:n])).dot(V[:n,:])

    return Z

def xx_Hankel_cov_mat(A,Pi,k,l):
    
    n = A.shape[0]
    assert n == A.shape[1] and n == Pi.shape[0] and n == Pi.shape[1]
    
    H = np.zeros((k*n, l*n))
    print(H.shape)
    
    for kl_ in range(k+l-1):        
        lamK = np.linalg.matrix_power(A,kl_+1).dot(Pi)
        if kl_ < k-0.5:     
            for l_ in range(0, min(kl_ + 1,l)):
                offset0, offset1 = (kl_-l_)*n, l_*n
                H[offset0:offset0+n, offset1:offset1+n] = lamK
                
        else:
            for l_ in range(0, min(k+l - kl_ -1, l, k)):
                offset0, offset1 = (k - l_ - 1)*n, ( l_ + kl_ + 1 - k)*n
                H[offset0:offset0+n,offset1:offset1+n] = lamK
            
    return H

def yy_Hankel_cov_mat(C,A,Pi,k,l,Om=None):
    
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

def f_l2_Hankel(C,A,Pi,k,l,Qs,Om):

    if len(C.shape) < 2:
        C  = C.reshape(Qs[0].shape[0], A.shape[0])        

    err = 0.
    for k_ in range(k):
        for l_ in range(l):
            APi = np.linalg.matrix_power(A, k_+l_ + 1).dot(Pi)  
            err += f_l2_block(C,APi,Qs[k_+l_],Om)
            
    return err/(k*l)
    
def f_l2_block(C,A,Q,Om):

    v = (C.dot(A.dot(C.T)))[Om] - Q[Om]
    
    return v.dot(v.T)/np.sum(Om)

def g_l2_Hankel(C,A,Pi,k,l,Qs,idx_grp, obs_idx):
    
    is_vec = True if (len(C.shape) < 2 or np.min(C.shape)==1) else False
    if is_vec:
        C  = C.reshape(Qs[0].shape[0], A.shape[0])   

    p,n = C.shape
        
    grad = np.zeros((p,n))
    for i in range(len(idx_grp)):
        
        def co_observed(x):
            for idx in obs_idx:
                if x in idx and i in idx:
                    return True
            return False
                
        co_obs_i = [idx_grp[item] for item in np.arange(len(idx_grp)) if co_observed(item)]
        co_obs_i = np.sort(np.hstack(co_obs_i))
        
        for k_ in range(k):
            for l_ in range(l):
                APi = np.linalg.matrix_power(A, k_+l_+1).dot(Pi)                
                grad[idx_grp[i],:] += g_l2_idxgrp(C,APi,Qs[k_+l_],idx_grp[i],co_obs_i)
      
    return grad.reshape(p*n,) if is_vec else grad

def g_l2_idxgrp(C,A,Q,idx_grp_i,co_obs_i):

    Cc, CcT = C[co_obs_i,:].dot(A), C[co_obs_i,:].dot(A.T) 
    Ci, Qic = C[idx_grp_i,:], Q[np.ix_(idx_grp_i, co_obs_i)]
     
    return (Ci.dot(Cc.T) - Qic).dot(Cc) + (Ci.dot(CcT.T) - Qic).dot(CcT)

def comp_subpop_index_mats(sub_pops,idx_grp,overlap_grp,idx_overlap):

    p = np.max([np.max(sub_pops[i]) for i in range(len(sub_pops))]) + 1
    
    Om = np.zeros((p,p), dtype=bool)
    for i in range(len(sub_pops)):
        Om[np.ix_(sub_pops[i],sub_pops[i])] = True

    Ovw = np.zeros((p,p), dtype=int)
    for i in range(len(sub_pops)):
        Ovw[np.ix_(sub_pops[i],sub_pops[i])] += 1
    Ovw = np.minimum(np.maximum(Ovw-1, 0),1)
    Ovw = np.asarray(Ovw, dtype=bool)

    Ovc = np.zeros((p,p), dtype=bool)
    for i in range(len(overlap_grp)):
        for j in range(len(idx_overlap[i])):
            Ovc[np.ix_(idx_grp[overlap_grp[i]],sub_pops[idx_overlap[i][j]])] = True
            Ovc[np.ix_(sub_pops[idx_overlap[i][j]],idx_grp[overlap_grp[i]])] = True
            Ovc[np.ix_(idx_grp[overlap_grp[i]],idx_grp[overlap_grp[i]])] = False
    
    return Om, Ovw, Ovc


###############################################################################
# deterministic MIMO LTI subspace identification
###############################################################################

def d_system(pars, stype='LTI'):

    assert stype in ('LTI',)

    if stype=='LTI':
        def sys(inputs):
            return d_sim_system_LTI(pars, inputs)

    return sys


def d_sim_system_LTI(pars, inputs):

    p,n = pars['C'].shape
    T,m = inputs.shape

    data, stateseq = np.zeros((T,p)), np.zeros((T,n))
    stateseq[0,:] = pars['mu0']
    for t in range(1,T):
        stateseq[t,:] = pars['A'].dot(stateseq[t-1,:]) + \
                        pars['B'].dot(inputs[t,:])
        data[t,:] = pars['C'].dot(stateseq[t,:]) + \
                    pars['D'].dot(inputs[t,:])

    return data, stateseq


def d_est_input_responses(pars, k, l=0, X0=None):

    p,n = pars['C'].shape
    m = pars['B'].shape[1]

    Pi = dlyap(pars['A'],pars['Q'])  
    assert np.all(np.isreal(Pi))

    X0 = np.random.multivariate_normal(np.zeros(n),Pi,m) if X0 is None else X0

    assert X0.shape==(m,n)
    K = k+l+1  # total number of time offsets considered
    Gs = [np.zeros((p,m)) for i in range(K)]

    for idx in range(m):
        data,stateseq,inputs = np.zeros((K,p)),np.zeros((K,n)),np.zeros(m)
        inputs[idx] = 1

        for t in range(K):
            if t==0:
                stateseq[0,:] = X0[idx,:]
            else:
                stateseq[t,:] = pars['A'].dot(stateseq[t-1,:])
            if t==l+1:
                stateseq[t,:] += pars['B'].dot(inputs)

            Gs[t][:,idx] = pars['C'].dot(stateseq[t,:])
            if t==l:
                Gs[t][:,idx] += pars['D'].dot(inputs)

    return Gs, X0


def d_calc_impulse_responses(pars):

    A,B,C,D = pars['A'], pars['B'], pars['C'], pars['D']

    def G(t):
        
        return D if t==0 else C.dot(np.linalg.matrix_power(A,t-1)).dot(B)

    return G


def d_transfer_function(A,B,C,D):

    I = np.eye(A.shape[0])

    def G(z):

        return D + C.dot(np.linalg.inv(z * I - A)).dot(B)

    return G


def d_est_init_state(pars, Gs):

    k = len(Gs)    
    observ = observability_mat(pars, k), 
    Y = blockarray([ [Gs[i]] for i in range(k) ])

    x0 = np.linalg.pinv(observ).dot(Y)


def Ho_Kalman(G,k,l=None,comp_A_from='observability'):

    l = k if l is None else l
    assert k > 1 and l > 1

    if isinstance(G, list):
        p,m = G[0].shape        
        assert len(G)>l and len(G)>k  # G[0] is for t=0! 
        H_kl = blockarray(
            [[G[i+1].reshape(p,m) for i in range(j,j+l)] for j in range(k)])

    else:
        p,m = G(0).shape
        H_kl = blockarray(
            [[G(i+1).reshape(p,m) for i in range(j,j+l)] for j in range(k)])

    U,s,V = np.linalg.svd(H_kl)
    sqs = np.sqrt(s)

    n = np.sum(np.abs(sqs)>1e-3) # latent dimensionality determined from data!
    U, sqS, V = U[:,:n], np.diag(sqs[:n]), V[:n,:]

    observ, reach  = U.dot(sqS), sqS.dot(V)
    assert np.allclose(H_kl, observ.dot(reach))

    if comp_A_from == 'observability':
        A = comp_A(observ, p, k, comp_A_from)
    if comp_A_from == 'reachability':
        A = comp_A(reach, m, l, comp_A_from)


    pars_est = {'A' : A, 
                'B' : reach[:n,:m],
                'C' : observ[:p,:n],
                'D' : G[0] if isinstance(G,list) else G(0)}

    return pars_est, observ, reach, sqs, n

def comp_A(mat, p, k, comp_A_from='observability'):
    "mat is either observability or reachability. p=m, k=l for the latter"

    if comp_A_from == 'reachability':
        A = mat[:,p:k*p].dot(np.linalg.pinv(mat[:,:-p])) 
    elif comp_A_from == 'observability':
        A = np.linalg.pinv(mat[:-p,:]).dot(mat[p:k*p,:])

    return A


def Ho_Kalman_nonzero_init(pars_true, k, l=None):

    l = k if l is None else l
    assert k > 1 and l > 1

    Gs, _ = d_est_input_responses(pars_true,k+l,k) # zero-padded input

    # do Ho-Kalman with response functions starting from input pulse
    pars_est, observ , reach , sqs , n_est = Ho_Kalman(Gs[k:],k,l)

    # estimate non-zero x(0) using observablity and the padded zero inputs
    Y = blockarray([ [Gs[i]] for i in range(k) ])
    Xmk_est = np.linalg.pinv(observ).dot(Y)
    X0_est = np.linalg.matrix_power(pars_est['A'],k).dot(Xmk_est)

    # correct estimates of B and D using estimate x(0)
    pars_est['B'] -= pars_est['A'].dot(X0_est)
    pars_est['D'] -= pars_est['C'].dot(X0_est)

    return pars_est, sqs, n_est, X0_est


def stitching_Ho_Kalman(pars_true, sub_pops, k, l=None, method='impulses'):

    l = k if l is None else l
    assert k > 1 and l > 1

    assert method in ('impulses', 'parameters_obs', 'parameters_reach')

    p,m = pars_true['D'].shape

    num_sub_pops = len(sub_pops)
    pars_js, X0_est_js, M_js, n_est_js, = [],[],[], np.zeros(num_sub_pops,dtype=int)
    Gs = [np.zeros((p,m)) for i in range(k+l+1)]

    for j in range(num_sub_pops):

        # estimate response functions from subsets of true parameters
        pars_obs = pars_true.copy()
        pars_obs['C'] = pars_obs['C'][sub_pops[j],:]
        pars_obs['D'] = pars_obs['D'][sub_pops[j],:]
        # do Ho-Kalman on partial input response functions
        pars_j, _, n_est_j, X0_est_j = Ho_Kalman_nonzero_init(pars_obs,k,l)
        
        # store results for rotation later on
        pars_js.append(pars_j)
        X0_est_js.append(X0_est_j)
        n_est_js[j] = n_est_j
        

    if method=='impulses':
        #  _directly_ stitch the impulse responses  
        # using x(0)=0 renders the latent coordinate systems unimportant
        for j in range(num_sub_pops):
            for t in range(k+l+1):
                Gs[t][sub_pops[j],:] = d_calc_impulse_responses(pars_js[j])(t)
        pars_est,_,_,_,n_est=Ho_Kalman(Gs,k,l=None,comp_A_from='observability')

    elif len(method)>10 and method[:10] =='parameters':

        n_est = n_est_js[0] # assume now that all n's are equal, assert later

        # esp. for stitching from reachability: rotate all systems to subpop #1
        pars_est = {'A' : pars_js[0]['A'].copy(), 
                    'B' : pars_js[0]['B'].copy(), 
                    'C' : np.zeros((p,n_est)),
                    'D' : np.zeros((p,m))
                    }

        # identify overlaps
        _, idx_grp = get_obs_index_groups({'sub_pops':sub_pops,
            'obs_pops':np.arange(num_sub_pops)}, p)
        overlap_grp, _, idx_overlap = get_obs_index_overlaps(idx_grp, sub_pops)

        if method[10:14]=='_obs':

            # get list of pairs (i,j) of systems for rotating j onto i
            pairs_ij = traverse_subpops(sub_pops, idx_overlap, 0).T
            pars_est['A'] = pars_js[pairs_ij[0,0]]['A'] # use basis of first
            pars_est['B'] = pars_js[pairs_ij[0,0]]['B'] # subpop i for all

            for idx in range(pairs_ij.shape[0]):
                i, j = pairs_ij[idx,0], pairs_ij[idx,1]

                pars_js[j] = rotate_latent_bases_obs(sub_pops[i], 
                    sub_pops[j], pars_js[i], pars_js[j],overwrite=True)

                pars_est['D'][sub_pops[i],:] = pars_js[i]['D']
                pars_est['D'][sub_pops[j],:] = pars_js[j]['D']
                pars_est['C'][sub_pops[i],:] = pars_js[i]['C']
                pars_est['C'][sub_pops[j],:] = pars_js[j]['C']                

        elif method[10:16]=='_reach':

            pars_est['D'][sub_pops[0],:] = pars_js[0]['D']
            pars_est['C'][sub_pops[0],:] = pars_js[0]['C']

            for j in range(1, num_sub_pops):

                pars_js[j] = rotate_latent_bases_reach(sub_pops[0], 
                    sub_pops[j], pars_js[0], pars_js[j],overwrite=True)

                pars_est['D'][sub_pops[j],:] = pars_js[j]['D']
                pars_est['C'][sub_pops[j],:] = pars_js[j]['C']   

    return pars_est, n_est  

def rotate_latent_bases_obs(sub_pop_i,sub_pop_j,pars_i,pars_j,overwrite=False):

    n = pars_i['C'].shape[1]
    assert n == pars_j['C'].shape[1]

    overlap = np.intersect1d(sub_pop_i, sub_pop_j)
    ov_i = idx_global2local(overlap,sub_pop_i)
    ov_j = idx_global2local(overlap,sub_pop_j)

    Oij_i = observability_mat((pars_i['A'], pars_i['C'][ov_i,:]),n)
    Oij_j = observability_mat((pars_j['A'], pars_j['C'][ov_j,:]),n)

    Mji = np.linalg.pinv(Oij_j).dot(Oij_i)
    Mij = np.linalg.inv(Mji)

    pars_out = pars_j if overwrite else pars_j.copy()

    pars_out['C'] = pars_out['C'].dot(Mji)
    pars_out['A'] = Mij.dot(pars_j['A']).dot(Mji) if not pars_j['A'] is None else pars_i['A'].copy()
    pars_out['B'] = Mij.dot(pars_j['B']) if not pars_j['B'] is None else pars_i['B'].copy()

    return pars_out    

def rotate_latent_bases_reach(sub_pop_i,sub_pop_j,pars_i,pars_j,overwrite=False):

    n = pars_i['C'].shape[1]
    assert n == pars_j['C'].shape[1]

    Cij_j = reachability_mat((pars_j['A'], pars_j['B']), n)
    Cij_i = reachability_mat((pars_i['A'], pars_i['B']), n)

    Mji = Cij_j.dot(np.linalg.pinv(Cij_i))
    Mij = np.linalg.inv(Mji)

    pars_out = pars_j if overwrite else pars_j.copy()

    pars_out['C'] = pars_out['C'].dot(Mji)
    pars_out['A'] = Mij.dot(pars_j['A']).dot(Mji) if not pars_j['A'] is None else pars_i['A'].copy()
    pars_out['B'] = Mij.dot(pars_j['B']) if not pars_j['B'] is None else pars_i['B'].copy()

    return pars_out


def traverse_subpops(sub_pops, idx_overlap, i_start=0):

    num_sub_pops = len(sub_pops)
    sub_pops_todo = np.ones(num_sub_pops,dtype=bool)

    return collect_children(sub_pops_todo, idx_overlap, i_start)

def collect_children(sub_pops_todo, idx_overlap, i):

    sub_pops_todo[i] = False
    js = find_overlaps(i, idx_overlap, sub_pops_todo)

    if js.size > 0:
        js = js[np.in1d(js,np.where(sub_pops_todo)[0])]
        sub_pops_todo[js] = False
        idx = np.hstack((i * np.ones(js.shape,dtype=int), js))
        for j in js:
            idx = np.vstack((idx,collect_children(sub_pops_todo,idx_overlap,j)))
    else:
        idx = np.array([],dtype=int).reshape(0,2)


    return idx


def find_first_overlap(i, idx_overlap, sub_pops_todo):

    for j in np.where(sub_pops_todo)[0]:                
        for idx in range(len(idx_overlap)):
            if i in idx_overlap[idx] and j in idx_overlap[idx]:
                return j, idx_overlap[idx]

    raise Exception('found no overlap with any other subpopulation')

def find_overlaps(i, idx_overlap, sub_pops_todo):

    overlaps = []
    for j in np.where(sub_pops_todo)[0]:                
        for idx in range(len(idx_overlap)):
            if i in idx_overlap[idx] and j in idx_overlap[idx]:
                overlaps.append(j)
                break
    return np.array(overlaps,dtype=int)[:,np.newaxis]



def idx_global2local(overlap, sub_pop):

    idxi = range(overlap.size) 
    return np.array([np.where(overlap[i] == sub_pop)[0][0] for i in idxi])


def data_reconstruction_MSE(data, data_est):

    return np.mean( (data-data_est)**2, axis=0 )


def get_obs_index_groups(obs_scheme,p):
    """ INPUT:
        obs_scheme: observation scheme for given data, stored in dictionary
                    with keys 'sub_pops', 'obs_time', 'obs_pops'
        p:          dimensionality of observed variables y
    Computes index groups for given observation scheme. 

    """
    try:
        sub_pops = obs_scheme['sub_pops'];
        obs_pops = obs_scheme['obs_pops'];
    except:
        raise Exception(('provided obs_scheme dictionary does not have '
                         'the required fields sub_pops and obs_pops.'))        

    J = np.zeros((p, len(sub_pops))) # binary matrix, each row gives which 
    for i in range(len(sub_pops)):      # subpopulations the observed variable
        if sub_pops[i].size > 0:        # y_i is part of
            J[sub_pops[i],i] = 1   

    twoexp = np.power(2,np.arange(len(sub_pops))) # we encode the binary rows 
    hsh = np.sum(J*twoexp,1)                     # of J using binary numbers

    lbls = np.unique(hsh)         # each row of J gets a unique label 
                                     
    idx_grp = [] # list of arrays that define the index groups
    for i in range(lbls.size):
        idx_grp.append(np.where(hsh==lbls[i])[0])

    obs_idx = [] # list of arrays giving the index groups observed at each
                 # given time interval
    for i in range(len(obs_pops)):
        obs_idx.append([])
        for j in np.unique(hsh[np.where(J[:,obs_pops[i]]==1)]):
            obs_idx[i].append(np.where(lbls==j)[0][0])            
    # note that we only store *where* the entry was found, i.e. its 
    # position in labels, not the actual label itself - hence we re-defined
    # the labels to range from 0 to len(idx_grp)

    return obs_idx, idx_grp


def get_obs_index_overlaps(idx_grp, sub_pops):
    """ returns
        overlap_grp - list of index groups found in more than one subpopulation
        idx_overlap - list of subpopulations in which the corresponding index group is found

    """
    num_sub_pops = len(sub_pops) if isinstance(sub_pops, (list,tuple)) else subs_pops.size
    num_idx_grps = len(idx_grp)

    idx_overlap = []
    idx = np.zeros(num_idx_grps, dtype=int)
    for j in range(num_idx_grps):
        idx_overlap.append([])
        for i in range(num_sub_pops):
            if np.any(np.intersect1d(sub_pops[i], idx_grp[j])):
                idx[j] += 1
                idx_overlap[j].append(i)
        idx_overlap[j] = np.array(idx_overlap[j])

    overlaps = [idx_grp[i] for i in np.where(idx>1)[0]]
    overlap_grp = [i for i in np.where(idx>1)[0]]
    idx_overlap = [idx_overlap[i] for i in np.where(idx>1)[0]]

    return overlaps, overlap_grp, idx_overlap


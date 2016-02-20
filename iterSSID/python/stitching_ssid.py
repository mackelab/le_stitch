import numpy as np
import matplotlib.pyplot as plt
import warnings
import control
from scipy.linalg import solve_discrete_lyapunov as dlyap

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

    params = []
    broken = False

    SIGfp_new, SIGyy_new = generateCovariancesFP(seq=seq, hS=n)

    lnPSIGfp, lnPSIGyy = np.isnan(SIGfp_new),np.isnan(SIGyy_new) # for indexing
    lPSIGfp, lPSIGyy = np.invert(lnPSIGfp), np.invert(lnPSIGyy) 
    PSIGfp, PSIGyy = SIGfp_new[lPSIGfp], SIGyy_new[lPSIGyy]  # observed parts

    # 'zero'-th iteration: used ssidSVD on Hankel cov matrix with NaN set to 0:
    SIGfp_new[lnPSIGfp] = 0
    SIGyy_new[lnPSIGyy] = 0
    params.append(post_process(ssidSVD(SIGfp=SIGfp_new,
                                       SIGyy=SIGyy_new,
                                       n=n,
                                       pi_method=pi_method)))


    # main iterations: fill in NAN's iteratively using missing-value SVD:
    SIGfp_old = SIGfp_new.copy() if init is None else init
    SIGyy_old = SIGyy_new.copy()

    for t in range(1,num_iter+1):

        params.append(post_process(ssidSVD(SIGfp=SIGfp_old,
                                           SIGyy=SIGyy_old,
                                           n=n,
                                           pi_method=pi_method,
                                           params_old=params[-1])))

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
    
    UU,SS,VV = np.linalg.svd(SIGfp) # SIGfp = UU.dot(diag(SS).dot(VV)

    VV = VV.T
    SS = np.diag(SS[:n])
    UU = UU[:,:n]
    VV = VV[:,:n]
   
    Obs = np.dot(UU,SS)
    A = np.linalg.lstsq(Obs[:-p,:],Obs[p:,:])[0]
    C = Obs[:p,:]
    Chat = VV[:p,:n]
    

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
        warnings.warn(('Will not solve DARE, using heuristics; this might '
            'lead to poor estimates of Q and Q0'))  
        Pi = np.linalg.lstsq(A,np.dot(Chat.T,np.linalg.pinv(C.T)))[0]
                   
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

    if isinstance(params['Q'], complex):
        params['Q'] = np.real(params['Q'])
        
    params['Q'] = (params['Q'] + params['Q'].T)/2
    
    if np.min(np.linalg.eig(params['Q'])[0]) < 0:
        
        a,b = np.linalg.eig(params['Q'])
        params['Q'] = a*np.max(b,10**-10)*a.T
    
    if 'Q0' not in params:
        params['Q0'] = np.real(dlyap(params['A'],params['Q']))
    
    params.update({'x0':np.zeros(n)}) 
    params['R']  = np.diag(np.diag(params['R']))

    return params

def add_d(data, params):

    params['d'] = np.mean(data,axis = 0)

    return params

def symmetrize(A):
	
    return (A + A.T)/2


import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided
from ssidid import progprint_xrange

###########################################################################
# Utility (general)
###########################################################################

def symmetrize(A):
    
    return (A + A.T)/2

def blockarray(*args,**kwargs):
    "Taken from Matthew J. Johnson's 'pybasicbayes' code package"
    return np.array(np.bmat(*args,**kwargs),copy=False)

def soft_impute(X, n, eps = 0, max_iter = 500, P_sig=None):

    nan_m = np.asarray(np.isnan(X),dtype=np.float)
    P_sig_X = X.copy()
    P_sig_X[np.isnan(P_sig_X)] = 0

    Z = np.zeros(X.shape)
    for i in range(max_iter):
        U,s,V = np.linalg.svd(P_sig_X + (Z * nan_m))
        Z = U[:,:n].dot(np.diag(s[:n])).dot(V[:n,:])

    return Z

def matrix_power_derivative(A,m,i,j):
    
    n = A.shape[0]
    J_ij = np.zeros((n,n))
    J_ij[i,j] = 1
                    
    ddA = np.zeros((n,n))
    
    for r in range(m):
        ddA += np.linalg.matrix_power(A,r).dot(J_ij).dot( \
            np.linalg.matrix_power(A,m-r-1))
    return ddA    



###########################################################################
# Utility (SSID-specific)
###########################################################################



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

def xx_Hankel_cov_mat(A,Pi,k,l):
    "matrix with blocks cov(x_t+m, x_t) m = 1, ..., k+l-1 on the anti-diagonal"
    
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

###########################################################################
# constructing Hankel matrices
###########################################################################

def yy_Hankel_cov_mat(C,A,Pi,k,l,Om=None,linear=True):
    "matrix with blocks cov(y_t+m, y_t) m = 1, ..., k+l-1 on the anti-diagonal"
    
    p,n = C.shape
    if linear:
        assert n == A.shape[1] and n == Pi.shape[0] and n == Pi.shape[1]
    else:
        assert (k+l)*n == A.shape[0] and n == A.shape[1]
        
    assert (Om is None) or (Om.shape == (p,p))
    if not Om is None:
        Om_idx = np.asarray(Om, dtype=float)
        Om_idx[~Om] = np.nan

    H = np.empty((k*p, l*p))
    
    for kl_ in range(k+l-1):        

        if linear:
            AmPi = np.linalg.matrix_power(A,kl_+1).dot(Pi) 
        else:
            AmPi = A[(kl_+1)*n:(kl_+2)*n, :]

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

# large-scale problems may require solutions to be computed in smaller chunks

def chunking_blocks(f, a, b, max_size):

    out = 0.
    size_a, size_b = a.size, b.size
    max_i, max_j = int(np.ceil(size_a/max_size)), int(np.ceil(size_b/max_size))

    for i in range(max_i):
        idx_i = range(i*max_size, np.minimum((i+1)*max_size, size_a)) 
        for j in range(max_j):
            idx_j = range(j*max_size, np.minimum((j+1)*max_size, size_a))
            out += f(idx_i, idx_j, i, j)
            
    return out

def comp_model_covariances(pars, lag_range, 
    mmap=False, chunksize=None, data_path='../fits/', verbose=False):
    "returns list of time-lagged covariances cov(y_t+m, y_t) m = 1, ..., k+l-1"
    
    kl = len(lag_range)
    p = pars['C'].shape[0]
    
    chunksize = p if chunksize is None else chunksize

    max_i = p//chunksize
    assert np.allclose(max_i * chunksize, p) 
        
    Qs = [None]
    for m in kl:
        m_ = lag_range[m]
        if mmap:
            Qs.append(np.memmap(data_path+'Qs_'+str(m_), dtype=np.float, 
                mode='w+', shape=(p,p)))
            if verbose: 
                print('computing time-lagged covariance for lag m =', str(m))      
        else:
            Qs.append(np.empty((p,p)))

        APi = np.linalg.matrix_power(pars['A'], m_).dot(pars['Pi'])
        for i in range(max_i):
            idx_i  = range(i*chunksize, (i+1)*chunksize)
            for j in range(max_i):
                idx_j = range(j*chunksize, (j+1)*chunksize)
                Qs[m][np.ix_(idx_i,idx_j)] = pars['C'][idx_i,:].dot( \
                    APi ).dot(pars['C'][idx_j,:].T) 
                if mmap:
                    del Qs[m]
                    Qs.append(np.memmap(data_path+'Qs_'+str(m_), 
                        dtype=np.float, mode='r+', shape=(p,p)))
        if mmap:
            del Qs[m]
            Qs.append(np.memmap(data_path+'Qs_'+str(m_), dtype=np.float, 
                mode='r', shape=(p,p)))

        if lag_range[m] == 0:
            Qs[m][range(p),range(p)] += pars['R']

    return Qs


def gen_data(p,n,lag_range,T,nr,eig_m_r, eig_M_r, eig_m_c, eig_M_c, 
             mmap, chunksize, data_path, idx_a=None, idx_b=None, snr=(.75, 1.25),
             verbose=False, whiten=False):

    nr = n if nr is None else nr
    nc, nc_u = n - nr, (n - nr)//2
    assert nc_u * 2 == nc 

    idx_a = np.arange(p) if idx_a is None else idx_a
    idx_b = idx_a if idx_b is None else idx_b
    assert np.all(idx_a == np.sort(idx_a))
    assert np.all(idx_a == np.sort(idx_a))
    pa, pb = len(idx_a), len(idx_b)

    kl = len(lag_range)
    kl_ = np.max(lag_range)+1
    ev_r = np.linspace(eig_m_r, eig_M_r, nr)
    ev_c = np.exp(2 * 1j * np.pi * np.random.vonmises(mu=0, kappa=1000, size=nc_u))
    ev_c = np.linspace(eig_m_c, eig_M_c, (n - nr)//2) * ev_c

    pars_true,Qs,_ = gen_sys(p=p,n=n,lag_range=lag_range, nr=nr,ev_r=ev_r,ev_c=ev_c,
                             snr=snr, calc_stats=T==np.inf,return_masked=False,
                             mmap=mmap,chunksize=chunksize,data_path=data_path,
                             whiten=whiten)
    pars_true['d'], pars_true['mu0'] = np.zeros(p), np.zeros(n), 
    pars_true['V0'] = pars_true['Pi'].copy()


    if T == np.inf:
        x,y = np.zeros((n,0)), np.zeros((p,0))
    else:
        x,y = draw_data(pars=pars_true, T=T, 
                        mmap=mmap, chunksize=chunksize, data_path=data_path)

    return pars_true, x, y, idx_a, idx_b

def gen_sys(p,n,lag_range,nr=None,ev_r=None,ev_c=None,snr=(.75, 1.25),
            calc_stats=True,return_masked=True, whiten=False,
            mmap=False, chunksize=None,data_path='../fits'):

    kl = len(lag_range)
    pars_true = gen_pars(p,n,nr=nr,ev_r=ev_r,ev_c=ev_c,snr=snr,whiten=whiten)
    if calc_stats:
        Qs_full = comp_model_covariances(pars_true, lag_range, mmap=mmap, 
            chunksize=chunksize,data_path=data_path)
        if return_masked:
            Qs = comp_model_covariances(pars_true, lag_range, mmap=mmap, 
                chunksize=chunksize,data_path=data_path) 
        else: 
            Qs = Qs_full
    else:
        Qs, Qs_full = [], []
        for m in range(kl):
            Qs.append(None)
            Qs_full.append(None)

    return pars_true, Qs, Qs_full


def gen_pars(p,n, nr=None, ev_r = None, ev_c = None, 
             snr = (.75, 1.25), whiten=False):
    "draws parameters for an LDS"

    # generate dynamics matrix A

    nr = n if nr is None else nr
    nc, nc_u = n - nr, (n - nr)//2
    assert nc_u * 2 == nc 

    if not ev_r is None:
        assert ev_r.size == nr

    if not ev_c is None:
        assert ev_c.size == nc_u

    Q, D = np.zeros((n,n), dtype=complex), np.zeros(n, dtype=complex)

    # draw real eigenvalues and eigenvectors
    D[:nr] = np.linspace(0.8, 0.99, nr) if ev_r is None else ev_r 
    Q[:,:nr] = np.random.normal(size=(n,nr))
    Q[:,:nr] /= np.sqrt((Q[:,:nr]**2).sum(axis=0)).reshape(1,nr)

    # draw complex eigenvalues and eigenvectors
    if ev_c is None:
        circs = np.exp(2 * 1j * np.pi * np.random.vonmises(mu=0, kappa=1000, size=nc_u))
        scales = np.linspace(0.5, 0.9, nc_u)
        ev_c_r, ev_c_c = scales * np.real(circs), scales * np.imag(circs)
    else:
        ev_c_r, ev_c_c = np.real(ev_c), np.imag(ev_c) 
    V = np.random.normal(size=(n,n))
    for i in range(nc_u):
        Vi = V[:,i*2:(i+1)*2] / np.sqrt( np.sum(V[:,i*2:(i+1)*2]**2) )
        Q[:,nr+i], Q[:,nr+nc_u+i] = Vi[:,0]+1j*Vi[:,1], Vi[:,0]-1j*Vi[:,1] 
        D[nr+i], D[nr+i+nc_u] = ev_c_r[i]+1j*ev_c_c[i], ev_c_r[i]-1j*ev_c_c[i]

    A = Q.dot(np.diag(D)).dot(np.linalg.inv(Q))
    assert np.allclose(A, np.real(A))
    A = np.real(A)

    # generate innovation noise covariance matrix Q

    Q = np.atleast_2d(stats.wishart(5*n, np.eye(n)).rvs()/n)
    Pi = np.atleast_2d(sp.linalg.solve_discrete_lyapunov(A, Q))

    L = np.linalg.cholesky(Pi)
    Linv = np.linalg.inv(L)
    A, Q = Linv.dot(A).dot(L), Linv.dot(Q).dot(Linv.T)
    Pi = np.atleast_2d(sp.linalg.solve_discrete_lyapunov(A, Q))

    # generate emission-related matrices C, R

    C = np.random.normal(size=(p,n)) / np.sqrt(n)
    NSR = np.random.uniform(size=p, low=snr[0], high=snr[1]) # 1/SNR
    if whiten:
        C /= np.atleast_2d(np.sqrt(np.sum(C*C.dot(Pi), axis=1) * (1 + NSR))).T

    R = np.sum(C*C.dot(Pi), axis=1) * NSR

    try:
        B = np.linalg.cholesky(Pi)
    except:
        B = np.nan * np.ones((n,n))
        
    return { 'A': A, 'B': B, 'Q': Q, 'Pi': Pi, 'C': C, 'R': R }


def draw_data(pars,T, mmap=False, chunksize=None, data_path='../fits/'):
    "cythonise me!"

    p,n = pars['C'].shape
    chunksize = p if chunksize is None else chunksize
    max_i = int(np.ceil(p/chunksize))

    def chunk_range(max_i):
        if p > 1000:
            return progprint_xrange(max_i, perline=100)
        else:
            return range(max_i)

    # start with noise terms
    L = np.linalg.cholesky(pars['Q'])
    x = np.random.normal(size=(T,n))
    x = x.dot(L.T)

    # step thourhg latent dynamics
    x[0,:]  = pars['mu0'].copy() 
    x[0,:] += np.linalg.cholesky(pars['V0']).dot(np.random.normal(size=n))
    for t in range(1,T):
        x[t,:] += pars['A'].dot(x[t-1,:])

    # do emissions
    L = np.sqrt(pars['R'])
    if mmap:
        y = np.memmap(data_path+'y', dtype=np.float, mode='w+', shape=(T,p))
    else:
        y = np.empty(shape=(T,p))

    for i in chunk_range(max_i):
        idx_i = range(i*chunksize, np.minimum((i+1)*chunksize, p)) 
        y[:,idx_i] = np.random.normal(size=(T, len(idx_i)))*np.atleast_2d(L[idx_i]) \
                    + x.dot(pars['C'][idx_i,:].T)
        if mmap:
            del y # releases RAM, forces flush to disk
            y = np.memmap(data_path+'y', dtype=np.float, mode='r+', shape=(T,p))
    if mmap:
        del y # releases RAM, forces flush to disk
        y = np.memmap(data_path+'y', dtype=np.float, mode='r', shape=(T,p))

    return x, y


###########################################################################
# Utility (stitching-specific)
###########################################################################

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
        idx_overlap - list of subpopulations in which the corresponding index 
                      group is found

    """
    num_sub_pops = len(sub_pops) 
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

def get_subpop_stats(sub_pops, p, obs_pops=None, verbose=False):
    "computes a collection of helpful index sets for the stitching context"

    if obs_pops is None:
        obs_pops = tuple(range(len(sub_pops)))
    obs_idx, idx_grp = get_obs_index_groups(obs_scheme={'sub_pops': sub_pops,
        'obs_pops': obs_pops},p=p)
    overlaps, overlap_grp, idx_overlap = get_obs_index_overlaps(idx_grp, \
        sub_pops)

    def co_observed(x, i):
        for idx in obs_idx:
            if x in idx and i in idx:
                return True
        return False        

    num_idx_grps, co_obs = len(idx_grp), []
    for i in range(num_idx_grps):    
        co_obs.append([idx_grp[x] for x in np.arange(len(idx_grp)) \
            if co_observed(x,i)])
        co_obs[i] = np.sort(np.hstack(co_obs[i]))

    if verbose:
        print('idx_grp:', idx_grp)
        print('obs_idx:', obs_idx)
        
    Om, Ovw, Ovc = comp_subpop_index_mats(sub_pops,idx_grp,overlap_grp,\
        idx_overlap)    
    
    if verbose:
        plt.figure(figsize=(20,10))
        plt.subplot(1,3,1)
        plt.imshow(Om,interpolation='none')
        plt.title('Observation pattern')
        plt.subplot(1,3,2)
        plt.imshow(Ovw,interpolation='none')
        plt.title('Overlap pattern')
        plt.subplot(1,3,3)
        plt.imshow(Ovc,interpolation='none')
        plt.title('Cross-overlap pattern')
        plt.show()
        
    return obs_idx,idx_grp,co_obs,overlaps,overlap_grp,idx_overlap,Om,Ovw,Ovc

def comp_subpop_index_mats(sub_pops,idx_grp,overlap_grp,idx_overlap):
    "return masks for observed, overlapping and cross-overlapping matrix parts"

    p = np.max([np.max(sub_pops[i]) for i in range(len(sub_pops))]) + 1

    if p < 10000:
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
                Ovc[np.ix_(idx_grp[overlap_grp[i]], \
                    sub_pops[idx_overlap[i][j]])] = True
                Ovc[np.ix_(sub_pops[idx_overlap[i][j]], \
                    idx_grp[overlap_grp[i]])] = True
                Ovc[np.ix_(idx_grp[overlap_grp[i]], \
                    idx_grp[overlap_grp[i]])] = False

    else:
        Om, Ovw, Ovc = None, None, None
    
    return Om, Ovw, Ovc

# a few functions for ordering subpopulations by overlap:

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
            idx=np.vstack((idx,collect_children(sub_pops_todo,idx_overlap,j)))
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




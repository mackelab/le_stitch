import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import warnings
import control
from scipy.linalg import solve_discrete_lyapunov as dlyap
from numpy.lib.stride_tricks import as_strided


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
    pars_js, X0_est_js, M_js = [],[],[] 
    n_est_js = np.zeros(num_sub_pops,dtype=int)
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
    if pars_j['A'] is None:
        pars_out['A'] = pars_i['A'].copy()
    else:
        pars_out['A'] = Mij.dot(pars_j['A']).dot(Mji) 
    if pars_j['B'] is None: 
        pars_out['B'] = pars_i['B'].copy()
    else:
        pars_out['B'] = Mij.dot(pars_j['B']) 

    return pars_out    

def rotate_latent_bases_reach(sub_pop_i,sub_pop_j,pars_i,pars_j,
        overwrite=False):

    n = pars_i['C'].shape[1]
    assert n == pars_j['C'].shape[1]

    Cij_j = reachability_mat((pars_j['A'], pars_j['B']), n)
    Cij_i = reachability_mat((pars_i['A'], pars_i['B']), n)

    Mji = Cij_j.dot(np.linalg.pinv(Cij_i))
    Mij = np.linalg.inv(Mji)

    pars_out = pars_j if overwrite else pars_j.copy()

    pars_out['C'] = pars_out['C'].dot(Mji)
    if pars_j['A'] is None:
        pars_out['A'] = pars_i['A'].copy()
    else:
        pars_out['A'] = Mij.dot(pars_j['A']).dot(Mji) 
    if pars_j['B'] is None: 
        pars_out['B'] = pars_i['B'].copy()
    else:
        pars_out['B'] = Mij.dot(pars_j['B'])

    return pars_out


def data_reconstruction_MSE(data, data_est):

    return np.mean( (data-data_est)**2, axis=0 )

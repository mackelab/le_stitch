import numpy as np
import scipy as sp
from scipy import stats
import numbers       # to check for numbers.Integral, for isinstance(x, int)

import matplotlib
import matplotlib.pyplot as plt
from IPython import display  # for live plotting in jupyter

from scipy.io import savemat # store intermediate results 


"""
TO DO: kalman_smoother urgently needs help with memory management. 
       If we retain the convergence of latent covariance matrices check,
       then we really should re-index the arrays to not have all those zeros
       in there.  
       In principle, all we need for the M-step are partial sums over E[x_t],
       E[x_t x_t'], E[x_t, x_{t-1]'] during the individual observation 
       intervals for each subpopulation, not the full traces for each 
       individual t. 
"""

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def fit_lds(y, 
            u,
            x_dim,            
            obs_scheme,
            pars,
            max_iter=100,
            epsilon=0, 
            eps_cov=0,
            plot_flag=True, 
            trace_pars_flag=False, 
            trace_stats_flag=False,
            diag_R_flag=True,
            use_A_flag=True, 
            use_B_flag=False,
            save_file=None):

    """ INPUT:
        y:         data array of observed variables
        u:         data array of input variables
        x_dim:      dimensionality of (sole subgroup of) latent state X
        obs_scheme: observation scheme for given data, stored in dictionary
                   with keys 'sub_pops', 'obs_time', 'obs_pops'
        pars:      dictionary of parameters to start fitting. 
        max_iter:   maximum allowed iterations for iterative fitting (e.g. EM)
        epsilon:   convergence criterion, e.g. difference of log-likelihoods
        plot_flag: boolean, specifying if fitting progress is visualized
        trace_pars_flag:  boolean, specifying if entire parameter updates 
                           history or only the current state is kept track of 
        trace_stats_flag: boolean, specifying if entire history of inferred  
                           latents or only the current state is kept track of 
        diag_R_flag: boolean, specifying diagonality of observation noise
        eps_cov: convergence criterion for posterior covariances
        use_A_flag:    boolean, specifying whether or not to fit parameter A
        use_B_flag:    boolean, specifying whether or not to use parameter B
        save_file : (path to folder and) name of file for storing results.  
        Fits an LDS model to data.

    """
    # stats is a container for statistics derived for the time series, from
    #  simple dimensionalities to inferred latent states for each time point
    stats = _setup_stats(y=y,
                         x_dim=x_dim,
                         u_dim=_get_u_dim(u, use_B_flag))

    if stats['u_dim'] == 0:
        u = None

    # check observation scheme
    # The observation scheme is crucial to both the sitching context and to
    # any missing data in y! One should always check if the provided 
    # observation scheme is the intended one.
    obs_scheme = dict(obs_scheme) # deep copy
    check_obs_scheme(obs_scheme=obs_scheme,
                     y_dim=stats['y_dim'],
                     t_tot=stats['t_tot'])

    # check provided initial parameters
    pars = dict(pars)
    check_pars(pars=pars,
               x_dim=x_dim,
               y_dim=stats['y_dim'],
               u_dim=stats['u_dim'])

    # check selected fitting options 
    check_options(max_iter=max_iter,
                  epsilon=epsilon,
                  eps_cov=eps_cov,
                  plot_flag=plot_flag,
                  trace_pars_flag=trace_pars_flag,
                  trace_stats_flag=trace_stats_flag,
                  diag_R_flag=diag_R_flag,
                  use_A_flag=use_A_flag,
                  use_B_flag=use_B_flag,
                  save_file=save_file)     

    # imprint all fitting options and data structures onto E- and M-step
    # generate e_step(stats,pars)
    e_step = _wrap_lds_e_step(y=y,
                              u=u,
                              obs_scheme=obs_scheme,
                              eps=eps_cov) 
    # generate e_step(pars)
    m_step = _wrap_lds_m_step(y=y,
                              u=u,
                              obs_scheme=obs_scheme,
                              use_A_flag=use_A_flag,
                              use_B_flag=use_B_flag,
                              diag_R_flag=diag_R_flag)
    # generate em_iterate(pars, save_file))
    em_iterate = _wrap_lds_em_iterate(e_step=e_step,
                                      m_step=m_step,
                                      max_iter=max_iter,
                                      epsilon=epsilon,
                                      plot_flag=plot_flag,
                                      trace_pars_flag=trace_pars_flag,
                                      trace_stats_flag=trace_stats_flag)


    # evaluate initial state       
    print('convergence criterion for E-step (tolerance on matrix changes):')
    print(eps_cov)

    pars, ll = em_iterate(pars=pars, save_file=save_file)

    print('finished EM')

    return pars, ll


def _setup_stats(y, x_dim, u_dim):

    stats = {
             't_tot': y.shape[1], 
             'n_tr':  y.shape[2], 
             'y_dim': y.shape[0],
             'u_dim': u_dim,
             'x_dim': x_dim
             }

    return stats


def _get_u_dim(u, use_B_flag=None):

    if use_B_flag is None:
        if (isinstance(u, np.ndarray) and len(u.shape)==3
            and u.shape[1]>0 and u.shape[2]>0):
            u_dim = u.shape[0]
        else:
            u_dim = 0
    # else: use_B_flag has to be a boolean, assuming input was checked properly
    elif not use_B_flag:
        u_dim = 0
    elif (isinstance(u, np.ndarray) and len(u.shape)==3
          and u.shape[1]>0 and u.shape[2]>0):
        u_dim = u.shape[0]   
    # else: any other case is not allowed:
    else:                            
        if isinstance(u, np.ndarray):
            raise Exception(('If provided, input data u has to be an array '
                             'with three dimensions, (u_dim,t_tot,n_tr). To '
                             'not include any input data, set use_B_flag = '
                             'False. The shape of provided u is '), u.shape)
        else:
            raise Exception(('If provided, input data u has to be an array '
                             'with three dimensions, (u_dim,t_tot,n_tr). To '
                             'not include any input data, set use_B_flag = '
                             'False. The provided u is '), u)
    return u_dim


def _wrap_lds_em_iterate(e_step,m_step,max_iter,epsilon,
                         plot_flag,trace_pars_flag,trace_stats_flag):

    def em_iterate(pars,save_file=None):

        stats, lltr, t_conv_ft, t_conv_sm = e_step(pars=pars)    

        ll_new = np.sum(lltr) # discarding distinction between trials
        ll_old = -float('Inf')

        dll = []              # performance trace for status plotting
        log10 = np.log(10)    # for convencience, see below
        lls = [ll_new.copy()] # performance trace to be returned


        # start EM iterations, run until convergence 
        _trace_pars(pars=pars, trace_pars_flag=trace_pars_flag)
        _trace_stats(stats=stats, trace_stats_flag=trace_stats_flag)
        

        step_count = 0       
        conv_flag = False

        while not conv_flag:

            ll_old = ll_new
            step_count += 1            

                    
            pars = m_step(stats=stats)            
            

            stats, lltr, t_conv_ft, t_conv_sm = e_step(pars=pars)        

            ll_new = np.sum(lltr) # discarding distinction between trials
            lls.append(ll_new.copy())


            # store intermediate results for each time step
            _trace_pars(pars=pars, trace_pars_flag=trace_pars_flag)
            _trace_stats(stats=stats, trace_stats_flag=trace_stats_flag)
            if not save_file is None:
                np.savez(save_file+'_tmp_'+str(step_count),
                         step_count,lltr,  
                         pars,
                         eps_cov,t_conv_ft,t_conv_sm)

            dll.append(ll_new - ll_old)
            if plot_flag:
                # dynamically plot log of log-likelihood difference
                _plot_ll_progress(lls=lls, dll=dll)
                
            conv_flag = not (ll_new-ll_old>epsilon and (step_count<max_iter))

            if ll_new < ll_old and step_count < max_iter:
                # special case: should actually not happen!
                conv_flag = _decide_on_ill_conv(ll_new=ll_new,
                                                ll_old=ll_old,
                                                conv_flag=conv_flag)

        return pars, np.array(lls)

    return em_iterate 


def _decide_on_ill_conv(ll_new, ll_old, conv_flag):

    #print('ll_new - ll_old')
    #print( ll_new - ll_old )
    #print(('WARNING! Lower bound decreased during EM '
    #       'algorithm. This is impossible for an LDS'))
    #print('Will do another round (probably osciilation?)')

    return False


def _trace_pars(pars, trace_pars_flag):

    if trace_pars_flag:
        # check if called for the first time:
        if not 'As' in pars:
            pars['As']   = [pars['A'].copy()]
            pars['Bs']   = [pars['B'].copy()]
            pars['Qs']   = [pars['Q'].copy()]
            pars['mu0s'] = [pars['mu0'].copy()]
            pars['V0s']  = [pars['V0'].copy()]
            pars['Cs']   = [pars['C'].copy()]
            pars['ds']   = [pars['d'].copy()]
            pars['Rs']   = [pars['R'].copy()]
        else: 
            pars['As'].append(pars['A'].copy())
            pars['Bs'].append(pars['B'].copy())
            pars['Qs'].append(pars['Q'].copy())
            pars['mu0s'].append(pars['mu0'].copy())
            pars['V0s'].append(pars['V0'].copy())
            pars['Cs'].append(pars['C'].copy())
            pars['ds'].append(pars['d'].copy())
            pars['Rs'].append(pars['R'].copy())   


def _trace_stats(stats, trace_stats_flag):

    if trace_stats_flag:
        # check if called for the first time:
        if not 'exts' in stats:
            stats['exts']    = [stats['ext'].copy()]
            stats['extxts']   = [stats['extxt'].copy()]
            stats['extxtm1s'] = [stats['extxtm1'].copy()]
        else:
            stats['exts'].append(stats['ext'].copy())
            stats['extxts'].append(stats['extxt'].copy())
            stats['extxtm1s'].append(stats['extxtm1'].copy())



def _plot_ll_progress(lls, dll):

    plt.clf()
    plt.figure(1,figsize=(15,15))
    plt.subplot(1,2,1)
    plt.plot(lls)
    plt.xlabel('#iter')
    plt.ylabel('log-likelihood')
    plt.subplot(1,2,2)
    plt.plot(dll[-50:])
    plt.xlabel('#iter')
    plt.ylabel('ll_{new} - ll_{old}')
    display.display(plt.gcf())
    display.clear_output(wait=True)



"""                             E-step                                      """



def _check_mat_conv_MSE(A, B, eps=0):        

    if np.mean(np.power(A-B,2)) < eps:
        return True
    else: 
        return False
                             

def _check_mat_conv_empty(A, B, eps=0):        

    return False    


def _wrap_lds_e_step(y,u,obs_scheme,check_mat_conv=_check_mat_conv_MSE,eps=0):

    if eps>0:

        def e_step(pars):

            return lds_e_step(pars=pars,
                              y=y,
                              u=u,
                              obs_scheme=obs_scheme,
                              check_mat_conv=check_mat_conv,
                              eps=eps)

    else:  # no need to check convergence with non-positive stopping criterion

        if not check_mat_conv == _check_mat_conv_empty:
            print(('non-positive stopping criterion for convergence of latent '
                   'covariances.'))
            print(('will not track latent covariance convergence.'))

        def e_step(pars):

            return lds_e_step(pars=pars,
                              y=y,
                              u=u,
                              obs_scheme=obs_scheme,
                              check_mat_conv=_check_mat_conv_empty,
                              eps=eps)

    return e_step


def lds_e_step(pars,y,u,obs_scheme,
               check_mat_conv=_check_mat_conv_MSE,eps=0): 
    """ 
    see Bishop, 'Pattern Recognition and Machine Learning', ch. 13
    for formulas and input/output naming conventions

    """ 
    # create container for relevant (inferred) statistics for the M-step
    stats = _setup_stats(y=y,
                         x_dim=pars['A'].shape[0],
                         u_dim=_get_u_dim(u))

    # pre-compute B*u, as potentially x_dim << u_dim, and the computation is
    # faster when done in a big batch vs. done on each time step individually
    Bu = np.zeros((stats['x_dim'], stats['t_tot'], stats['n_tr']))
    if (isinstance(u, np.ndarray) and u.size>0 
        and u.shape[1]==stats['t_tot'] and u.shape[2]==stats['n_tr']):
        for tr in range(stats['n_tr']):        # i.e. if u is e.g. empty, we 
            Bu[:,:,tr] = np.dot(pars['B'], u[:,:,tr])  # just leave B*u = 0!


    # for the E-step (unlike the M-step), we can nicely seperate the Kalman
    # filter and smoother stages for individual trials and hence do so:
    stats['ext']     = np.empty((stats['x_dim'],                 
                                 stats['t_tot'], stats['n_tr'])) 
    stats['extxt']   = np.empty((stats['x_dim'], stats['x_dim'], 
                                 stats['t_tot'], stats['n_tr']))
    stats['extxtm1'] = np.empty((stats['x_dim'], stats['x_dim'], 
                                 stats['t_tot'], stats['n_tr']))

    # keep the following variables seperate from stats, as they are not
    # strictly needed for the EM-algorithm to operate
    ll = np.empty(stats['n_tr'])
    t_conv_ft = np.empty(( len(obs_scheme['obs_time']), stats['n_tr']))
    t_conv_sm = np.empty(( len(obs_scheme['obs_time']), stats['n_tr']))

    for tr in range(stats['n_tr']):

        stats_tr = {
                    'x_dim': stats['x_dim'],
                    'y_dim': stats['y_dim'],
                    'u_dim': stats['u_dim'],
                    't_tot': stats['t_tot'],
                    'n_tr' : stats['n_tr']
                    }
        t_conv_ft[:,tr] = kalman_filter(stats=stats_tr,
                                        pars=pars,
                                        Bu_tr=Bu[:,:,tr],
                                        y_tr=y[:,:,tr],
                                        obs_scheme=obs_scheme,
                                        check_mat_conv=check_mat_conv,
                                        eps=eps)    
        t_conv_sm[:,tr] = kalman_smoother(stats=stats_tr,
                                          pars=pars,
                                          Bu_tr=Bu[:,:,tr],
                                          obs_time=obs_scheme['obs_time'],
                                          t_conv_ft=t_conv_ft[:,tr],
                                          check_mat_conv=check_mat_conv,
                                          eps=eps)
        kalman_pars2moments(stats=stats_tr,
                            obs_time=obs_scheme['obs_time'],
                            t_conv_ft=t_conv_ft[:,tr],
                            t_conv_sm=t_conv_sm[:,tr])

        ll[tr] = np.sum(stats_tr['logc'],axis=0) # sum over times, get vector

        # set pointer and delete to prevent overwriting and avoid deep copy
        stats['ext'][      :,:,tr] = stats_tr['ext'].copy()     
        stats['extxt'][  :,:,:,tr] = stats_tr['extxt'].copy()   
        stats['extxtm1'][:,:,:,tr] = stats_tr['extxtm1'].copy() 
        del stats_tr['ext'], stats_tr['extxt'], stats_tr['extxtm1']                              # currently deleting twice (see re-declaration of stats_tr above), need to fix this when debugging multi-trial solution
    
    return stats, ll, t_conv_ft, t_conv_sm


def kalman_pars2moments(stats, obs_time, t_conv_ft, t_conv_sm):
    """
    see Bishop, 'Pattern Recognition and Machine Learning', ch. 13
    for formulas and input/output naming conventions  

    """                
    # introduced some heavy indexing to reflect changes to Kalman filter and
    # smoother code that no longer compute poster covariances for every time
    # step invidivually, but check for convergence. Times of convergence are
    # given by t_conv_ft for the (forward) Kalman filter, and t_conv_sm for
    # the (backward) Kalman smoother.  

    stats['ext']     = stats['mu'].copy()           # E[x_t]                                  # check for future versions if these actually need to be initialised as deep copies
    stats['extxt']   = stats['V'].copy()            # E[x_t, x_t]
    stats['extxtm1'] = np.empty(stats['V'].shape)   # E[x_t x_{t-1}'] 
    stats['extxtm1'][:,:,0] = 0

    t = 0
    for i in range(obs_time.size):            
        while t <= t_conv_ft[i]: # before the filter covariances converged
            # in this interval, the filtered covariances kept changing, and
            # thus so did the smoothed covariances. We have to look up all.                
            stats['extxt'][:,:,t] += np.outer(stats['mu'][:,t], 
                                              stats['mu'][:,t])
            t += 1 
        V_hconv = stats['V'][:,:, t_conv_sm[i]]
        while t <= t_conv_sm[i]: # after filter and smoother converged
            # a little confusing, the smoother runs backwards in time and
            # in this middle section hence has already converged. We can
            # precompute V_h * J as they are both constant here.                
            stats['extxt'][:,:,t] = ( V_hconv 
                                    + np.outer(stats['mu'][:,t], 
                                               stats['mu'][:,t]))
            t += 1 
        while t < obs_time[i]:    # after filter, before smoother converged                
            stats['extxt'][:,:,t] += np.outer(stats['mu'][:,t], 
                                              stats['mu'][:,t])
            t += 1 

    t = 1        
    for i in range(obs_time.size):            
        while t <= t_conv_ft[i]: # before the filter covariances converged
            # in this interval, the filtered covariances kept changing, and
            # thus so did the smoothed covariances. We have to look up all.                
            stats['extxtm1'][:,:,t] = ( np.dot(stats['V'][:,:, t], 
                                               stats['J'][:,:,t-1].transpose())
                                      + np.outer(stats['mu'][:,t], 
                                                 stats['mu'][:,t-1]) )
            t += 1 

        Jconv = stats['J'][:,:,t_conv_ft[i]] # J depends only on filter output
        Jconvtr = Jconv.transpose()          # and hence converges with it
        VhJconv = np.dot(stats['V'][:,:, t_conv_sm[i]], Jconvtr)
        while t < t_conv_sm[i]: # after filter and smoother converged
            # a little confusing, the smoother runs backwards in time and
            # in this middle section hence has already converged. We can
            # precompute V_h * J as they are both constant here.                
            stats['extxtm1'][:,:,t] = ( VhJconv 
                                      + np.outer(stats['mu'][:,t], 
                                                 stats['mu'][:,t-1]) )
            t += 1 
        while t < obs_time[i]:    # after filter, before smoother converged     
            stats['extxtm1'][:,:,t] = ( np.dot(stats['V'][:,:, t], 
                                               Jconvtr) 
                                      + np.outer(stats['mu'][:,t], 
                                                 stats['mu'][:,t-1]) )
            t += 1 
        # t == obs_time[i] now
        stats['J'][:,:,t-1] = Jconv.copy() # does nothing if not converged, 
                                           # otherwise gives starting info 
                                           # for next subpopulation



"""                         Kalman filter                                   """



def kalman_filter(stats,pars,Bu_tr,y_tr,obs_scheme,check_mat_conv,eps=0):
    """ 
    see Bishop, 'Pattern Recognition and Machine Learning', ch. 13
    for formulas and input/output naming conventions  
    Implements the Kalman filter for a single trial (single time series)

    """       
    x_range = range(stats['x_dim'])        # convenience variables, 
    Iq      = np.identity(stats['x_dim'])  # will be used repeatedly

    stats['V']    = np.zeros((stats['x_dim'],stats['x_dim'],stats['t_tot']))                   # check for future version if these are still needed to be initialised as zero
    stats['P']    = np.zeros((stats['x_dim'],stats['x_dim'],stats['t_tot']))
    stats['Pinv'] = np.zeros((stats['x_dim'],stats['x_dim'],stats['t_tot'])) 

    # the following can be overwritten if still around form the last trial:
    if (not ('mu' in stats and stats['mu'].shape==(stats['x_dim'],
                                                   stats['t_tot']))):
        stats['mu']   = np.empty((stats['x_dim'],stats['t_tot']))
    if  not ('logc' in stats and stats['mu'].size==(stats['t_tot'])):
        stats['logc'] = np.empty((               stats['t_tot']))

    # estimates of posterior covariances converge fast, so we want to check for
    # this and stop computing them once reasonable
    t_conv_ft = np.array(obs_scheme['obs_time'])-1 # latest possible time
    
    # first time step: [mu0,V0] -> [mu1,V1]
    idx = obs_scheme['sub_pops'][obs_scheme['obs_pops'][0]]
    _kft_init_step(stats=stats,
                   pars=pars,
                   y_tr=y_tr,
                   Bu_tr=Bu_tr, 
                   x_range=x_range, 
                   Iq=Iq,
                   idx=idx)

    # we divide the rest of the time series into intervals in which the same
    # subpopulation is observed, and process them one at a time:    
    t_start = 1 
    for i in range(len(obs_scheme['obs_time'])):
        idx = obs_scheme['sub_pops'][obs_scheme['obs_pops'][i]]
        t_conv_ft[i] = _kft_obs_interval(stats=stats,
                                         pars=pars, 
                                         y_tr=y_tr,
                                         Bu_tr=Bu_tr, 
                                         t_start=t_start, 
                                         t_end=obs_scheme['obs_time'][i], 
                                         x_range=x_range, 
                                         Iq=Iq,
                                         check_mat_conv=check_mat_conv,
                                         eps=eps,
                                         idx=idx)
        t_start = obs_scheme['obs_time'][i]
                                     
    # add constants and pre-factors to log-likelihood
    stats['logc'] = -1/2 * (stats['logc'] + stats['y_dim'] * np.log(2*np.pi))
    
    return t_conv_ft


def _kft_init_step(stats, pars, y_tr, Bu_tr, x_range, Iq, idx):
    """ 
    Implements the Kalman filter for the initial time step 

    """

    if idx.size > 0:
        # pre-compute for this group of observed variables
        Cj   = pars['C'][np.ix_(idx,x_range)]            # all these
        Rj   = pars['R'][idx]                           # operations    
        CtrRinv = Cj.transpose() / Rj                   # are order
        CtrRinvC = np.dot(CtrRinv, Cj)                  # O(y_dim) !  

        # pre-compute for this time step   
        mu0B0  = pars['mu0']+Bu_tr[:,0]                                        
        Cmu0B0 = np.dot(Cj,mu0B0) # O(y_dim)
        yDiff  = y_tr[idx,0] - pars['d'][idx] - Cmu0B0  # O(y_dim)   

        CtrRyDiff_Cmu0 = np.dot(CtrRinv, yDiff)         # O(y_dim)
        P0   = pars['V0'] # np.dot(np.dot(A,V0),Atr)+Q
                
        # compute Kalman gain components
        P0inv   = sp.linalg.inv(P0)                
        Kcore  = sp.linalg.inv(CtrRinvC+P0inv)                                                      
        Kshrt  = Iq  - np.dot(CtrRinvC, Kcore)
        PKsht  = np.dot(P0,    Kshrt) 
        KC     = np.dot(PKsht, CtrRinvC)        
        
        # update posterior estimates
        stats['mu'][ :,0] = mu0B0 + np.dot(PKsht,CtrRyDiff_Cmu0)
        stats['V'][:,:,0] = np.dot(Iq - KC, P0)
        stats['P'][:,:,0] = np.dot(np.dot(pars['A'],stats['V'][:,:,0]), 
                                           pars['Atr']) + pars['Q']
        stats['Pinv'][:,:,0] = sp.linalg.inv(stats['P'][:,:,0])

        # compute marginal probability y_0
        M    = sp.linalg.cholesky(P0)
        logdetCPCR    = (  np.sum(np.log(Rj)) 
                         + np.log(sp.linalg.det(
                               Iq + np.dot(M.transpose(),np.dot(CtrRinvC,M))))
                        )
        stats['logc'][0] = (  np.sum((yDiff * yDiff) / Rj)     
                          - np.dot(CtrRyDiff_Cmu0, np.dot(Kcore, CtrRyDiff_Cmu0)) 
                          + logdetCPCR
                         )
    else:  # no input at all, needs to be rewritten (would also be much faster)                                                
        mu0B0  = mu0+Bu_tr[:,0]
        P0 = V0
        stats['mu'][: ,0] = mu0B0 # no input, adding zero-mean innovation noise
        stats['V'][:,:,0] = P0  # Kalman gain is zero
        stats['P'][:,:,0] = np.dot(np.dot(pars['A'],stats['V'][:,:,0]), 
                                           pars['Atr']) + pars['Q']  
        stats['Pinv'][:,:,0] = sp.linalg.inv(stats['P'][:,:,0])          
        stats['logc'][0]     = 0   # setting log(N(y|0,Inf)) = log(1)


def _kft_obs_interval(stats, pars, y_tr, Bu_tr, t_start, t_end,
                      x_range, Iq, check_mat_conv, eps, idx):
    """ 
    Implements the Kalman filter for a time interval for which the observed
    subpopulation stays the same 

    """

    t_conv = t_end-1   # initialise as not converged wihtin this interval
    conv_flag = False  # 

    if idx.size > 0: # for non-empty observed populations:

        # pre-compute for this group of observed variables
        Cj   = pars['C'][np.ix_(idx,x_range)]           # all these
        Rj   = pars['R'][idx]                           # operations                        PRECOMPUTE AND TABULARIZE THESE
        CtrRinv = Cj.transpose() / Rj                   # are order
        CtrRinvC = np.dot(CtrRinv, Cj)                  # O(y_dim) !

        for t in range(t_start, t_end): 
                                               
            # pre-compute for this time step                                   
            AmuBu  = np.dot(pars['A'],stats['mu'][:,t-1]) + Bu_tr[:,t] 
            yDiff  = y_tr[idx,t]-pars['d'][idx]-np.dot(Cj,AmuBu)     # O(y_dim)
            CtrRyDiff_CAmu = np.dot(CtrRinv, yDiff)                  # O(y_dim)
                                               
            if not conv_flag:                                       
                # compute Kalman gain components
                Kcore  = sp.linalg.inv(CtrRinvC+stats['Pinv'][:,:,t-1])                                        
                Kshrt  = Iq  - np.dot(CtrRinvC, Kcore)
                PKsht  = np.dot(stats['P'][:,:,t-1],  Kshrt) 
                KC     = np.dot(PKsht, CtrRinvC)
                # update posterior covariances
                stats['V'][:,:,t] = np.dot(Iq - KC,stats['P'][:,:,t-1])
                stats['P'][:,:,t] = np.dot(np.dot(pars['A'],stats['V'][:,:,t]),
                                           pars['Atr']) + pars['Q']
                stats['Pinv'][:,:,t] = sp.linalg.inv(stats['P'][:,:,t])
                # compute normaliser for marginal probabilties of y_t
                M      = sp.linalg.cholesky(stats['P'][:,:,t-1])                                                     
                logdetCPCR = (  np.sum(np.log(Rj))                                  
                           + np.log(sp.linalg.det(Iq+np.dot(M.transpose(),
                                                     np.dot(CtrRinvC,M))))
                             )
                if check_mat_conv(stats['P'][:,:,t],stats['P'][:,:,t-1],eps):
                    t_conv = t
                    stats['V'][   :,:,t_end-1] = stats['V'][   :,:,t_conv]
                    stats['P'][   :,:,t_end-1] = stats['P'][   :,:,t_conv] 
                    stats['Pinv'][:,:,t_end-1] = stats['Pinv'][:,:,t_conv] 
                    conv_flag   = True

            # update posterior mean
            stats['mu'][ :,t] = AmuBu + np.dot(PKsht,CtrRyDiff_CAmu)

            # compute marginal probability y_t | y_0, ..., y_{t-1}
            stats['logc'][t] = (  np.sum((yDiff * yDiff) / Rj)   
                               - np.dot(CtrRyDiff_CAmu, np.dot(Kcore, 
                                                               CtrRyDiff_CAmu))
                               + logdetCPCR
                      )                            

    else:  # no input at all, needs to be rewritten (would also be much faster)             GIVE CONVERGENCE THIS CASE, AS WEll         

        for t in range(t_start, t_end): 

            AmuBu  = np.dot(pars['A'],stats['mu'][:,t-1]) + Bu_tr[:,t] 
            stats['mu'][ :,t] = AmuBu # adding zero-mean innovation noise
            stats['V'][:,:,t] = stats['P'][:,:,t-1]  # Kalman gain is zero
            stats['P'][:,:,t] = np.dot(np.dot(pars['A'],
                                              stats['V'][:,:,t]), 
                                              pars['Atr']) + pars['Q']
            stats['Pinv'][:,:,t] = sp.linalg.inv(stats['P'][:,:,t])          
            stats['logc'][t]     = 0   # setting log(N(y|0,Inf)) = log(1)


    return t_conv
                             


"""                         Kalman smoother                                 """



def kalman_smoother(stats,pars,Bu_tr,obs_time,
                    t_conv_ft=None,check_mat_conv=None,eps=0):        
    """ 
    see Bishop, 'Pattern Recognition and Machine Learning', ch. 13
    for formulas and input/output naming conventions   

    """        

    # add (pointers to) arrays for additional results
    stats['J'] = np.zeros((stats['x_dim'],stats['x_dim'],stats['t_tot']))

    # decide from coariance convergence-related inputs which function to use  
    #  unlike for the Kalman filter, the smoother changes in too many points
    #  to easily nest the basic version into the version that keeps track of
    #  convergence of latent covariance matrices. Hence, for ease of 
    #  development, we split the smoother into several different versions
    #  and now need to choose which one to use  
    ksm = _wrap_kalman_smoother(Bu_tr=Bu_tr,
                                obs_time=obs_time, 
                                t_conv_ft=t_conv_ft,
                                check_mat_conv=check_mat_conv,
                                eps=eps)
    
    # do the smoothing
    t_conv_sm = ksm(stats=stats, 
                    pars=pars)

    return t_conv_sm


def _wrap_kalman_smoother(Bu_tr, obs_time, t_conv_ft, check_mat_conv, eps):

    if (isinstance(t_conv_ft, np.ndarray) and 
        np.all(t_conv_ft.shape==obs_time.shape) and 
        np.any(t_conv_ft < obs_time-1)
        and eps > 0):

        def ksm(stats, pars):

            return _ksm_conv(stats=stats, 
                             pars=pars, 
                             Bu_tr=Bu_tr, 
                             obs_time=obs_time,
                             t_conv_ft=t_conv_ft,
                             check_mat_conv=check_mat_conv,
                             eps=eps)

    else:  # e.g. if simply t_conv_ft == None, or eps <= 0:

        if eps <= 0 and not check_mat_conv == _check_mat_conv_empty:
            print(('non-positive stopping criterion for convergence of latent '
                   'covariances.'))
            print(('will not track latent covariance convergence.'))

        def ksm(stats, pars):

            return _ksm_base(stats=stats, 
                             pars=pars, 
                             Bu_tr=Bu_tr)
    return ksm


def _ksm_base(stats, pars, Bu_tr):

    AmuBu = np.dot(pars['A'], stats['mu']) + Bu_tr

    t = stats['t_tot']-2 # t_tot-1 already done since mu and V stay unchanged

    while t >= 0: # note that obs_time_0[0] = 0
        stats['J'][:,:,t] = np.dot(np.dot(stats['V'][:,:,t], 
                                          pars['Atr']),
                                          stats['Pinv'][:,:,t])
        stats['mu'][ :,t] += np.dot(stats['J'][:,:,t],
                                    stats['mu'][:,t+1]-AmuBu[:,t])
        stats['V'][:,:,t] += np.dot(np.dot(
                                   stats['J'][:,:,t],
                                   stats['V'][:,:,t+1]-stats['P'][:,:,t]),
                                   stats['J'][:,:,t].transpose()
                                    ) 
        t -= 1

    return 0


def _ksm_conv(stats,pars,Bu_tr,obs_time,t_conv_ft,check_mat_conv,eps=0):

    # the smoother runs backwards in time and can converge after several time 
    # steps of running, i.e. some steps before the end of the time series 
    # segment. The very first few steps before the Kalman filter has converged, 
    # however, the smoother also cannot converge, as it depends on the filter 
    # outputs. The time of convergence for the smoother for each subpopulation
    # thus is in between t_conv_ft[i] and obs_time[i], and we need to split
    # up the total observation interval obs_time[i-1] to obs_time[i] into
    # different segments that are treated differently 

    # convenient to add zero to obs_time to know when to stop smoothing 
    obs_time_0 = np.empty(obs_time.size + 1)
    obs_time_0[0]  = 0
    obs_time_0[1:] = obs_time

    AmuBu = np.dot(pars['A'], stats['mu']) + Bu_tr

    t = stats['t_tot']-2 # t_tot-1 already done since mu and V stay unchanged
    if obs_time_0[-1]-obs_time_0[-2]==1: # we already processed whole subpop
        range_obstime = range(obs_time.size-1)[::-1] # and should skip it
    else:     # i.e. obs_time_0[-1]-obs_time_0[-2] > 1:
        range_obstime = range(obs_time.size)[::-1]

    # initialise convergence flag as not yet converged
    conv_flag = False

    t_conv_sm = t_conv_ft.copy()+1 # initialise with earliest possible time
    if t_conv_ft[-1]==stats['t_tot']-1:  # if V,P did not converge last time,
        t_conv_sm[-1] = stats['t_tot']-1 # this would be index out of bound 


    for i in range_obstime:

        conv_flag = False # reset convergence flag


        # in the following initial interval, P and V have converged, hence  
        # we can expect V_h to converge early on, as well
        Vconv = stats['V'][:,:,t_conv_ft[i]].copy()
        Jconv = np.dot(np.dot(Vconv, 
                              pars['Atr']),
                              stats['Pinv'][:,:,t_conv_ft[i]])
        Jconvtr = Jconv.transpose()
        Pconv = stats['P'][:,:,t_conv_ft[i]]        
        while t > t_conv_ft[i]: 
            stats['mu'][ :,t] += np.dot(Jconv,stats['mu'][:,t+1] - AmuBu[:,t]) 

            if not conv_flag:
                stats['V'][:,:,t] = ( Vconv 
                                    + np.dot(np.dot(
                                               Jconv, 
                                               stats['V'][:,:,t+1] - Pconv),
                                               Jconvtr)
                                    ) 
                if check_mat_conv(stats['V'][:,:,t], stats['V'][:,:,t+1], eps):
                    t_conv_sm[i] = t     # overwriting t_conv_ft[i] + 1
                    conv_flag     = True                                            

            t -= 1


        # at t = t_conv_ft[i], we update V_h to ensure we get the transition
        stats['mu'][ :,t] += np.dot(Jconv, 
                                    stats['mu'][:,t+1] - AmuBu[:,t]) 

        # now V_h[:,:,t] = Vconv, and if ifCoConv==False, it is still 
        # t_conv_sm[i] = t_conv_ft[i]+1 = t+1. Otherwise we the covariance
        # did converge and we also should look at t_conv_sm[i] now.
        stats['V'][:,:,t] = ( Vconv        
                            + np.dot(np.dot(
                                   Jconv, 
                                   stats['V'][:,:,t_conv_sm[i]] - Pconv),
                                   Jconvtr)
                             ) 
        stats['J'][:,:,t] = Jconv # here, store J for all later time points 
        t -= 1


        # in the following last interval, P and V still constantly change, so
        # we compute J for each time point invidivually
        while t >= obs_time_0[i]:
            stats['J'][:,:,t] = np.dot(np.dot(stats['V'][:,:,t], 
                                              pars['Atr']),
                                              stats['Pinv'][:,:,t])
            stats['mu'][ :,t] += np.dot(stats['J'][:,:,t],
                                        stats['mu'][:,t+1]-AmuBu[:,t])
            stats['V'][:,:,t] += np.dot(np.dot(
                                       stats['J'][:,:,t],
                                       stats['V'][:,:,t+1]-stats['P'][:,:,t]),
                                       stats['J'][:,:,t].transpose()
                                        ) 
            t -= 1

    return t_conv_sm



"""                             M-step                                      """



def _wrap_lds_m_step(y,u,obs_scheme,
                     use_A_flag=True,use_B_flag=False,diag_R_flag=True):

    def m_step(stats):
        return lds_m_step(stats=stats, 
                          y=y, 
                          u=u, 
                          obs_scheme=obs_scheme,
                          use_A_flag=use_A_flag, 
                          use_B_flag=use_B_flag, 
                          diag_R_flag=diag_R_flag)

    return m_step

def lds_m_step(stats, y, u, obs_scheme,
               use_A_flag = True, use_B_flag = False, diag_R_flag=True):   
    """
    see Bishop, 'Pattern Recognition and Machine Learning', ch. 13
    for formulas and input/output naming conventions   
    
    """                        
    if not (isinstance(u,np.ndarray) and 
        u.shape[1]==y.shape[1] and u.shape[2]==y.shape[2]):
        use_B_flag = False
    
    # compute all relevant statistics needed for the M-step, add to dict 'stats' 
    _setup_m_step(stats=stats,
                  y=y, 
                  u=u, 
                  obs_scheme=obs_scheme,
                  use_B_flag=use_B_flag)

    # Compute (closed-form) updated parameters
    pars = _m_step_pars(stats=stats,
                        obs_scheme=obs_scheme,
                        use_A_flag=use_A_flag, 
                        use_B_flag=use_B_flag, 
                        diag_R_flag=diag_R_flag)

    return pars        

def _setup_m_step(stats, y, u, obs_scheme, use_B_flag=False):

    # fill in observed data statistics where necessary
    _setup_m_step_yy(stats=stats,
                     y=y,
                     obs_scheme=obs_scheme)
    # compute full data statistics (i.e. those including the inferred x)
    _setup_m_step_yx(stats=stats,
                     y=y, 
                     obs_scheme=obs_scheme)     
    # compute input/data statistics (i.e. those including x and input u)
    if use_B_flag:
        _setup_m_step_ux(stats=stats,
                         u=u, 
                         obs_scheme=obs_scheme)
    
def _setup_m_step_yy(stats, y, obs_scheme):

    range_obstime = range(len(obs_scheme['obs_time']))

    # count occurence of each observed index group (for normalisations)
    if not 'ti' in stats:
        stats['ti'] = np.zeros(len(obs_scheme['idx_grp']));
        for tr in range(stats['n_tr']):
            for i in range_obstime:
                obs_idx = obs_scheme['obs_idx'][i]
                if i == 0:
                    stats['ti'][obs_idx] +=   obs_scheme['obs_time'][0]
                else:
                    stats['ti'][obs_idx] += ( obs_scheme['obs_time'][i] 
                                            - obs_scheme['obs_time'][i-1])

    # compute sum and (diagonal of) scatter matrix for observed states    
    if not 'sy' in stats:
        stats['sy'] = np.zeros(stats['y_dim'])
        for tr in range(stats['n_tr']):
            ytr = y[:,:,tr]            
            for i in range_obstime:
                idx = obs_scheme['sub_pops'][obs_scheme['obs_pops'][i]]
                if i == 0:
                    ts  = range(0, obs_scheme['obs_time'][i])                                            
                else:
                    ts  = range(obs_scheme['obs_time'][i-1],
                                obs_scheme['obs_time'][i])
                if idx.size>0:
                    stats['sy'][idx] += np.sum(ytr[np.ix_(idx,ts)],1)          
    if not 'syy' in stats:
        stats['syy']   = np.zeros(stats['y_dim']) # sum over y_t y_t'
        for tr in range(stats['n_tr']):
            ytr = y[:,:,tr]            
            for i in range_obstime:
                idx = obs_scheme['sub_pops'][obs_scheme['obs_pops'][i]]
                if i == 0:
                    ts  = range(0, obs_scheme['obs_time'][i])                                            
                else:
                    ts  = range(obs_scheme['obs_time'][i-1],
                                obs_scheme['obs_time'][i])
                if idx.size>0:
                    ytmp = ytr[np.ix_(idx,ts)]
                    stats['syy'][idx] += np.sum(ytmp*ytmp,1) 
        del ytmp      

def _setup_m_step_yx(stats, y, obs_scheme): 
    
    range_obstime = range(len(obs_scheme['obs_time']))

    # compute (diagonal of) scatter matrix accros observed and latent states
    # compute scatter matrices from posterior means for the latent states
    stats['sext']       = np.zeros(stats['x_dim'])   # sums of expected values
    stats['sextxt1toN'] = np.zeros((stats['x_dim'], stats['x_dim']))  
    stats['syext']      = np.zeros((stats['y_dim'], stats['x_dim']))  

    # versions of sums exclusively over data points where individual index
    # groups are observed:
    stats['sexts']   = np.zeros((stats['x_dim'], 
                                 len(obs_scheme['idx_grp'])))       
    stats['sextxts'] = np.zeros((stats['x_dim'], stats['x_dim'], 
                                 len(obs_scheme['idx_grp'])))            

    for tr in range(stats['n_tr']):              # collapse over trials ...
        ytr = y[:,:,tr]
        for i in range_obstime:         # ... but keep  apart
            idx = obs_scheme['sub_pops'][obs_scheme['obs_pops'][i]] 
            if i == 0:
                ts  = range(0, obs_scheme['obs_time'][i])                                            
            else:
                ts  = range(obs_scheme['obs_time'][i-1],
                            obs_scheme['obs_time'][i])

            tsext   = np.sum(stats['ext'][:,ts,tr],1)
            tsextxt = np.sum(stats['extxt'][:,:,ts,tr], 2)
            stats['sext']         += tsext          # these sum over 
            stats['sextxt1toN']   += tsextxt        # all times 
            for j in obs_scheme['obs_idx'][i]: 
                stats['sexts'][:,j]      += tsext   # these only sum entries 
                stats['sextxts'][:,:,j]  += tsextxt # seen by their index group

            if idx.size>0:     
                stats['syext'][idx,:] += np.einsum('in,jn->ij', # index groups 
                                 ytr[np.ix_(idx,ts)],   # do not overlap, so
                                 stats['ext'][:,ts,tr]) # store in same matrix

    del ytr

    # sum over y_t E[x_t']
    stats['sysext'] = np.outer(stats['sy'], stats['sext'])                                                             
    # partial sums over  E[x_t x_t']
    stats['sextxt2toN']   = (stats['sextxt1toN'] 
                            - np.sum(stats['extxt'][:,:,0 , :],2))  
    stats['sextxt1toNm1'] = (stats['sextxt1toN'] 
                            - np.sum(stats['extxt'][:,:,stats['t_tot']-1,:],2))
    # sum over E[x_t x_{t-1}']
    stats['sextxtm1'] = np.sum(stats['extxtm1'][:,:,1:stats['t_tot'],:], (2,3))

def _setup_m_step_ux(stats, u, obs_scheme):                                           # OBS_SCHEME CURRENTLY NOT USED: WILL HAVE TO DECIDE ON INPUT OBSERVATION PATTERNS

    # compute scatter matrix for input states
    if stats['suu'] is None:
        stats['suu']   = np.zeros((stats['u_dim'],stats['u_dim'])) 
        for tr in range(stats['n_tr']):
            utr = u[:,range(1,stats['t_tot']),tr]            
            stats['suu'] += np.dot(utr, utr.transpose())        
    if suuinv is None:
        stats['suuinv'] = sp.linalg.inv(stats['suu']) 

    # compute scatter matrix accros input and latent states
    stats['sextu'] = np.zeros((stats['x_dim'], stats['u_dim'])) 
    for tr in range(stats['n_tr']):        # collapse over trials ...
        stats['sextu'] += np.einsum('in,jn->ij', 
                            stats['ext'][:,range(1,stats['t_tot']),tr], 
                            u[:,range(1,stats['t_tot']),tr]) 
    stats['suextm1'] = np.zeros((u_dim, stats['x_dim']))
    for tr in range(stats['n_tr']):        # collapse over trials ...
        stats['suextm1'] += np.einsum('in,jn->ij', 
                                 u[:,range(1,stats['t_tot']),tr], 
                                 stats['ext'][:,range(0,stats['t_tot']-1),tr])

    # compute derived matrices as they eventually show up in the equations
    stats['suuvinvsuextm1'] = np.dot(stats['suuinv'], stats['suextm1'])
    stats['sExsuuusuExm1']  = np.dot(stats['sextu'],  stats['suuvinvsuextm1'])
    stats['sExm1suusuExm1'] = np.dot(stats['suextm1'].transpose(), 
                                     stats['suuvinvsuextm1'])

def _m_step_pars(stats, obs_scheme, use_A_flag, use_B_flag, diag_R_flag):

    # store parameters within dictionary
    pars = {}

    # initial distribution parameters
    _m_step_init(pars, stats)

    # latent dynamics parameters
    _m_step_dyns(pars, stats, use_A_flag, use_B_flag)

    # observed state parameters 
    _m_step_emis(pars, stats, obs_scheme, diag_R_flag)

    return pars 

def _m_step_init(pars, stats):                      

    pars['mu0'] =  np.sum( stats['ext'][:,0,:],     1)/stats['n_tr']                        # still blatantly
    pars['V0']  = (np.sum( stats['extxt'][:,:,0,:], 2)/stats['n_tr']                        # wrong in case of 
                 - np.outer(pars['mu0'],pars['mu0']))                                       # input on first step!    

def _m_step_dyns(pars, stats, use_A_flag=True, use_B_flag=False):                      

    if use_B_flag:                   
        if use_A_flag:
            pars['A'] = np.dot(stats['sextxtm1']-stats['sExsuuusuExm1'], 
                          sp.linalg.inv(  stats['sextxt1toNm1']
                                        - stats['sExm1suusuExm1']))                                    
        else:
            pars['A'] = np.zeros((stats['x_dim'],stats['x_dim']))
        pars['Atr'] = pars['A'].transpose()

        pars['B'] = np.dot(stats['sextu']-np.dot(pars['A'],
                                                 stats['suextm1'].transpose()),
                           stats['suuinv'])
        Btr = pars['B'].transpose()
        sextxtm1Atr = np.dot(stats['sextxtm1'], pars['Atr'])
        sextuBtr   = np.dot(stats['sextu'], Btr)
        Bsuextm1Atr = np.dot(pars['B'], np.dot(stats['suextm1'], pars['Atr']))
        pars['Q'] = (  stats['sextxt2toN']   
                     - sextxtm1Atr.transpose()
                     - sextxtm1Atr 
                     + np.dot(np.dot(pars['A'],stats['sextxt1toNm1']),
                              pars['Atr']) 
                     - sextuBtr.transpose()
                     - sextuBtr
                     + Bsuextm1Atr.transpose()
                     + Bsuextm1Atr
                     + np.dot(np.dot(pars['B'], stats['suu']), Btr)
                    ) / (stats['n_tr']*(stats['t_tot']-1))

    else: # reduce to non-input LDS equations

        if use_A_flag:
            pars['A'] = np.dot(stats['sextxtm1'], 
                       sp.linalg.inv(stats['sextxt1toNm1']))                                    
        else:
            pars['A'] = np.zeros((stats['x_dim'],stats['x_dim']))
        pars['Atr'] = pars['A'].transpose()
        sextxtm1Atr = np.dot(stats['sextxtm1'], pars['Atr'])
        pars['B']= np.array([])                
        pars['Q'] = (  stats['sextxt2toN']   
                     - sextxtm1Atr.transpose()
                     - sextxtm1Atr 
                     + np.dot(np.dot(pars['A'],stats['sextxt1toNm1']),
                              pars['Atr'])
                    ) / (stats['n_tr']*(stats['t_tot']-1))

def _m_step_emis(pars, stats, obs_scheme, diag_R_flag=True):
       
    range_idx_grp = range(len(obs_scheme['idx_grp']))

    # observed state parameters C, d    
    # The maximum likelihood solution for C, d in the context of missing data
    # are different from the standard solutions and given by 
    # C[i,:] = (sum_{t:ti} (y(i)_t-d(i)) x_t')  (sum_{t:ti} x_t x_'t)^-1
    # where {ti} is the set of time points t where variable y(i) is observed
    # d[i]   = 1/|ti| sum_{t:ti} y(i)_t - C[i,:] * x_t
    pars['C']   = np.zeros((stats['y_dim'],stats['x_dim']))    
    for i in range_idx_grp:
        ixg  = obs_scheme['idx_grp'][i]
        pars['C'][ixg,:] = np.dot(
                 stats['syext'][ixg,:]
               - np.outer(stats['sy'][ixg],stats['sexts'][:,i])/stats['ti'][i], 
                          sp.linalg.inv(
                               stats['sextxts'][:,:,i]
                             - np.outer(stats['sexts'][:,i],
                                        stats['sexts'][:,i])/stats['ti'][i]
                                        )
                                  )
    pars['d'] = np.zeros(stats['y_dim'])
    for i in range_idx_grp:    
        ixg  = obs_scheme['idx_grp'][i]
        pars['d'][ixg] = ( stats['sy'][ixg] 
                         - np.dot(pars['C'][ixg,:],
                                  stats['sexts'][:,i]))/ stats['ti'][i]

    # now use C, d to compute key terms of the residual noise
    CsextxtCtr = np.zeros(stats['y_dim'])
    sdext      = np.zeros((stats['y_dim'],stats['x_dim']))
    for i in range_idx_grp: 
        ixg  = obs_scheme['idx_grp'][i]
        sdext[ixg,:] += np.outer(pars['d'][ixg],stats['sexts'][:,i])
        Cj  = pars['C'][ixg,:]
        CsextxtCtr[ixg] += np.einsum('ij,ik,jk->i',
                                      Cj,Cj,stats['sextxts'][:,:,i])

    # compute observation noise parameter R
    pars['R'] = ( stats['syy'] - 2 * stats['sy'] * pars['d']
                + CsextxtCtr
                + 2 * np.sum(pars['C'] * (sdext-stats['syext']),1)
                ) 
    for i in range_idx_grp:
        pars['R'][obs_scheme['idx_grp'][i]] /= stats['ti'][i] # normalise
    pars['R'] += pars['d']*pars['d']         # normalisation of dd' cancels out 



"""                             utility                                     """



def get_obs_index_groups(obs_scheme,y_dim):
    """ INPUT:
        obs_scheme: observation scheme for given data, stored in dictionary
                   with keys 'sub_pops', 'obs_time', 'obs_pops'
        y_dim:        dimensionality of observed variables y
    Computes index groups for given observation scheme. 

    """
    try:
        sub_pops = obs_scheme['sub_pops'];
        obs_time = obs_scheme['obs_time'];
        obs_pops = obs_scheme['obs_pops'];
    except:
        raise Exception(('provided obs_scheme dictionary does not have '
                         'the required fields: sub_pops, obs_time, '
                         'and obs_pops.'))        

    J = np.zeros((y_dim, len(sub_pops))) # binary matrix, each row gives which 
    for i in range(len(sub_pops)):      # subpopulations the observed variable
        if sub_pops[i].size > 0:        # y_i is part of
            J[sub_pops[i],i] = 1   

    twoexp = np.power(2,np.arange(len(sub_pops))) # we encode the binary rows 
    hsh = np.sum(J*twoexp,1)                     # of J using binary numbers

    lbls = np.unique(hsh)         # each row of J gets a unique label 
                                     
    idx_grp = [] # list of arrays that define the index groups
    for i in range(lbls.size):
        idx_grp.append(np.where(hsh==lbls[i])[0])

    obs_idx = [] # list f arrays giving the index groups observed at each
                 # given time interval
    for i in range(len(obs_pops)):
        obs_idx.append([])
        for j in np.unique(hsh[np.where(J[:,obs_pops[i]]==1)]):
            obs_idx[i].append(np.where(lbls==j)[0][0])            
    # note that we only store *where* the entry was found, i.e. its 
    # position in labels, not the actual label itself - hence we re-defined
    # the labels to range from 0 to len(idx_grp)

    return obs_idx, idx_grp



"""                          input checking                                 """



def check_obs_scheme(obs_scheme,y_dim,t_tot):
    """ INPUT:
        y_dim : dimensionality of observed states y
        t_tot : trial length (in number of time points)
        obs_scheme: observation scheme for given data, stored in dictionary
                   with keys 'sub_pops', 'obs_time', 'obs_pops'
        Checks the internal validity of provided obs_scheme dictionaries that 
        contain the information on the observation scheme that is key to 
        any stitching and missing-value context. Checks if formatting is 
        correct, finds superfluous (e.g. never observed) subpopulations,
        and fills in index groups (via get_obs_index_groups()) if 
        necessary.

    """
    if obs_scheme is None:
        obs_scheme = {'sub_pops': [np.arange(y_dim)], # creates default case
                     'obs_time': np.array([t_tot]),   # of fully observed
                     'obs_pops': np.array([0])}       # population
    else: 
        try:
            obs_scheme['sub_pops'] # check for the 
            obs_scheme['obs_time'] # fundamental
            obs_scheme['obs_pops'] # information
        except:                   # have to give hard error here !
            raise Exception(('provided observation scheme is insufficient. '
                             'It requires the fields sub_pops, obs_time and '
                             'obs_pops. Not all those fields were given.'))

        # check sub_pops
        if not isinstance(obs_scheme['sub_pops'], list):
            raise Exception(('Variable sub_pops on top-level has to be a list. '
                             'However, it is '), obs_scheme['sub_pops']) 
        else: 
            for i in range(len(obs_scheme['sub_pops'])): 
                if isinstance(obs_scheme['sub_pops'][i], list):
                    obs_scheme['sub_pops'][i] = np.array(
                                                    obs_scheme['sub_pops'][i]
                                                         )
                elif not isinstance(obs_scheme['sub_pops'][i], np.ndarray):
                    raise Exception(('entries of variable sub_pops have to be '
                                     'ndarrays of variable indexes. '
                                     'However, the '  
                                       + str(i) + 
                                     '-th of them is '), 
                                      obs_scheme['sub_pops'][i]) 
        idx_union = np.sort(obs_scheme['sub_pops'][0]) # while loop for speed
        i = 1                                       # (could break early!)
        while not idx_union.size == y_dim and i < len(obs_scheme['sub_pops']):
            idx_union = np.union1d(idx_union, obs_scheme['sub_pops'][i]) 
            i += 1            
        if not (idx_union.size == y_dim and np.all(idx_union==np.arange(y_dim))):
            raise Exception(('all subpopulations together have to cover '
                             'exactly all included observed varibles y_i in y.'
                             'This is not the case. Change the difinition of '
                             'subpopulations in variable sub_pops or reduce '
                             'the number of observed variables y_dim. '
                             'The union of indices of all subpopulations is'),
                             idx_union )

        # check obs_time
        if isinstance(obs_scheme['obs_time'], list):
            obs_scheme['obs_time'] = np.array(obs_scheme['obs_time'])
        elif not isinstance(obs_scheme['obs_time'], np.ndarray):
            raise Exception(('variable obs_time has to be an ndarray of time '
                             'points where observed subpopulations switch.'))
        if not obs_scheme['obs_time'][-1]==t_tot:
            print(('Warning: entries of obs_time give the respective ends of '
                             'the periods of observation for any '
                             'subpopulation. Hence the last entry of obs_time '
                             'has to be the full recording length. Will '
                             'change the provided obs_time accordingly. '
                             'The last enry of obs_time before was '),
                              obs_scheme['obs_time'][-1])
            obs_scheme['obs_time'][-1] = t_tot
        if np.any(np.diff(obs_scheme['obs_time'])<1):
            raise Exception(('lengths of observation have to be at least 1. '
                             'minimal observation time for a subpopulation: '),
                             np.min(np.diff(obs_scheme['obs_time'])))

        # check obs_pops
        if isinstance(obs_scheme['obs_pops'], list):
            obs_scheme['obs_pops'] = np.array(obs_scheme['obs_pops'])
        elif not isinstance(obs_scheme['obs_pops'], np.ndarray):
            raise Exception('variable obs_pops has to be an np.ndarray')
        if not obs_scheme['obs_time'].size == obs_scheme['obs_pops'].size:
            raise Exception(('each entry of obs_pops gives the index of the '
                             'subpopulation observed up to the respective '
                             'time given in obs_time. Thus the sizes of the '
                             'two arrays have to match. They do not. '
                             'no. of subpop. switch points and no. of '
                             'subpopulations ovserved up to switch points '
                             'are '), (obs_scheme['obs_time'].size,
                                       obs_scheme['obs_pops'].size))

        idx_pops = np.sort(np.unique(obs_scheme['obs_pops']))
        if not np.min(idx_pops)==0:\
            raise Exception(('first subpopulation has to have index 0, but '
                             'is given the index '), np.min(idx_pops))
        elif not np.all(np.diff(idx_pops)==1):
            raise Exception(('subpopulation indices have to be consecutive '
                             'integers from 0 to the total number of '
                             'subpopulations. This is not the case. '
                             'Given subpopulation indices are '),
                              idx_pops)
        elif not idx_pops.size == len(obs_scheme['sub_pops']):
            raise Exception(('number of specified subpopulations in variable '
                             'sub_pops does not meet the number of '
                             'subpopulations indexed in variable obs_pops. '
                             'Delete subpopulations that are never observed, '
                             'or change the observed subpopulations in '
                             'variable obs_pops accordingly. The number of '
                             'indexed subpopulations is '),
                              len(obs_scheme['sub_pops']))

    try:
        obs_scheme['obs_idx']     # check for addivional  
        obs_scheme['idx_grp']     # (derivable) information
    except:                       # can fill in if missing !
        [obs_idx, idx_grp] = get_obs_index_groups(obs_scheme,y_dim)
        obs_scheme['obs_idx'] = obs_idx # add index groups and 
        obs_scheme['idx_grp'] = idx_grp # their occurences

    return obs_scheme


def check_pars(pars, x_dim=None, y_dim=None, u_dim=None):
    """ INPUT:
        pars: list of parameters for an LDS 
        x_dim: dimensionality of latent variables
        y_dim: dimensionality of observed variables
        u_dim: dimensionality of input variables
        Short function to check the consistency of dimensionality of parameters
        Returns nothing, only raises exceptions in case of inconsistencies.

    """
    # when not providing all variable dimensionalities, pick sizes of selected
    # variables and use them check for consistency with the other parameters.
    if x_dim is None:
        try:
            x_dim = pars['A'].shape[0]
        except:
            raise Exception(('when not providing latent dimensionality x_dim, '
                             'need to provide a valid dynamics parameter A.'))
    if y_dim is None:
        try:
            y_dim = pars['C'].shape[0]
        except:
            raise Exception(('when not providing observed dimensionality y_dim,'
                             ' need to provide a valid emission parameter C.'))
    if u_dim is None:
        try:
            u_dim = pars['B'].shape[1]
        except:
            raise Exception(('when not providing latent dimensionality u_dim, '
                             'need to provide a valid input parameter B.'))            


    # check latent state tranition matrix A
    if not np.all(pars['A'].shape == (x_dim,x_dim)):
        raise Exception(('Variable A has to be a (x_dim,x_dim)-array. '
                         'However, it has shape '), pars['A'].shape) 
    if not 'Atr' in pars:
        pars['Atr'] = pars['A'].transpose()      


    # check latent state input matrix B
    if not np.all(pars['B'].shape == (x_dim,u_dim)):
        raise Exception(('Variable B has to be a (x_dim,u_dim)-array. '
                         'However, it has shape '), pars['B'].shape)


    # check latent state innovation noise matrix Q
    if not np.all(pars['Q'].shape == (x_dim,x_dim)):
        raise Exception(('Variable Q has to be a (x_dim,x_dim)-array. '
                         'However, it has shape '), pars['Q'].shape)


    # check initial latent state mean mu0
    if (np.all(pars['mu0'].shape == (x_dim,1)) or 
        np.all(pars['mu0'].shape == (1,x_dim))):
        pars['mu0'] = pars['mu0'].reshape(x_dim)     
    if not np.all(pars['mu0'].shape == (x_dim,)):
        raise Exception(('Variable mu0 has to be a (x_dim,)-array. '
                         'However, it has shape '), pars['mu0'].shape)


    # check initial latent state covariance matrix V0
    if not np.all(pars['V0'].shape == (x_dim,x_dim)):
        raise Exception(('Variable V0 has to be a (x_dim,x_dim)-array. '
                         'However, it has shape '), pars['V0'].shape)


    # check emission matrix C
    if not np.all(pars['C'].shape == (y_dim,x_dim)):
        raise Exception(('Variable C has to be a (y_dim,x_dim)-array. '
                         'However, it has shape '), pars['C'].shape)


    # check emission noise covariance matrix d
    if (np.all(pars['d'].shape==(y_dim,1)) or 
        np.all(pars['d'].shape==(1,y_dim))):
        pars['d'] = pars['d'].reshape(y_dim)     
    if not np.all(pars['d'].shape == (y_dim,)):
        raise Exception(('Variable d has to be a (y_dim,)-array. '
                         'However, it has shape '), pars['d'].shape)        


    # check emission noise covariance matrix R
    if (not (np.all(pars['R'].shape == (y_dim,y_dim)) or
             np.all(pars['R'].shape == (y_dim, )   )) ):
        raise Exception(('Variable R is assumed to be diagonal. '
                         'Please provide the diagonal entries as'
                         ' (y_dim,)-array. The provided R has shape '),
                         pars['R'].shape) 


def check_options(max_iter, epsilon, eps_cov, plot_flag, 
                  trace_pars_flag, trace_stats_flag,
                  diag_R_flag, use_A_flag, use_B_flag, save_file): 

    if not (isinstance(max_iter, numbers.Integral) and max_iter > 0):
        raise Exception(('argument max_iter has to be a positive integer. '
                         'However, it is '), max_iter)

    if (not (isinstance(epsilon, (float, numbers.Integral)) and
            epsilon >= 0) ):
        raise Exception(('argument epsilon has to be a non-negative number. '
                         'However, it is '), epsilon)

    if not isinstance(plot_flag, bool):
        raise Exception(('argument plot_flag has to be a boolean. '
                         'However, it is '), plot_flag)

    if not isinstance(trace_pars_flag, bool):
        raise Exception(('argument trace_pars_flag has to be a boolean. '
                         'However, it is '), trace_pars_flag)
     
    if not isinstance(trace_stats_flag, bool):
        raise Exception(('argument trace_stats_flag has to be a boolean. '
                         'However, it is '), trace_stats_flag)

    if not isinstance(diag_R_flag, bool):
        raise Exception(('argument diag_R_flag has to be a boolean. '
                         'However, it is '), diag_R_flag)  

    if not isinstance(use_A_flag, bool):
        raise Exception(('argument use_A_flag has to be a boolean. '
                         'However, it is '), use_A_flag)   

    if not isinstance(use_B_flag, bool):
        raise Exception(('argument use_B_flag has to be a boolean. '
                         'However, it is '), use_B_flag)   

    if (not isinstance(eps_cov,(float,numbers.Integral))
        or not eps_cov >= 0):
        raise Exception(('eps_cov has to be a non-negative number. '
                        'However, it is '), eps_cov)
    if eps_cov > 1e-1:
        print(('Warning: Selected convergence criterion for latent '
               'covariance is very generous. Results of the E-step may '
               'be very imprecise. Consider using something like 1e-30.'))


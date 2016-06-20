import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import numbers
import time
import ssm_timeSeries as ts  # library for time series overhead
import ssm_fit               # library for state-space model fitting

import random
from datetime import datetime     # generate random seed for 
random.seed(datetime.now())       # np.random. Once this is fixed, all 
rngSeed = random.randint(0, 1000) # other 'randomness' is also fixed

from scipy.io import savemat # store results for comparison with Matlab code   

def run(x_dim, 
        y_dim, 
        u_dim, 
        t_tot, 
        obs_scheme=None, 

        max_iter=10, 
        epsilon=np.log(1.001),
        eps_cov=0,        
        plot_flag=False,
        trace_pars_flag=False,
        trace_stats_flag=False,
        diag_R_flag=True,
        use_A_flag=True,
        use_B_flag=False,

        pars_true=None,                          
        gen_A_true='diagonal', 
        lts_true=None,
        gen_B_true='random', 
        gen_Q_true='identity', 
        gen_mu0_true='random', 
        gen_V0_true='identity', 
        gen_C_true='random', 
        gen_d_true='scaled', 
        gen_R_true='fraction',

        pars_init=None,
        gen_A_init='diagonal', 
        lts_init=None,
        gen_B_init='random',  
        gen_Q_init='identity', 
        gen_mu0_init='random', 
        gen_V0_init='identity', 
        gen_C_init='random', 
        gen_d_init='mean', 
        gen_R_init='fractionObserved',

        u=None, input_type='pwconst',const_input_length=1,
        y=None, 
        x=None,
        interm_store_flag = False,
        save_file='LDS_data.mat'):

    """ INPUT:
        x_dim : dimensionality of latent states x
        y_dim : dimensionality of observed states y
        u_dim : dimensionality of input states u
        t_tot : trial length (in number of time points)
        obs_scheme: observation scheme for given data, stored in dictionary
                   with keys 'sub_pops', 'obs_time', 'obs_pops'
        max_iter: maximum number of allowed EM steps
        epsilon:    precision (stopping criterion) for deciding on convergence
                    of overall EM algorithm
        eps_cov: precision (stopping criterion) for deciding on convergence
                    of latent covariance estimates durgin the E-step        
        plot_flag   : boolean specifying if to visualise fitting progress`
        trace_pars_flag:  boolean, specifying if entire parameter updates 
                           history or only the current state is kept track of 
        trace_stats_flag: boolean, specifying if entire history of inferred  
                           latents or only the current state is kept track of 
        diag_R_flag      : boolean specifying if R is represented as diagonal
        use_A_flag  : boolean specifying whether to fit the LDS with parameter A
        use_B_flag  : boolean specifying whether to fit the LDS with parameter B

        pars_true : None, or list/np.ndarray/dict containing no, some or all
                   of the desired ground-truth parameters. Will identify any
                   parameters not handed over and will fill in the rest
                   according to selected strings below.
        gen_A_true   : string specifying methods of parameter generation
        lts_true    : ndarray with one entry per latent time scale (i.e. x_dim)
        gen_B_true   : string specifying methods of parameter generation
        gen_Q_true   :  ""
        gen_mu0_true :  "" 
        gen_C_true   :  ""
        gen_V0_true   :  "" 
        gen_d_true   :  ""
        gen_R_true   : (see below for details)
        pars_init : None, or list/np.ndarray/dict containing no, some or all
                   of the desired parameter initialisations. Will identify any
                   parameters not handed over and will fill in the rest
                   according to selected strings below.
        gen_A_init   : string specifying methods of parameter initialisation
        lts_init    : ndarray with one entry per latent time scale (i.e. x_dim)
        gen_B_init   : string specifying methods of parameter initialisation
        gen_Q_init   :  ""
        gen_mu0_init :  "" 
        gen_V0_init  :  ""
        gen_C_init   :  "" 
        gen_d_init   :  ""
        gen_R_init   : (see below for details)
        x: data array of latent variables
        y: data array of observed variables
        u: data array of input variables
        interm_store_flag : boolean, specifying whether or not to 
                                     store the intermediate results after 
                                     each EM cycle to the same folder as
                                     given by input variable save_file
        save_file : (path to folder and) name of file for storing results.  
        Generates parameters of an LDS, potentially by looking at given data.
        Can be used for for generating ground-truth parameters for generating
        data from an artificial experiment using the LDS, or for finding 
        parameter initialisations for fitting an LDS to data. Usage is slightly
        different in the two cases (see below). By nature of the wide range of
        applicability of the LDS model, this function contains many options
        (implemented as strings differing different cases, and arrays giving
         user-specified values such as timescale ranges), and is to be extended
        even further in the future.

    """
    if not isinstance(use_B_flag,bool):
        raise Exception('use_B_flag has to be a boolean. However, it is', use_B_flag)

    if not isinstance(use_A_flag,bool):
        raise Exception('use_A_flag has to be a boolean. However, it is', use_A_flag)

    if not isinstance(diag_R_flag,bool):
        raise Exception('diag_R_flag has to be a boolean. However, it is ',
                         diag_R_flag)

    if not isinstance(interm_store_flag,bool):
        raise Exception(('interm_store_flag has to be a boolean' 
                         'However, it is '), interm_store_flag)
       
    obs_scheme = ssm_fit.check_obs_scheme(obs_scheme=obs_scheme,
                                          y_dim=y_dim,
                                          t_tot=t_tot)

    if y is None:
        if lts_true is None:
            lts_true = np.linspace(0.9,0.98,x_dim)
        pars_true, pars_options_true = gen_pars(
                          x_dim=x_dim, 
                          y_dim=y_dim, 
                          u_dim=u_dim, 
                          pars_in=pars_true,
                          obs_scheme=obs_scheme,
                          gen_A=gen_A_true, 
                          lts=lts_true,
                          gen_B=gen_B_true, 
                          gen_Q=gen_Q_true, 
                          gen_mu0=gen_mu0_true, 
                          gen_V0=gen_V0_true, 
                          gen_C=gen_C_true,
                          gen_d=gen_d_true, 
                          gen_R=gen_R_true,
                          diag_R_flag=diag_R_flag)
        n_tr = 1 # fix to always just one repetition for now        

        # generate data from model
        print('generating data from model with ground-truth parameters')
        x,y,u = sim_data(pars=pars_true,
                         t_tot=t_tot,
                         n_tr=n_tr,
                         obs_scheme=obs_scheme,
                         u=u,
                         input_type=input_type,
                         const_input_length=const_input_length)

        Pi   = sp.linalg.solve_discrete_lyapunov(pars_true['A'], 
                                                 pars_true['Q'])
        Pi_t = np.dot(pars_true['A'].transpose(), Pi)

        stats_true, lltr, t_conv_ft, t_conv_sm = \
                        do_e_step(pars=pars_true, 
                                  y=y, 
                                  u=u, 
                                  obs_scheme=obs_scheme, 
                                  eps=eps_cov)

    else:  # i.e. if data provided
        pars_true = {}
        pars_true['A'] = 0
        pars_true['B'] = 0
        pars_true['Q'] = 0
        pars_true['mu0'] = 0
        pars_true['V0'] = 0
        pars_true['C'] = 0
        pars_true['d'] = 0
        pars_true['R'] = 0
        Pi   = 0
        Pi_t = 0
        ext_true = 0
        extxt_true = 0
        extxtm1_true = 0
        lltr = 0


    # get initial parameters
    if not use_A_flag: # overwrites any other parameter choices for A! Set A = 0 
        if isinstance(pars_init, dict) and ('A' in pars_init):
            pars_init['A'] = np.zeros((x_dim,x_dim))
        elif (isinstance(pars_init,(list,np.ndarray)) and 
              not pars_init[0] is None):
            iniPars[0] = np.zeros((x_dim,x_dim))
        elif not gen_A_init == 'zero': 
            print(('Warning: set flag use_A_flag=False, but did not set gen_A_init '
                   'to zero. Will overwrite gen_A_init to zero now.'))
            gen_A_init = 'zero'
    if not use_B_flag: # overwrites any other parameter choices for B! Set B = 0
        if isinstance(pars_init, dict) and ('B' in pars_init):
            pars_init['B'] = np.zeros((x_dim,u_dim))
        elif (isinstance(pars_init,(list,np.ndarray)) and 
              not pars_init[1] is None):
            iniPars[1] = np.zeros((x_dim,u_dim))
        elif not gen_B_init == 'zero': 
            print(('Warning: set flag ifBseA=False, but did not set gen_B_init '
                   'to zero. Will overwrite gen_B_init to zero now.'))
            gen_B_init = 'zero'

    if lts_init is None:
        lts_init = np.random.uniform(size=[x_dim])

    pars_init, pars_options_init = gen_pars(
                      x_dim=x_dim, 
                      y_dim=y_dim, 
                      u_dim=u_dim, 
                      pars_in=pars_init, 
                      obs_scheme=obs_scheme,
                      gen_A=gen_A_init, 
                      lts=lts_init,
                      gen_B=gen_B_init, 
                      gen_Q=gen_Q_init, 
                      gen_mu0=gen_mu0_init, 
                      gen_V0=gen_V0_init, 
                      gen_C=gen_C_init,
                      gen_d=gen_d_init, 
                      gen_R=gen_R_init,
                      diag_R_flag=diag_R_flag,
                      x=x, y=y, u=u)

    # check initial goodness of fit for initial parameters
    stats_init,ll_init,pars_first,stats_first,ll_first = do_first_em_cycle(
                                                          pars_init=pars_init, 
                                                          obs_scheme=obs_scheme, 
                                                          y=y, 
                                                          u=u, 
                                                          eps=eps_cov,
                                                          use_A_flag=use_A_flag,
                                                          use_B_flag=use_B_flag,
                                                          diag_R_flag=diag_R_flag)
    if interm_store_flag:
        save_file_interm = save_file
    else: 
        save_file_interm = None


    fit_lds = setup_fit_lds(y=y, 
                            u=y, 
                            max_iter=max_iter,
                            epsilon=epsilon, 
                            eps_cov=eps_cov,
                            plot_flag=plot_flag, 
                            trace_pars_flag=trace_pars_flag, 
                            trace_stats_flag=trace_stats_flag, 
                            diag_R_flag=diag_R_flag,
                            use_A_flag=use_A_flag, 
                            use_B_flag=use_B_flag)

    # fit the model to data          
    print('fitting model to data')
    t = time.time()
    pars_hat,ll = fit_lds(x_dim=x_dim,
                          pars=pars_init, 
                          obs_scheme=obs_scheme,
                          save_file=save_file_interm)

    elapsed_time = time.time() - t
    print('elapsed time for fitting is')
    print(elapsed_time)


    stats_hat = ssm_fit._setup_stats(y, x_dim, u_dim)
    if use_B_flag:
        stats_hat, ll_hat, t_conv_ft, t_conv_sm = \
         ssm_fit.lds_e_step(pars=pars_hat,
                            y=y,
                            u=u, 
                            obs_scheme=obs_scheme,
                            eps=eps_cov)

    else:
        stats_hat, ll_hat, t_conv_ft, t_conv_sm = \
         ssm_fit.lds_e_step(pars=pars_hat,
                            y=y,
                            u=None, 
                            obs_scheme=obs_scheme,
                            eps=eps_cov)            

    Pi_h   = sp.linalg.solve_discrete_lyapunov(pars_hat['A'], 
                                               pars_hat['Q'])
    Pi_t_h  = np.dot(pars_hat['A'].transpose(), Pi_h)

    # save results for visualisation (with Matlab code)
    if u is None:
        u = 0
        B_h = 0
        B_hs = [0]
    if B_h is None:
        B_h = 0
        B_hs = [0]
    save_file_m = {'x': x, 'y': y, 'u' : u, 'll' : ll, 
                      'T' : t_tot, 'Trial':n_tr, 'elapsedTime' : elapsed_time,
                      'inputType' : input_type,
                      'constInputLngth' : const_input_length,
                      'ifUseB':use_B_flag, 'ifUseA':use_A_flag, 
                      'epsilon':epsilon,
                      'ifPlotProgress':plot_flag,
                      'ifTraceParamHist':trace_pars_flag,
                      'ifTraceStatsHist':trace_stats_flag,
                      'ifRDiagonal':diag_R_flag,
                      'ifUseB':use_B_flag,
                      'covConvEps':eps_cov,        
                      'truePars':pars_true,
                      'initPars':pars_init,
                      'firstPars':pars_first,
                      'estPars': pars_hat,
                      'stats_0': stats_init,
                      'stats_1': stats_first,
                      'stats_h': stats_hat,
                      'stats_true': stats_true,
                      'Pi':Pi,'Pi_h':Pi_h,'Pi_t':Pi_t,'Pi_t_h': Pi_t_h,
                      'obsScheme' : obs_scheme}

    savemat(save_file,save_file_m) # does the actual saving

    # translating naming convention to mattjj's code where applicable:
    np.savez(save_file, data=np.squeeze(y).T, 
                        stateseq=np.squeeze(x).T, 
                        u=u, 
                        likes=ll.tolist(), 
                        T=t_tot,
                        num_tria=n_tr,
                        elapsedTime=elapsed_time,
                        eps=epsilon,
                        diag_R_flag=diag_R_flag,
                        use_B_flag=use_B_flag,
                        covConvEps=eps_cov,        
                        pars_true=pars_true,
                        pars_init=pars_init,
                        pars_first=pars_first,
                        pars_hat=pars_hat,
                        stats_init=stats_init,
                        stats_first=stats_first,
                        stats_hat=stats_hat,
                        stats_true=stats_true,
                        Pi=Pi,Pi_h=Pi_h,Pi_t=Pi_t,Pi_t_h=Pi_t_h,
                        obs_scheme=obs_scheme)

    return y,x,u,pars_hat,pars_init,pars_true

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def gen_pars(x_dim, y_dim, u_dim=0, 
             pars_in=None, 
             obs_scheme=None,
             gen_A='diagonal', lts=None,
             gen_B='random', 
             gen_Q='identity', 
             gen_mu0='random', 
             gen_V0='identity', 
             gen_C='random', 
             gen_d='scaled', 
             gen_R='fraction',
             diag_R_flag=True,
             x=None, y=None, u=None): 
    """ INPUT:
        x_dim : dimensionality of latent states x
        y_dim : dimensionality of observed states y
        u_dim : dimensionality of input states u
        pars_in:    None, or list/np.ndarray/dict containing no, some or all
                   of the desired parameters. This function will identify which
                   parameters were not handed over and will fill in the rest
                   according to selected paramters below.
        obs_scheme: observation scheme for given data, stored in dictionary
                   with keys 'sub_pops', 'obs_time', 'obs_pops'
        gen_A   : string specifying methods of parameter generation
        lts    : ndarray with one entry per latent time scale (i.e. x_dim many)
        gen_B   :  ""
        gen_Q   :  ""
        gen_mu0 :  "" 
        gen_V0  :  ""
        gen_C   :  "" 
        gen_d   :  ""
        gen_R   : (see below for details)
        x: data array of latent variables
        y: data array of observed variables
        u: data array of input variables
        Generates parameters of an LDS, potentially by looking at given data.
        Can be used for for generating ground-truth parameters for generating
        data from an artificial experiment using the LDS, or for finding 
        parameter initialisations for fitting an LDS to data. Usage is slightly
        different in the two cases (see below). By nature of the wide range of
        applicability of the LDS model, this function contains many options
        (implemented as strings differing different cases, and arrays giving
         user-specified values such as timescale ranges), and is to be extended
        even further in the future.

    """
    totNumParams = 8 # an input-LDS model has 8 parameters: A,B,Q,mu0,V0,C,d,R

    """ pars optional inputs to this function """
    if (not (isinstance(lts, np.ndarray) and 
             (np.all(lts.shape==(x_dim,)) or np.all(lts.shape==(x_dim,1)))
            ) ):
        print('lts (latent time scales)')
        raise Exception(('variable lts has to be an ndarray of shape (x_dim,)'
                         ' However, it is '), lts)


    if y is None:
        if not x is None:
            raise Exception(('provided latent state sequence x but not '
                             'observed data y.')) 
        if not u is None:
            raise Exception(('provided input sequence u but not '
                             'observed data y.')) 

    else: # i.e. if y is provided:
        if not (isinstance(y,np.ndarray) 
                and len(y.shape)==3 and y.shape[0]==y_dim):
            raise Exception(('When providing optional input y, it has to be '
                             'an np.ndarray of dimensions (y_dim,t_tot,n_tr). '
                             'It is not.'))
        else:
            t_tot = y.shape[1] # take these values from y and compare with 
            n_tr  = y.shape[2] # x, u, as we need consistency

    if not (x is None or (isinstance(x,np.ndarray) 
                          and len(x.shape)==3 
                          and x.shape[0]==x_dim
                          and x.shape[1]==t_tot
                          and x.shape[2]==n_tr) ):
        raise Exception(('When providing optional input x, it has to be an '
                         'np.ndarray of dimensions (x_dim,t_tot,n_tr). '
                         'It is not'))

    if not (u is None or (isinstance(u,np.ndarray) 
                          and len(u.shape)==3 
                          and u.shape[0]==u_dim
                          and u.shape[1]==t_tot
                          and u.shape[2]==n_tr) ):
        raise Exception(('When providing optional input x, it has to be an '
                         'np.ndarray of dimensions (x_dim,t_tot,n_tr). '
                         'It is not'))


    if (gen_C == 'PCA') or (gen_R == 'fractionObserved'):  
        cov_y = np.cov(y[:,:,0]-np.mean(y, (1,2)).reshape(y_dim,1)) 
        # Depending on the observation scheme, not all entries of the data
        # covariance are also interpretable, and the entries of cov_y for 
        # variable pairs (y_i,y_j) that were not observed together may indeed
        # contain NaN's of Inf's depending on the choice of representation of
        # missing data entries. Keep this in mind when selecting parameter 
        # initialisation methods such as gen_C=='PCA', which will work with 
        # the full matrix cov_y.
        # Note that the diagonal of cov_y should also be safe to use. 

    gen_pars_flags = np.ones(totNumParams, dtype=bool) # which pars to generate

    pars_out = {}

    """ parse (optional) user-provided true model parameters: """
    # allow comfortable use of dictionaries:
    if isinstance(pars_in, dict):

        if 'A' in pars_in:
            if np.all(pars_in['A'].shape==(x_dim,x_dim)): 
                pars_out['A']   = pars_in['A'].copy()
            else:
                raise Exception(('Bad initialization for LDS parameter A.'
                                 'Shape not matching dimensionality of x. '
                                 'Given shape of A is '), pars_in['A'].shape)
            gen_pars_flags[0] = False
        if 'B' in pars_in:
            if u_dim > 0:        
                if np.all(pars_in['B'].shape==(x_dim,u_dim)): 
                    pars_out['B']  = pars_in['B'].copy()
                else:
                    raise Exception(('Bad initialization for LDS parameter B.'
                                     'Shape not matching dimensionality of x,u'
                                     '. Given shape of B is '),pars_in[1].shape)
            else: # if we're not going to use B anyway, ...
                pars_out['B'] = np.array([0]) 
            gen_pars_flags[1] = False
        if 'Q' in pars_in:
            pars_out['Q'] = pars_in['Q']
            if np.all(pars_in['Q'].shape==(x_dim,x_dim)): 
                pars_out['Q']   = pars_in['Q'].copy()
            else:
                raise Exception(('Bad initialization for LDS parameter Q.'
                                 'Shape not matching dimensionality of x. '
                                 'Given shape of Q is '), pars_in[2].shape)
            gen_pars_flags[2] = False
        if 'mu0' in pars_in:
            if pars_in['mu0'].size==x_dim: 
                pars_out['mu0'] = pars_in['mu0'].copy()
            else:
                raise Exception(('Bad initialization for LDS parameter mu0.'
                                 'Shape not matching dimensionality of x. '
                                 'Given shape of mu0 is '), pars_in['mu0'].shape)
            gen_pars_flags[3] = False
        if 'V0' in pars_in:
            if np.all(pars_in['V0'].shape==(x_dim,x_dim)): 
                pars_out['V0'] = pars_in['V0'].copy()
            else:
                raise Exception(('Bad initialization for LDS parameter V0.'
                                 'Shape not matching dimensionality of x. '
                                 'Given shape of V0 is '), pars_in['V0'].shape)
            gen_pars_flags[4] = False
        if 'C' in pars_in:
            if np.all(pars_in['C'].shape==(y_dim,x_dim)):
                pars_out['C'] = pars_in['C'].copy()
            else:
                raise Exception(('Bad initialization for LDS parameter C.'
                                 'Shape not matching dimensionality of y, x. '
                                 'Given shape of C is '), pars_in['C'].shape)
            gen_pars_flags[5] = False
        if 'd' in pars_in:
            if np.all(pars_in['d'].shape==(y_dim,)):
                pars_out['d'] = pars_in['d'].copy()
            else:
                raise Exception(('Bad initialization for LDS parameter d.'
                                 'Shape not matching dimensionality of y, x. '
                                 'Given shape of d is '), pars_in['d'].shape)              
            gen_pars_flags[6] = False
        if 'R' in pars_in:
            if diag_R_flag:
                if np.all(pars_in['R'].shape==(y_dim,)):
                    pars_out['R'] = pars_in['R'].copy()
                elif np.all(pars_in['R'].shape==(y_dim,y_dim)):
                    pars_out['R'] = pars_in['R'].copy().diagonal()
                else:
                    raise Exception(('Bad initialization for LDS '
                                     'parameter R. Shape not matching '
                                     'dimensionality of y. '
                                     'Given shape of R is '),pars_in['R'].shape)                  
            else:
                if np.all(pars_in['R'].shape==(y_dim,y_dim)):
                    pars_out['R'] = pars_in['R'].diagonal().copy()
                else:
                    raise Exception(('Bad initialization for LDS '
                                     'parameter R. Shape not matching '
                                     'dimensionality of y. '
                                     'Given shape of R is '),pars_in['R'].shape)      
            gen_pars_flags[7] = False

    elif not pars_in is None:
        raise Exception('provided input parameter variable pars_in has to be '
                        'a dictionary with (optional) key-value pairs for '
                        'desired parameter initialisations. For no specified, '
                        'initialisations, use {} or None. However, pars_in is',
                         pars_in)

    """ fill in missing parameters (could be none, some, or all) """
    # generate latent state tranition matrix A
    if gen_pars_flags[0]:
        if lts is None:
            lts = np.random.uniform(size=[x_dim])
        if gen_A == 'diagonal':
            pars_out['A'] = np.diag(lts) # lts = latent time scales
        elif gen_A == 'full':
            pars_out['A'] = np.diag(lts) # lts = latent time scales
            while True:
                W    = np.random.normal(size=[x_dim,x_dim])
                if np.abs(np.linalg.det(W)) >0.001:
                    break
            pars_out['A'] = np.dot(np.dot(W, pars_out['A']), np.linalg.inv(W))
        elif gen_A == 'random':
            pars_out['A'] = np.random.normal(size=[x_dim,x_dim])            
        elif gen_A == 'zero':  # e.g. when fitting without dynamics
            pars_out['A'] = np.zeros((x_dim,x_dim))            
        else:
            raise Exception(('selected type for generating A not supported. '
                             'Selected type is '), gen_A)
    # There is inherent degeneracy in any LDS regarding the basis in the latent
    # space. Any rotation of A can be corrected for by rightmultiplying C with
    # the inverse rotation matrix. We do not wish to limit A to any certain
    # basis in latent space, but in a first approach may still initialise A as
    # diagonal matrix .     

    # generate latent state input matrix B
    if gen_pars_flags[1]:
        if gen_B == 'random':
            pars_out['B'] = np.random.normal(size=[x_dim,u_dim])            
        elif gen_B == 'zero': # make sure is default if use_B_flag=False
            pars_out['B'] = np.zeros((x_dim,u_dim))            
        else:

            raise Exception(('selected type for generating B not supported. '
                             'Selected type is '), gen_B)
    # Parameter B is never touched within the code unless use_B_flag == True,
    # hence we don't need to ensure its correct dimensionality if use_B_flag==False

    # generate latent state innovation noise matrix Q
    if gen_pars_flags[2]:                             # only one implemented standard 
        if gen_Q == 'identity':     # case: we can *always* rotate x
            pars_out['Q']    = np.identity(x_dim)  # so that Q is the identity 
        else:

            raise Exception(('selected type for generating Q not supported. '
                             'Selected type is '), gen_Q)
    # There is inherent degeneracy in any LDS regarding the basis in the latent
    # space. One way to counter this is to set the latent covariance to unity.
    # We don't hard-fixate this, as it prevents careful study of when stitching
    # can really work. Nevertheless, we can still initialise parameters Q as 
    # unity matrices without commiting to any assumed structure in the  final
    # innovation noise estimate. 
    # Note that the initialisation choice for Q should be in agreement with the
    # initialisation of C! For instance when setting Q to the identity and 
    # when getting C from PCA, one should also normalise the rows of C with
    # the sqrt of the variances of y_i, i.e. really whiten the assumed 
    # latent covariances instead of only diagonalising them.            

    # generate initial latent state mean mu0
    if gen_pars_flags[3]:
        if gen_mu0 == 'random':
            pars_out['mu0']  = np.random.normal(size=[x_dim])
        elif gen_mu0 == 'zero': 
            pars_out['mu0']  = np.zeros(x_dim)
        else:
            raise Exception(('selected type for generating mu0 not supported. '
                             'Selected type is '), gen_mu0)
    # generate initial latent state covariance matrix V0
    if gen_pars_flags[4]:
        if gen_V0 == 'identity': 
            pars_out['V0']   = np.identity(x_dim)  
        else:
            raise Exception(('selected type for generating V0 not supported. '
                             'Selected type is '), gen_V0)
    # Assuming long time series lengths, parameters for the very first time
    # step are usually of minor importance for the overall fitting result
    # unless they are overly restrictive. We by default initialise V0 
    # non-commitingly to the identity matrix (same as Q) and mu0 either
    # to all zero or with a slight random perturbation on that.   

    # generate emission matrix C
    if gen_pars_flags[5]:
        if gen_C == 'random': 
            pars_out['C'] = np.random.normal(size=[y_dim, x_dim])
        elif gen_C == 'PCA':
            if y is None:
                raise Exception(('tried to set emission matrix C from results '
                                 'of a PCA on the observed data without '
                                 'providing any data y'))            
            w, v = np.linalg.eig(cov_y-np.diag(R_0))                           
            w = np.sort(w)[::-1]                 
            # note that we also enforce equal variance for each latent dim. :
            pars_out['C'] = np.dot(v[:, range(x_dim)], 
                                np.diag(np.sqrt(w[range(x_dim)])))  
        else:
            raise Exception(('selected type for generating C not supported. '
                             'Selected type is '), gen_C)
    # C in many cases is the single-most important parameter to properly 
    # initialise. If the data is fully observed, a basic and powerful solution
    # is to use PCA on the full data covariance (after attributing a certain 
    # fraction of variance to R). In stitching contexts, this however is not
    # possible. Finding a good initialisation in the context of incomplete data
    # observation is not trivial. 

    # check for resulting stationary covariance of latent states x
    Pi    = sp.linalg.solve_discrete_lyapunov(pars_out['A'], 
                                              pars_out['Q'])
    Pi_t  = np.dot(pars_out['A'].transpose(),Pi)  # time-lagged cov(y_t,y_{t-1})
    CPiC = np.dot(pars_out['C'], np.dot(Pi, pars_out['C'].transpose())) 

    # generate emission noise covariance matrix R
    if gen_pars_flags[7]:
        if gen_R == 'fraction':
            # set R_ii as 25% to 125% of total variance of y_i
            pars_out['R'] =(0.25+np.random.uniform(size=[y_dim]))*CPiC.diagonal()
        elif gen_R == 'fractionObserved':
            if y is None:
                raise Exception(('tried to set emission noise covariance R as '
                                 'a fraction of data variance without '
                                 'providing any data y'))
            if y_dim>1:
                pars_out['R']   = 0.1 * cov_y.diagonal()  
            else:
                pars_out['R']   = 0.1 * np.array(cov_y).reshape(1,)

        elif gen_R == 'identity':
            gen_R = np.ones(y_dim)
        elif gen_R == 'zero':                        # very extreme case!
            gen_R = np.zeros(y_dim)
        else:
            raise Exception(('selected type for generating R not supported. '
                             'Selected type is '), gen_R)
    # C and R should not be initialised independently! Following on the idea
    # of (diagonal) R being additive private noise for the individual variables
    # y_i, we can initialise R as being a certain fraction of the observed 
    # noise. When initialising R from data, we have to be carefull not to
    # attribute too much noise to R, as otherwise the remaining covariance 
    # matrix cov(y)-np.diag(R) might no longer be positive definite!

    # generate emission noise covariance matrix d
    if gen_pars_flags[6]:
        if gen_d == 'scaled':
            pars_out['d'] = (np.sqrt(
                                    np.mean(
                                            np.diag( CPiC
                                                   + np.diag(pars_out['R']
                                                   )
                                            )
                                    )
                            )
                            * np.random.normal(size=y_dim))
        elif gen_d == 'random':
            pars_out['d'] = np.random.normal(size=y_dim)
        elif gen_d == 'zero':
            pars_out['d'] = np.zeros(y_dim)
        elif gen_d == 'mean':
            if y is None:
                raise Exception(('tried to set observation offset d as the '
                                 'data mean without providing any data y'))
            pars_out['d'] = np.mean(y,(1,2)) 
        else:
            raise Exception(('selected type for generating d not supported. '
                              'Selected type is '), gen_d)
    # A bad initialisation for d can spell doom for the entire EM algorithm,
    # as this may offset the estimates of E[x_t] far away from zero mean in
    # the first E-step, so as to capture the true offset present in data y. 
    # This in turn ruins estimates of the linear dynamics: all the eigenvalues
    # of A suddenly have to be close to 1 to explain the constant non-decaying
    # offset of the estimates E[x_t]. Hence the ensuing M-step will generate
    # a parameter solution that is immensely far away from optimal parameters,
    # and the algorithm most likely gets stuck in a local optimum long before
    # it found its way to any useful parameter settings (in fact, C and d of
    # the first M-step will adjust to the offset in the latent states and 
    # hence contribute to the EM algorithm sticking to latent offset and bad A)

    # collect options for debugging and experiment-tracking purposes
    options_init = {
                 'gen_A'   : gen_A,
                 'lts'    : lts,
                 'gen_B'   : gen_B,
                 'gen_Q'   : gen_Q,
                 'gen_mu0' : gen_mu0,
                 'gen_V0'  : gen_V0,
                 'gen_C'   : gen_C,
                 'gen_d'   : gen_d,
                 'gen_R'   : gen_R
                    }

    """ check validity (esp. of user-provided parameters), return results """

    return pars_out, options_init


#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def sim_data(pars,
             t_tot,n_tr=1,
             obs_scheme=None,
             u=None,input_type='pwconst',const_input_length=1):
    """ INPUT:
        pars  : (list of) LDS parameters
        t_tot : trial length (in number of time points)
        n_tr : number of trials
        obs_scheme: observation scheme for given data, stored in dictionary
                   with keys 'sub_pops', 'obs_time', 'obs_pops'
        u: data array of input variables
        input_type: string specifying the type of input data to be generated
        const_input_length: if input_type=='pwconst', gives the stretches of time
                            over which u will be generated as piecewise constant
        Generates data from an LDS model with ground-truth parameters pars. 
        Data will be of recording length t_tot, with n_tr many trials. 
        Input u is optional, and can also be generated by this function
        in case of simple piecewise constant input, when specifying the
        relevant input variables 'input_type' and 'constInpugLngth'. 
        A core task of with this function (that at core just calls 
        upon the ssm_timeSeries library) is to correctly differ between
        the cases of i) having no input u and ii) having input matrix B=0,
        in a way that will be understood by ssm_timeSeries.

    """
    x_dim = pars['A'].shape[0] # get x_dim from A
    y_dim = pars['C'].shape[0] # get y_dim from C

    if np.all(pars['R'].shape==(y_dim,)): # ssm_timeSeries assumes R to be a 
        pars = pars.copy()               # full y_dim-by-y_dim covariance matrix.
        pars['R'] = np.diag(pars['R'])   # Need to reshape

    if u is None:
        u_dim = 0
        gen_input_flag = False
    elif isinstance(u, np.ndarray) and u.shape[1]==t_tot and u.shape[2]==n_tr:
        u_dim = u.shape[0]
        gen_input_flag = False
    elif isinstance(u, numbers.Integral):
        u_dim = u
        gen_input_flag = True        

    if isinstance(pars['B'],np.ndarray) and pars['B'].size>0 and u_dim==0:
        print(('Warning: Parameter B is initialised, but u_dim = 0. '
               'Algorithm will ignore input for the LDS, outcomes may be '
               'not as expected.'))

    if gen_input_flag:
        if input_type=='pwconst': # generate piecewise constant input
            u = np.zeros((u_dim,t_tot,n_tr))
            for tr in range(n_tr):
                for i in range(int(np.floor(t_tot/const_input_length))):
                    idxRange = range((i-1)*const_input_length, i*const_input_length)
                    u[:,idxRange,tr] = np.random.normal(size=[1])
                u[:,:,tr] -= np.mean(u[:,:,tr])
        elif input_type=='random': # generate random Gaussian input
            u = np.random.normal(size=[u_dim,t_tot,n_tr])
            for tr in range(n_tr):
                u[:,:,tr] -= np.mean(u[:,:,tr])
        else:
            raise Exception(('selected option for input generation '
                             'not supported. It is possible to directly '
                             'hand over pre-computed inputs u.'))

    # reformat parameters into list, as is expected by ssm_timeSeries 
    pars_ts = [pars['A'],pars['B'],pars['Q'],
               pars['mu0'],pars['V0'],
               pars['C'],pars['d'],pars['R']]
    if u_dim > 0:
        seq = ts.setStateSpaceModel('iLDS',[x_dim,y_dim,u_dim],pars_ts) # initiate 
        seq.giveEmpirical().addData(n_tr,t_tot,[u],rngSeed)            # draw data
    else: 
        parsNoInput = pars_ts.copy()
        parsNoInput[1] = np.zeros((x_dim,1))
        seq = ts.setStateSpaceModel('iLDS',[x_dim,y_dim,1],parsNoInput) 
        seq.giveEmpirical().addData(n_tr,t_tot,None,rngSeed)          # draw data


    x = seq.giveEmpirical().giveData().giveTracesX()
    y = seq._empirical._data.giveTracesY()        

    return x,y,u

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def setup_fit_lds(
            y, 
            u, 
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

    def fit(x_dim, pars, obs_scheme=None, save_file=None):

        """ INPUT:
            x_dim:      dimensionality of (sole subgroup of) latent state X
            pars:      dictionary of parameters to start fitting. 
            obs_scheme: observation scheme for given data, stored in dictionary
                       with keys 'sub_pops', 'obs_time', 'obs_pops'
            Fits an LDS model to data.

        """

        return ssm_fit.fit_lds(y=y, 
                               u=u,
                               x_dim=x_dim,            
                               obs_scheme=obs_scheme,
                               pars=pars,
                               max_iter=max_iter,
                               epsilon=epsilon, 
                               eps_cov=eps_cov,
                               plot_flag=plot_flag, 
                               trace_pars_flag=trace_pars_flag, 
                               trace_stats_flag=trace_stats_flag, 
                               diag_R_flag=diag_R_flag,
                               use_A_flag=use_A_flag, 
                               use_B_flag=use_B_flag,
                               save_file=save_file)
    return fit
#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def do_e_step(pars, y, u=None, obs_scheme=None, eps=1e-30):
    """ INPUT:
        pars  : (list of) LDS parameters
        y: data array of observed variables
        u: data array of input variables
        obs_scheme: observation scheme for given data, stored in dictionary
                   with keys 'sub_pops', 'obs_time', 'obs_pops'
        eps:       precision (stopping criterion) for deciding on convergence
                   of latent covariance estimates durgin the E-step 
        Wrapper function to quickly compute a single E-step with given data y
        and LDS parameters pars. This function mostly exists to deal with the
        difference between having no input u and not using it (i.e. parameter
        B = 0). 
    """

    if (u is None and 
        not (pars['B'] is None or pars['B'] == [] or
             (isinstance(pars['B'],np.ndarray) and pars['B'].size==0) or
             (isinstance(pars['B'],np.ndarray) and pars['B'].size>0 
              and np.max(abs(pars['B']))==0) or
             (isinstance(pars['B'],(float,numbers.Integral)) 
              and pars['B']==0))):
        print(('Warning: Parameter B is initialised, but input u = None. '
               'Algorithm will ignore input for the LDS, outcomes may be '
               'not as expected.'))

    y_dim = y.shape[0]
    if isinstance(u, np.ndarray):
        u_dim = u.shape[0]
    else:
        u_dim = 0

    if obs_scheme is None:
        t_tot = y.shape[1]
        print('creating default observation scheme: Full population observed')
        obs_scheme = {'sub_pops': [list(range(y_dim))], # creates default case
                     'obs_time': [t_tot],               # of fully observed
                     'obs_pops': [0]}                   # population

    ssm_fit.check_pars(pars=pars,
                       y_dim=y_dim, 
                       u_dim=u_dim)                     

    stats, lltr, t_conv_ft, t_conv_sm = \
        ssm_fit.lds_e_step(pars=pars,
                           y=y,
                           u=u, 
                           obs_scheme=obs_scheme,
                           eps=eps)        

    return stats, lltr, t_conv_ft, t_conv_sm

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def do_first_em_cycle(pars_init, obs_scheme, y, u=None, eps=1e-30, 
                           use_A_flag=True,use_B_flag=False,diag_R_flag=True):
    """ INPUT:
        pars:      collection of initial parameters for LDS
        obs_scheme: observation scheme for given data, stored in dictionary
                   with keys 'sub_pops', 'obs_time', 'obs_pops'
        y:         data array of observed variables
        u:         data array of input variables
        eps:       precision (stopping criterion) for deciding on convergence
                   of latent covariance estimates durgin the E-step                   
        This function serves to quickly get the results of one EM-cycle. It is
        mostly intended to generate results that can quickly be compared with
        other EM implementations or different parameter initialisation methods. 
    """

    y_dim = y.shape[0]
    u_dim = ssm_fit._get_u_dim(u)
    t_tot = y.shape[1]

    pars_init = dict(pars_init)
    ssm_fit.check_pars(pars=pars_init, 
                       y_dim=y_dim, 
                       u_dim=u_dim)         
    obs_scheme = dict(obs_scheme)
    ssm_fit.check_obs_scheme(obs_scheme=obs_scheme,
                             y_dim=y_dim,
                             t_tot=t_tot)                                   

    # do one E-step
    stats_init, ll_init, t_conv_ft, t_conv_sm    = \
      ssm_fit.lds_e_step(pars=pars_init,
                         y=y,
                         u=u, 
                         obs_scheme=obs_scheme,
                         eps=eps)        

    # do one M-step      
    pars_first = ssm_fit.lds_m_step(stats=stats_init,
                                    y=y, 
                                    u=u,
                                    obs_scheme=obs_scheme,
                                    use_A_flag=use_A_flag,
                                    use_B_flag=use_B_flag,
                                    diag_R_flag=diag_R_flag)

    # do another E-step
    stats_first, ll_first, t_conv_ft, t_conv_sm = \
      ssm_fit.lds_e_step(pars=pars_first,
                         y=y,
                         u=u, 
                         obs_scheme=obs_scheme,
                         eps=eps)        

    return stats_init, ll_init, pars_first, stats_first, ll_first

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def evaluate_fit(pars, y, u, ext, extxt, extxtm1, lls):
    """ TO BE EXTENDED """

    t_tot = y.shape[1]
    n_tr  = y.shape[2]
    x_dim = A.shape[0]
    y_dim = C.shape[0]
    u_dim = u.shape[0]

    Pi_h    = sp.linalg.solve_discrete_lyapunov(pars['A'], 
                                                pars['Q'])
    Pi_t_h  = np.dot(pars['A'].transpose(), Pi_h)


    dataCov  = np.cov(y[:,0:t_tot-1,0], y[:,1:t_tot,0])
    cov_yy    = dataCov[np.ix_(np.arange(0, y_dim), np.arange(0,     y_dim))]
    cov_yy_m1 = dataCov[np.ix_(np.arange(0, y_dim), np.arange(y_dim,2*y_dim))]

    plt.figure(1)
    cmap = matplotlib.cm.get_cmap('brg')
    clrs = [cmap(i) for i in np.linspace(0, 1, x_dim)]
    for i in range(x_dim):
        plt.subplot(x_dim,1,i)
        plt.plot(x[i,:,0], color=clrs[i])
        plt.hold(True)
        if (np.mean( np.square(x[i,:,0] - ext_h[i,:,0]) ) < 
            np.mean( np.square(x[i,:,0] + ext_h[i,:,0]) )  ):
            plt.plot( ext_h[i,:,0], color=clrs[i], ls=':')
        else:
            plt.plot(-ext_h[i,:,0], color=clrs[i], ls=':')

    m = np.min([Pi_h.min(), cov_yy.min()])
    M = np.max([Pi_h.max(), cov_yy.max()])       
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(np.dot(np.dot(pars['C'], Pi_h),pars['C'].transpose())+pars['R'], 
               interpolation='none')
    plt.title('cov_hat(y_t,y_t)')
    plt.clim(m,M)
    plt.subplot(1,2,2)
    plt.imshow(cov_yy,    interpolation='none')
    plt.title('cov_emp(y_t,y_t)')
    plt.clim(m,M)

    plt.figure(2)
    m = np.min([cov_yy_m1.min(), Pi_t_h.min()])
    M = np.max([cov_yy_m1.max(), Pi_t_h.max()])
    plt.subplot(1,2,1)
    plt.imshow(np.dot(np.dot(pars['C'], Pi_t_h), pars['C'].transpose()), 
               interpolation='none')
    plt.title('cov_hat(y_t,y_{t-1})')
    plt.clim(m,M)
    plt.subplot(1,2,2)
    plt.imshow(cov_yy_m1,    interpolation='none')
    plt.title('cov(y_t,y_{t-1})')
    plt.clim(m,M)

    plt.figure(3)
    plt.plot(np.sort(np.linalg.eig(pars['A'])[0]), 'r')

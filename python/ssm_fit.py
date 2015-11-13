import numpy as np
import scipy as sp
from scipy import stats
import numbers       # to check for numbers.Integral, for isinstance(x, int)

import matplotlib
import matplotlib.pyplot as plt
from IPython import display  # for live plotting in jupyter

from scipy.io import savemat # store intermediate results 

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _getInitPars(y, u, xDim, obsScheme, ifUseB,
                 initA   ='random',
                 initB   ='random',
                 initQ   ='unity',
                 initmu0 ='zero',
                 initV0  ='unity',
                 initC   ='random',
                 initd   ='mean',
                 initR   ='fraction',
                 ifVisualiseInitPars = True,
                 ):
    """ OUT = _getInitPars(y*,u*,xDim*,obsScheme*,ifUseB*,
                           initA,initB,initQ,initmu0,initV0,initC,initd,initR,
                           ifVisualiseInitPars)
        y:         data array of observed variables
        u:         data array of input variables
        xDim:      dimensionality of (sole subgroup of) latent state X
        obsScheme: observation scheme for given data, stored in dictionary
                   with keys 'subpops', 'obsTimes', 'obsPops'
        initA   : string specifying methods of parameter initialisation
        initB   :  ""
        initQ   :  ""
        initmu0 :  "" 
        initV0  :  ""
        initC   :  "" 
        initd   :  ""
        initR   : (see below for details)
        ifVisualiseInitPars : boolean, specifying whether to plot the outputs
                              of this function for inspection
        Initialises parameters of an LDS, potentially by looking at the data.

    """
    yDim  = y.shape[0]
    T     = y.shape[1]
    Trial = y.shape[2]
    if isinstance(u,np.ndarray) and u.shape[1]==T and u.shape[2]==Trial:
        uDim = u.shape[0]
    else:
        uDim = 0        

    covy = np.cov(y[:,:,0]-np.mean(y, (1,2)).reshape(yDim,1)) 
    # Depending on the observation scheme, not all entries of the data 
    # covariance are also interpretable, and the entries of covy for pairs of 
    # variables (y_i,y_j) that were not observed together may indeed contain
    # NaN's depending on the choice of representation of missing data entries.
    # Keep this in mind when selecting parameter initialisation methods such
    # as options['initC']=='PCA', which will work with the full matrix covy.
    # Note that the diagonal of covy should also be safe to use. 

    # Latent dynamics matrix A
    if initA == 'random':
        A_0   = np.diag(np.random.uniform(size=[xDim]))
    if initA == 'zero':
        A_0   = np.zeros([xDim,xDim])
    # There is inherent degeneracy in any LDS regarding the basis in the latent
    # space. Any rotation of A can be corrected for by rightmultiplying C with
    # the inverse rotation matrix. We do not wish to limit A to any certain
    # basis in latent space, but in a first approach may still initialise A as
    # diagonal matrix .     

    # Input matrix B for input to latent dynamics
    if uDim > 0 and ifUseB and initB == 'random':
        B_0   = np.random.normal(size=[xDim, uDim])   
    elif not ifUseB:
        B_0   = 0
    elif uDim == 0:
        B_0 = np.zeros([xDim, uDim])
        print(('Warning: Latent input parameter B was innitialised with '
               'shape (xDim,0), and is flagged to be used (ifUseB = True).'))
    # Parameter B is never touched within the code unless ifUseB == True,
    # hence we don't need to ensure its correct dimensionality if ifUseB==False

    # Innovation noise matrix Q
    if initQ == 'unity':
        Q_0   = 3 * np.identity(xDim)              
    # There is inherent degeneracy in any LDS regarding the basis in the latent
    # space. One way to counter this is to set the latent covariance to unity.
    # We do not do this, as it prevents careful study of when stitching can
    # really work. Nevertheless, we can still initialise parameters Q as 
    # unity matrices without commiting to any assumed structure in the  final
    # innovation noise estimate. 
    # Note that the initialisation choice for Q should be in agreement with the
    # initialisation of C! For instance when setting Q to the identity and 
    # when getting C from PCA, one should also normalise the rows of C with
    # the sqrt of the variances of y_i, i.e. really whiten the assumed 
    # latent covariances instead of only diagonalising them.

    # Mean mu0 and covariance V0 of latent Markov chain initial element x_0
    if initmu0 == 'zero':
        mu0_0 = np.zeros(xDim)   
    elif initmu0 == 'random':
        mu0_0 = np.random.normal(size=[xDim])   

    if initV0 == 'unity':
        V0_0  = np.identity(xDim)               
    # Assuming long time series lengths, parameters for the very first time
    # step are usually of minor importance for the overall fitting result
    # unless they are overly restrictive. We by default initialise V0 
    # non-commitingly to the identity matrix (same as Q) and mu0 either
    # to all zero or with a slight random perturbation on that.   

    # Emission mixture matrix C and (privat) emission noise variances R
    if initR == 'fraction':
        R_0   = 0.1 * covy.diagonal()
    if initC == 'PCA':
        w, v = np.linalg.eig(covy-np.diag(R_0))                           
        w = np.sort(w)[::-1] # eigenvalues not always sorted according to numpy                                              
        C_0 = np.dot(v[:, range(xDim)], np.diag(np.sqrt(w[range(xDim)])))  
    elif initC == 'random':
        C_0 = np.random.normal(size=[yDim,xDim])
    # C and R should not be initialised independently!
    # C in many cases is the single-most important parameter to properly 
    # initialise. If the data is fully observed, a basic and powerful solution
    # is to use PCA on the full data covariance (after attributing a certain 
    # fraction of variance to R). In stitching contexts, this however is not
    # possible. Finding a good initialisation in the context of incomplete data
    # observation is not trivial. 

    # Emission offset d
    if initd == 'mean':
        d_0    = np.mean(y,(1,2)) 
    if initd == 'zero':            # can lead to really bad results (see below)
        d_0    = np.zeros(yDim) 
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

    # wrap up initialisation parameters for documentation
    initOptions = {
                 'initA'   : initA,
                 'initB'   : initB,
                 'initQ'   : initQ,
                 'initmu0' : initmu0,
                 'initV0'  : initV0,
                 'initC'   : initC,
                 'initd'   : initd,
                 'initR'   : initR,
                 'ifVisualiseInitPars' : ifVisualiseInitPars,
                    }

    if ifVisualiseInitPars:
        plt.figure(1, figsize=(15,10))
        if initC == 'PCA':
            plt.subplot(1,2,1)
            plt.plot(w)
            plt.title('spectrum of data covariance matrix (R_0 removed)')         
            plt.subplot(1,2,2)
            plt.imshow(C_0, interpolation='none')
            plt.title('C_0 : initialisation for C')
            plt.ylabel('y_i')
            plt.xlabel('x_j')
        else:   
            plt.imshow(C_0.transpose(), interpolation='none')
            plt.title('C_0 : initialisation for C (transposed!)')
            plt.xlabel('y_i')
            plt.ylabel('x_j')
        plt.figure(2, figsize=(15,10))
        plt.subplot(1,2,1)
        plt.imshow(covy, interpolation='none')
        plt.title('true covariance matrix')
        plt.subplot(1,2,2)
        plt.imshow(np.dot(C_0, C_0.transpose()), interpolation='none')
        plt.title('C_0 * C_0^T')

    initPars = [A_0,B_0,Q_0,mu0_0,V0_0,C_0,d_0,R_0] 

    return [initPars, initOptions]

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _fitLDS(y, 
            u,
            obsScheme,
            initPars,
            fitoptions,
            xDim=None,
            saveFile=None):
    """ OUT = _fitLDS(y*,obsScheme*,initPars, maxIter, epsilon, 
                    ifPlotProgress,xDim)
        y:         data array of observed variables
        u:         data array of input variables
        obsScheme: observation scheme for given data, stored in dictionary
                   with keys 'subpops', 'obsTimes', 'obsPops'
        initPars:  set of parameters to start fitting. If == None, the 
                   parameters currently stored in the model will be used,
                   otherwise needs initPars = [A,Q,mu0,V0,C,R]
        fitoptions: dictionary containing fitting options
        -maxIter:   maximum allowed iterations for iterative fitting (e.g. EM)
        -epsilon:   convergence criterion, e.g. difference of log-likelihoods
        -ifPlotProgress: boolean, specifying if fitting progress is visualized
        -ifTraceParamHist: boolean, specifying if entire parameter updates 
                           history or only the current state is kept track of 
        -ifRDiagonal: boolean, specifying diagonality of observation noise
        xDim:      dimensionality of (sole subgroup of) latent state X
        Fits an LDS model to data.

    """
    yDim  = y.shape[0] # Data organisation convention (also true for input u): 
    T     = y.shape[1] # yDim is dimensionality of observed data, T is trial
    Trial = y.shape[2] # length (in bins), Trial is number of trials (with
                       # idential trial length, observation structure etc.)

    # check observation scheme
    # The observation scheme is crucial to both the sitching context and to
    # any missing data in y! One should always check if the provided 
    # observation scheme is the intended one.
    if obsScheme is None:
        obsScheme = {'subpops': [list(range(yDim))], # creates default case
                     'obsTime': [T],                 # of fully observed
                     'obsPops': [0]}                 # population
    else: 
        try:
            obsScheme['subpops'] # check for the 
            obsScheme['obsTime'] # fundamental
            obsScheme['obsPops'] # information
        except:                   # have to give hard error here !
            print('obsScheme')
            print(obsScheme)
            raise Exception(('provided observation scheme is insufficient. '
                             'It requires the fields subpops, obsTime and '
                             'obsPops. Not all those fields were given.'))
    try:
        obsScheme['obsIdxG']     # check for addivional  
        obsScheme['idxgrps']     # (derivable) information
    except:                       # can fill in if missing !
        [obsIdxG, idxgrps] = _computeObsIndexGroups(obsScheme,yDim)
        obsScheme['obsIdxG'] = obsIdxG # add index groups and 
        obsScheme['idxgrps'] = idxgrps # their occurences                                
    
    # unpack some of the fitting options to keep code uncluttered
    maxIter           = fitoptions['maxIter']
    epsilon           = fitoptions['epsilon']
    ifPlotProgress    = fitoptions['ifPlotProgress']
    ifTraceParamHist  = fitoptions['ifTraceParamHist']
    ifRDiagonal       = fitoptions['ifRDiagonal']     
    covConvEps        = fitoptions['covConvEps']
    ifFitA            = fitoptions['ifFitA']

    if not (isinstance(maxIter, numbers.Integral) and maxIter > 0):
        print('maxIter:')
        print(maxIter)
        raise Exception('argument maxIter has to be a positive integer')

    if (not (isinstance(epsilon, (float, numbers.Integral)) and
            epsilon > 0) ):
        print('epsilon:')
        print(epsilon)
        raise Exception('argument epsilon has to be a positive number')
            
    if not isinstance(ifPlotProgress, bool):
        print('ifPlotProgress:')
        print(ifPlotProgress)
        raise Exception('argument ifPlotProgress has to be a boolean')

    if not isinstance(ifTraceParamHist, bool):
        print('ifTraceParamHist:')
        print(ifTraceParamHist)
        raise Exception('argument ifTraceParamHist has to be a boolean')
     
    if not isinstance(ifRDiagonal, bool):
        print('ifRDiagonal:')
        print(ifRDiagonal)
        raise Exception('argument ifRDiagonal has to be a boolean')   

    if not isinstance(ifFitA, bool):
        print('ifFitA:')
        print(ifFitA)
        raise Exception('argument ifFitA has to be a boolean')   

    if (not isinstance(covConvEps,(float,numbers.Integral))
        or not covConvEps >= 0):
        print('covConvEps:')
        print(epsilon)
        raise Exception(('covConvEps has to be a non-negative number'))

    if not fitoptions['ifUseB']:
        u = None
        uDim = 0
    elif (isinstance(u, np.ndarray) and len(u.shape)==3
          and u.shape[1]==y.shape[1] and u.shape[2]==y.shape[2]):
        uDim = u.shape[0]        # i.e. ifUseB = True, and u is good
    else:                        # i.e. ifUseB = True, but u is bad
        if isinstance(u, np.ndarray):
            print('u.shape')
            print( u.shape )
        else:
            print('u')
            print( u )
        raise Exception(('If provided, input data u has to be an array with '
                         'three dimensions, (uDim,T,Trial). To not include '
                         'any input data, set ifUseB = False.'))

    [A,B,Q,mu0,V0,C,d,R,xDim] = _unpackInitPars(initPars,
                                                uDim,yDim,None,
                                                ifRDiagonal)  

    A_0 = A.copy()

    E_step = _LDS_E_step 
    M_step = _LDS_M_step 

    # evaluate initial state       
    print('convergence criterion for E-step (tolerance on matrix changes):')
    print(covConvEps)
    [Ext, Extxt, Extxtm1, LLtr, tCovConvFt, tCovConvSm] = E_step(
                                         A,B,Q,mu0,V0,C,d,R,y,u, 
                                         obsScheme,covConvEps)
    LL_new = np.sum(LLtr)
    LL_old = -float('Inf')
    dLL = []              # performance trace for status plotting
    log10 = np.log(10)    # for convencience, see below
    LLs = [LL_new.copy()] # performance trace to be returned
    stepCount = 0        
    ifBreakLoop = False

    # start EM iterations, run until convergence 
    if ifTraceParamHist:
        As   = [A]
        Bs   = [B]
        Qs   = [Q]
        mu0s = [mu0]
        V0s  = [V0]
        Cs   = [C]
        ds   = [d]
        Rs   = [R]
        Exts    = [Ext]
        Extxts  = [Extxt]
        Extxtm1s= [Extxtm1]
    
    while LL_new - LL_old > epsilon and stepCount < maxIter:

        LL_old = LL_new
        
        sy     = None #
        syy    = None # initialize, then copy results, as
        suu    = None # there is no need to compute 
        suuinv = None # these values twice
        Ti     = None #                                 
        
        [A,B,Q,mu0,V0,C,d,R,sy,syy,suu,suuinv,Ti] = M_step(
                                             Ext, 
                                             Extxt, 
                                             Extxtm1,
                                             y, 
                                             u,
                                             obsScheme,
                                             sy,
                                             syy,
                                             suu,
                                             suuinv,
                                             Ti)

        if not ifFitA:
            A = A_0 # diry solution: just revert to initialisation


        # store intermediate results for each time step
        if ifTraceParamHist:
            As.append(A.copy())
            Bs.append(B.copy())
            Qs.append(Q.copy())
            mu0s.append(mu0.copy())
            V0s.append(V0.copy())
            Cs.append(C.copy())
            ds.append(d.copy())
            Rs.append(R.copy())    
            Exts.append(Ext.copy())
            Extxts.append(Extxt.copy())
            Extxtm1s.append(Extxtm1.copy())
        
        stepCount += 1            

        if not saveFile is None:
            np.savez(saveFile+'_tmp_'+str(stepCount),
                     stepCount,LLtr,  
                     A,B,Q,mu0,V0,C,d,R,
                     covConvEps,tCovConvFt,tCovConvSm)
        
        [Ext, Extxt, Extxtm1, LLtr, tCovConvFt, tCovConvSm] = E_step(
                                             A, 
                                             B,
                                             Q, 
                                             mu0, 
                                             V0, 
                                             C,
                                             d,
                                             R,
                                             y,
                                             u,
                                             obsScheme,
                                             covConvEps)


        LL_new = np.sum(LLtr) # discarding distinction between trials
        LLs.append(LL_new.copy())

        if LL_new < LL_old:
            print('LL_new - LL_old')
            print( LL_new - LL_old )
            print(('WARNING! Lower bound decreased during EM '
                   'algorithm. This is impossible for an LDS. '
                   'Continue?'))
            #print('Press Y to continue or N to cancel')
            #inp = input("Enter (y)es or (n)o: ")
            #if inp == "no" or inp.lower() == "n":
            #    return None # break EM loop because st. is wrong

        dLL.append(LL_new - LL_old)
        if ifPlotProgress:
            # dynamically plot log of log-likelihood difference
            plt.clf()
            plt.figure(1,figsize=(15,15))
            plt.subplot(1,2,1)
            plt.plot(LLs)
            plt.xlabel('#iter')
            plt.ylabel('log-likelihood')
            plt.subplot(1,2,2)
            plt.plot(dLL)
            plt.xlabel('#iter')
            plt.ylabel('LL_{new} - LL_{old}')
            #plt.subplot(2,2,3)
            #plt.plot(Ext[0,range(200),0])
            #plt.xlabel('n')
            #plt.ylabel('first component of x_n')
            display.display(plt.gcf())
            display.clear_output(wait=True)
            
    LLs = np.array(LLs)     
    
    if ifTraceParamHist:    
        return [As, Bs, Qs, mu0s, V0s, Cs, ds, Rs, LLs]
    else:
        return [[A], [B], [Q], [mu0], [V0], [C], [d], [R], LLs]

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _LDS_E_step(A,B,Q,mu0,V0,C,d,R,y,u,obsScheme,eps=0): 
    """ OUT = _LDS_E_step(A*,Q*,mu0*,V0*,C*,R*,y*,obsScheme*,ifRDiagonal)
    see Bishop, 'Pattern Recognition and Machine Learning', ch. 13
    for formulas and input/output naming conventions
    The variable obsScheme is a dictionary that contains information
    on observed subpopulations of y that are observed at each time point.
    Be careful when inspecting the intermediate variables such as mu, or V in
    isolation. The functions are written to work fast, so to save on some 
    copying of variables from one array to another, arrays maybe reused under
    a different name. It is e.g. (mu is mu_h)==True

    """ 
    try:
        subpops = obsScheme['subpops'];
        obsTime = obsScheme['obsTime'];
        obsPops = obsScheme['obsPops'];
    except:
        print('obsScheme:')
        print(obsScheme)
        raise Exception(('provided obsScheme dictionary does not have '
                         'the required fields: subpops, obsTimes, '
                         'and obsPops.'))

    Bu = np.zeros([A.shape[0], y.shape[1], y.shape[2]])
    if (isinstance(u, np.ndarray) and u.size>0 
        and u.shape[1]==y.shape[1] and u.shape[2]==y.shape[2]):
        for tr in range(y.shape[2]):           # i.e. if u is e.g. empty, we 
            Bu[:,:,tr] = np.dot(B, u[:,:,tr])  # just leave B*u = 0!

    [mu,V,P,Pinv,logc,tCovConvFt] = _KalmanFilter(A,Bu,Q,mu0,V0,C,d,R,
                                                  y,obsScheme,eps)    
    [mu_h,V_h,J,tCovConvSm]           = _KalmanSmoother(A,Bu,mu,V,P,Pinv,
                                                        obsTime,tCovConvFt,
                                                        eps)
    [Ext,Extxt,Extxtm1] = _KalmanParsToMoments(mu_h,V_h,J,
                                               obsTime,tCovConvFt,tCovConvSm)
        
    LL = np.sum(logc,axis=0) # sum over times, get Trial-dim. vector
    
    return [Ext, Extxt, Extxtm1, LL, tCovConvFt, tCovConvSm]

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _KalmanFilter(A,Bu,Q,mu0,V0,C,d,R,y,obsScheme,eps=0):
    """ OUT = _KalmanFilter(A*,Bu*,Q*,mu0*,V0*,C*,d*,R*,y*,obsScheme*,eps)
    see Bishop, 'Pattern Recognition and Machine Learning', ch. 13
    for formulas and input/output naming conventions   
    The variable obsScheme is a dictionary that contains information
    on observed subpopulations of y that are observed at each time point
    This implementation assumes R to be diagonal!
    """
    xDim  = A.shape[0]
    yDim  = y.shape[0]
    T     = y.shape[1]
    Trial = y.shape[2]

    mu    = np.zeros([xDim,     T,Trial])
    V     = np.zeros([xDim,xDim,T,Trial])
    P     = np.zeros([xDim,xDim,T,Trial])
    Pinv  = np.zeros([xDim,xDim,T,Trial]) # also needed for Kalman smoothing!
    logc  = np.zeros([          T,Trial])

    try:
        subpops = obsScheme['subpops'];
        obsTime = obsScheme['obsTime'];
        obsPops = obsScheme['obsPops'];
    except:
        print('obsScheme:')
        print(obsScheme)
        raise Exception(('provided obsScheme dictionary does not have '
                         'the required fields: subpops, obsTimes, '
                         'and obsPops.'))
        
    if np.all(R.shape == (yDim,yDim)):
        R = R.diagonal()
    elif not np.all(R.shape == (yDim,)):
        print('yDim:')
        print( yDim  )
        print('R.shape:')
        print( R.shape  )
        raise Exception(('Variable R is assumed to be diagonal. '
                         'Please provide the diagonal entries as'
                         ' (yDim,)-array')) 

    xRange = range(xDim)
    Atr = A.transpose()        
    Iq = np.identity(xDim)

    # estimates of posterior covariances converge fast, so we want to check for
    # this and stop computing them after a while
    tCovConvFt = np.array(obsTime)-1 # initialise with latest possible time
    ifCovConv  = False
    
    for tr in range(Trial):

        # first time step: [mu0,V0] -> [mu1,V1]
        idx = subpops[obsPops[0]]

        if len(idx) > 0:
            # pre-compute for this group of observed variables
            Cj   = C[np.ix_(idx,xRange)]                    # all these
            Rinv = 1/R[idx]                                 # operations    
            CtrRinv = Cj.transpose() * Rinv                 # are order
            CtrRinvC = np.dot(CtrRinv, Cj)                  # O(yDim) !  

            # pre-compute for this time step   
            mu0B0  = mu0+Bu[:,0,tr]                                        
            Cmu0B0 = np.dot(Cj,mu0B0) # O(yDim)
            yDiff  = y[idx,0,tr] - d[idx] - Cmu0B0          # O(yDim)   

            CtrRyDiff_Cmu0 = np.dot(CtrRinv, yDiff)         # O(yDim)
            P0   = V0 # = np.dot(np.dot(A, V0), Atr) + Q
                    
            # compute Kalman gain components
            P0inv   = sp.linalg.inv(P0)                
            Kcore  = sp.linalg.inv(CtrRinvC+P0inv)                                                      
            Kshrt  = Iq  - np.dot(CtrRinvC, Kcore)
            PKsht  = np.dot(P0,    Kshrt) 
            KC     = np.dot(PKsht, CtrRinvC)        
            
            # update posterior estimates
            mu[ :,0,tr] = mu0B0 + np.dot(PKsht,CtrRyDiff_Cmu0)
            V[:,:,0,tr] = np.dot(Iq - KC, P0)
            P[:,:,0,tr] = np.dot(np.dot(A,V[:,:,0,tr]), Atr) + Q
            Pinv[:,:,0,tr] = sp.linalg.inv(P[:,:,0,tr])
            #print('filter -1 touched t =' + str(0))

            # compute marginal probability y_0
            M    = sp.linalg.cholesky(P0)
            logdetCPCR    = (  np.sum(np.log(R[idx])) 
                             + np.log(sp.linalg.det(
                                   Iq + np.dot(M.transpose(),np.dot(CtrRinvC,M))))
                            )
            logc[ 0,tr] = (  np.sum(Rinv * yDiff * yDiff)       
                           - np.dot(CtrRyDiff_Cmu0, np.dot(Kcore, CtrRyDiff_Cmu0)) 
                           + logdetCPCR
                          )
        else:  # no input at all, needs to be rewritten (would also be much faster)                                                
            mu0B0  = mu0+Bu[:,0,tr]
            P0 = V0
            mu[ :,0,tr] = mu0B0 # no input, just adding zero-mean innovation noise
            V[:,:,0,tr] = P0  # Kalman gain is zero
            P[:,:,0,tr] = np.dot(np.dot(A,V[:,:,0,tr]), Atr) + Q  
            Pinv[:,:,0,tr] = sp.linalg.inv(P[:,:,0,tr])          
            logc[ 0, tr] = 0   # setting log(N(y|0,Inf)) = log(1)
            #print('filter 0 touched t =' + str(0))

                
        t = 1 # now start with second time step ...
        for i in range(len(obsTime)):
            idx = subpops[obsPops[i]]

            if len(idx) > 0:                                       
                # pre-compute for this group of observed variables
                Cj   = C[np.ix_(idx,xRange)]                    # all these
                Rj   = R[idx]                                   # operations                        PRECOMPUTE AND TABULARIZE THESE
                CtrRinv = Cj.transpose() / Rj                   # are order
                CtrRinvC = np.dot(CtrRinv, Cj)                  # O(yDim) !

                if ifCovConv: # if we stopped tracking those due to convergence
                    P[:,:,t-1,tr]    =    P[:,:,tCovConvFt[i-1],tr] # need to drag
                    Pinv[:,:,t-1,tr] = Pinv[:,:,tCovConvFt[i-1],tr] # these forward
                    ifCovConv = False # reset convergence flag                                      GIVE CONVERGENCE TO THE OTHER CODE, AS WELL

                while t < obsTime[i]: 
                                                       
                    # pre-compute for this time step                                   
                    AmuBu  = np.dot(A,mu[:,t-1,tr]) + Bu[:,t, tr] 
                    yDiff  = y[idx,t,tr] - d[idx] - np.dot(Cj,AmuBu) # O(yDim)                                              
                    CtrRyDiff_CAmu = np.dot(CtrRinv, yDiff)          # O(yDim)
                                                       
                    if not ifCovConv:                                       
                        # compute Kalman gain components
                        Kcore  = sp.linalg.inv(CtrRinvC+Pinv[:,:,t-1,tr])                                        
                        Kshrt  = Iq  - np.dot(CtrRinvC, Kcore)
                        PKsht  = np.dot(P[:,:,t-1,tr],  Kshrt) 
                        KC     = np.dot(PKsht, CtrRinvC)
                        # update posterior covariances
                        V[:,:,t,tr] = np.dot(Iq - KC,P[:,:,t-1,tr])
                        P[:,:,t,tr] = np.dot(np.dot(A,V[:,:,t,tr]), Atr) + Q
                        Pinv[:,:,t,tr] = sp.linalg.inv(P[:,:,t,tr])
                        # compute normaliser for marginal probabilties of y_t
                        M      = sp.linalg.cholesky(P[:,:,t-1,tr])                                                     
                        logdetCPCR = (  np.sum(np.log(Rj))                                  
                                   + np.log(sp.linalg.det(Iq+np.dot(M.transpose(),
                                                             np.dot(CtrRinvC,M))))
                                     )
                        if np.mean(np.power(P[:,:,t,tr]-P[:,:,t-1,tr],2)) < eps:
                            tCovConvFt[i] = t
                            ifCovConv   = True

                    # update posterior mean
                    mu[ :,t,tr] = AmuBu + np.dot(PKsht,CtrRyDiff_CAmu)
                    #print('filter 1 touched t =' + str(t))

                    # compute marginal probability y_t | y_0, ..., y_{t-1}
                    logc[ t,tr] = (  np.sum((yDiff * yDiff) / Rj)   
                                   - np.dot(CtrRyDiff_CAmu, np.dot(Kcore, 
                                            CtrRyDiff_CAmu))
                                   + logdetCPCR
                                  )            
                    t += 1

            else:  # no input at all, needs to be rewritten (would also be much faster)

                while t < obsTime[i]: 

                    AmuBu  = np.dot(A,mu[:,t-1,tr]) + Bu[:,t, tr] 
                    mu[ :,t,tr] = AmuBu # no input, just adding zero-mean innovation noise
                    V[:,:,t,tr] = P[:,:,t-1,tr]  # Kalman gain is zero
                    P[:,:,t,tr] = np.dot(np.dot(A,V[:,:,t,tr]), Atr) + Q  
                    Pinv[:,:,t,tr] = sp.linalg.inv(P[:,:,t,tr])          
                    logc[ t, tr] = 0   # setting log(N(y|0,Inf)) = log(1)

                    #print('filter 2 touched t =' + str(t))
                    t += 1

            # copy posterior covariances from the time we stopped updating
            #if tCovConvFt[i]<obsTime[i]:
            #    V[:,:,range(tCovConvFt[i],obsTime[i]),tr] = \
            #        V[:,:,tCovConvFt[i],tr].reshape(xDim,xDim,1)
            #    P[:,:,range(tCovConvFt[i],obsTime[i]),tr] = \
            #        P[:,:,tCovConvFt[i],tr].reshape(xDim,xDim,1)
            #    Pinv[:,:,range(tCovConvFt[i],obsTime[i]),tr] = \
            #        Pinv[:,:,tCovConvFt[i],tr].reshape(xDim,xDim,1)
                                     
    logc = -1/2 * (logc + yDim * np.log(2*np.pi))
    
    return [mu,V,P,Pinv,logc,tCovConvFt]

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _KalmanSmoother(A, Bu, mu, V, P, Pinv, obsTime, tCovConvFt, eps=0):        
    """ OUT = _KalmanSmoother(A*,Bu*,mu*,V*,P*)
    see Bishop, 'Pattern Recognition and Machine Learning', ch. 13
    for formulas and input/output naming conventions   
    """        
    T     = mu.shape[1]
    Trial = mu.shape[2]
    mu_h = mu
    V_h  = V.copy()
    J    = np.zeros([mu.shape[0],mu.shape[0],T,Trial])
    Atr = A.transpose()

    # the smoother runs backwards in time and can converge after several time 
    # steps of running, i.e. some steps before the end of the time series 
    # segment. The very first few steps before the Kalman filter has converged, 
    # however, the smoother also cannot converge, as it depends on the filter 
    # outputs. The time of convergence for the smoother for each subpopulation
    # is in between tCovConvFt[i] and obsTime[i]. 

    tCovConvSm = tCovConvFt.copy()+1 # initialise with earliest possible time
    if tCovConvFt[-1]==T-1:  # if V,P did not converge for the last subpop
        tCovConvSm[-1] = T-1 # this would now be T, which is index out of bound 

    for tr in range(Trial):
        AmuBu = np.dot(A, mu[:,:,tr]) + Bu[:,:,tr]

        V_h[:,:,T-1,tr] = V[:,:,tCovConvFt[-1],tr].copy()
        # if V did not converge, tCovConvFt[-1]=T, and this copies itself!
        # mu_h[:,:,T-1,tr] = mu[:,:,T-1,tr] is already corrct

        t = T-2 # T-1 already correct by initialisation of mu_h, V_h
        if obsTime[-1]-obsTime[-2] == 1: # we already processed a whole subpop
            rangeobstime = range(1,len(obsTime)-1)[::-1] # and need to skip it
        else:     # i.e. obsTime[-1]-obsTime[-2] > 1:
            rangeobstime = range(1,len(obsTime))[::-1]

        for i in rangeobstime:
            # J,P depend only on filter output and hence converge along with it

            Vconv = V[:,:,tCovConvFt[i],tr].copy()
            Jconv = np.dot(np.dot(Vconv, Atr),Pinv[:,:,tCovConvFt[i],tr])
            Jconvtr = Jconv.transpose()
            Pconv = P[:,:,tCovConvFt[i],tr]

            ifCovConv = False # reset convergence flag
            while t > tCovConvFt[i]: 
                # in this interval, P and V have converged, hence we can 
                # expect V_h to converge early on, as well
                mu_h[ :,t,tr] += np.dot(Jconv, mu_h[:,t+1,tr] - AmuBu[:,t]) 

                if not ifCovConv:
                    V_h[:,:,t,tr] =  (Vconv 
                                      + np.dot(np.dot(Jconv, 
                                                   V_h[:,:,t+1,tr] - Pconv),
                                                   Jconvtr)
                                      ) 
                    if np.mean(np.power(V_h[:,:,t,tr]-V_h[:,:,t+1,tr],2))<eps:
                        tCovConvSm[i] = t     # overwriting tCovConvFt[i] + 1
                        ifCovConv     = True                                            

                t -= 1
            # at t = tCovConvFt[i], we update V_h to ensure we get the transition
            mu_h[ :,t,tr] += np.dot(Jconv, mu_h[:,t+1,tr] - AmuBu[:,t]) 

            # now V_h[:,:,t,tr] = Vconv, and if ifCoConv==False, it is still 
            # tCovConvSm[i] = tCovConvFt[i]+1 = t+1. Otherwise we the covariance
            # did converge and we also should look at tCovConvSm[i] now.
            V_h[:,:,t,tr] =  (Vconv        
                              + np.dot(np.dot(Jconv, 
                                       V_h[:,:,tCovConvSm[i],tr] - Pconv),
                                       Jconvtr)
                                      ) 
            J[:,:,t,tr] = Jconv # here, store J for all later time points 
            t -= 1
            # now t < tCovConv[i], i.e. we have all J,V,P again. 
            while t >= obsTime[i-1]:
                # in this interval, P and V still constantly change, so
                # we compute J for each time point invidivually
                J[:,:,t,tr] = np.dot(np.dot(V[:,:,t,tr], Atr),
                                            Pinv[:,:,t,tr])
                mu_h[ :,t,tr] += np.dot(J[:,:,t,tr], mu_h[:,t+1,tr]-AmuBu[:,t]) 

                V_h[:,:,t,tr] += np.dot(np.dot(J[:,:,t,tr], 
                                               V_h[:,:,t+1,tr] - P[:,:,t,tr]),
                                               J[:,:,t,tr].transpose()) 
                t -= 1

        # case for first subpopulation
        ifCovConv = False

        Vconv =  V[:,:,tCovConvFt[0],tr].copy()
        Jconv = np.dot(np.dot(Vconv, Atr), Pinv[:,:,tCovConvFt[0],tr])
        Jconvtr = Jconv.transpose()
        Pconv   = P[:,:,tCovConvFt[0],tr] 

        while t > tCovConvFt[0]: # no need for new J, as all was converged
            mu_h[ :,t,tr] += np.dot(Jconv, mu_h[:,t+1,tr] - AmuBu[:,t]) 

            if not ifCovConv:
                V_h[:,:,t,tr] = (Vconv 
                                 + np.dot(np.dot(Jconv, 
                                                 V_h[:,:,t+1,tr] - Pconv),
                                                 Jconvtr) 
                                 )
                if np.mean(np.power(V_h[:,:,t,tr]-V_h[:,:,t+1,tr],2))<eps:
                    tCovConvSm[0] = int(t)
                    ifCovConv     = True                    

            t -= 1
        # now V_h[:,:,t,tr] = Vconv, and if ifCoConv==False, it is still 
        # tCovConvSm[i] = tCovConvFt[i]+1 = t+1. Otherwise we the covariance
        # did converge and we also should look at tCovConvSm[i] now.
        mu_h[ :,t,tr] += np.dot(Jconv, mu_h[:,t+1,tr] - AmuBu[:,t]) 
        V_h[:,:,t,tr] =  (Vconv        
                          + np.dot(np.dot(Jconv, 
                                   V_h[:,:,tCovConvSm[0],tr] - Pconv),
                                   Jconvtr)
                                  ) 
        J[:,:,t,tr] = Jconv # here, store J for all later time points 
        t -= 1
        # now t < tCovConv[i], i.e. we have all J,V,P again. 
        while t >= 0:
            J[:,:,t,tr] = np.dot(np.dot(V[:,:,t,tr], Atr),
                                        Pinv[:,:,t,tr])
            mu_h[ :,t,tr] += np.dot(J[:,:,t,tr], mu_h[:,t+1,tr] - AmuBu[:,t]) 
            V_h[:,:,t,tr] += np.dot(np.dot(J[:,:,t,tr], 
                                           V_h[:,:,t+1,tr] - P[:,:,t,tr]),
                                           J[:,:,t,tr].transpose()) 

            t -= 1     

    return [mu_h,V_h,J,tCovConvSm]

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _KalmanParsToMoments(mu_h, V_h, J,obsTime,tCovConvFt,tCovConvSm):
    """ OUT = _KalmanParsToMoments(mu)h*,V_h*,J*)
    see Bishop, 'Pattern Recognition and Machine Learning', ch. 13
    for formulas and input/output naming conventions  
    The variable obsScheme is a dictionary that contains information
    on observed subpopulations of y that are observed at each time point 
    """                

    # introduced some heavy indexing to reflect changes to Kalman filter and
    # smoother code that no longer compute poster covariances for every time
    # step invidivually, but check for convergence. Times of convergence are
    # given by tCovConvFt for the (forward) Kalman filter, and tCovConvSm for
    # the (backward) Kalman smoother.  

    T    = mu_h.shape[1]
    Trial= mu_h.shape[2]

    Ext   = mu_h.copy()             # E[x_t]                        
    Extxt = V_h.copy()              # E[x_t, x_t]
    Extxtm1 = np.zeros(V_h.shape)   # E[x_t x_{t-1}'] 

    for tr in range(Trial):
        t = 0
        for i in range(len(obsTime)):            
            while t <= tCovConvFt[i]: # before the filter covariances converged
                # in this interval, the filtered covariances kept changing, and
                # thus so did the smoothed covariances. We have to look up all.                
                Extxt[:,:,t,tr] += np.outer(mu_h[:,t,tr], mu_h[:,t,tr])
                t += 1 
            V_hconv = V_h[:,:, tCovConvSm[i], tr]
            while t <= tCovConvSm[i]: # after filter and smoother converged
                # a little confusing, the smoother runs backwards in time and
                # in this middle section hence has already converged. We can
                # precompute V_h * J as they are both constant here.                
                Extxt[:,:,t,tr] =  (V_hconv 
                                    + np.outer(mu_h[:,t,tr], mu_h[:,t,tr]))
                t += 1 
            while t < obsTime[i]:    # after filter, before smoother converged                
                Extxt[:,:,t,tr] += np.outer(mu_h[:,t,tr], mu_h[:,t,tr])
                t += 1 

    for tr in range(Trial):
        t = 1        
        for i in range(len(obsTime)):            
            while t <= tCovConvFt[i]: # before the filter covariances converged
                # in this interval, the filtered covariances kept changing, and
                # thus so did the smoothed covariances. We have to look up all.                
                Extxtm1[:,:,t,tr] =  (np.dot(V_h[:,:, t, tr], 
                                             J[:,:,t-1,tr].transpose()) 
                                    + np.outer(mu_h[:,t,tr], mu_h[:,t-1,tr]) )
                t += 1 

            Jconv   = J[:,:,tCovConvFt[i],tr] # J depends only on filter output 
            Jconvtr = Jconv.transpose()    # and hence converges along with it
            VhJconv = np.dot(V_h[:,:, tCovConvSm[i], tr], Jconvtr)
            while t < tCovConvSm[i]: # after filter and smoother converged
                # a little confusing, the smoother runs backwards in time and
                # in this middle section hence has already converged. We can
                # precompute V_h * J as they are both constant here.                
                Extxtm1[:,:,t,tr] =  (VhJconv 
                                    + np.outer(mu_h[:,t,tr], mu_h[:,t-1,tr]) )
                t += 1 
            while t < obsTime[i]:    # after filter, before smoother converged     
                Extxtm1[:,:,t,tr] =  (np.dot(V_h[:,:, t, tr], 
                                             Jconvtr) 
                                    + np.outer(mu_h[:,t,tr], mu_h[:,t-1,tr]) )
                t += 1 
            # t == obsTime[i] now
            J[:,:,t-1,tr] = Jconv.copy() # does nothing if not converged, otherwise 
                                         # gives starting info for next subpopulation


    return [Ext, Extxt, Extxtm1] 

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _LDS_M_step(Ext, Extxt, Extxtm1, y, u, obsScheme, 
                 sy=None, syy=None, suu=None, suuinv=None, Ti=None):   
    """ OUT = _LDS_M_step(Ext*,Extxt*,Extxtm1*,y*,obsScheme*,syy)
    see Bishop, 'Pattern Recognition and Machine Learning', ch. 13
    for formulas and input/output naming conventions   
    The variable obsScheme is a dictionary that contains information
    on observed subpopulations of y that are observed at each time point
    The optional variable syy is the mean outer product of observations
    y. If not provided, it will be computed on the fly.
    The optional variable Ti counts the number of times that variables
    y_i, i=1,..,yDim occured. If not provided, it will be computed on the fly.
    
    """                        
    xDim  = Ext.shape[0]
    T     = Ext.shape[1]
    Trial = Ext.shape[2]    
    yDim  = y.shape[0]

    if (isinstance(u,np.ndarray) and 
        u.shape[1]==y.shape[1] and u.shape[2]==y.shape[2]):
        uDim  = u.shape[0]
    else:
        uDim  = 0
    
    xRange = range(xDim)

    # unpack observation scheme
    try:
        subpops = obsScheme['subpops'] # indeces of subpopulation members
        obsTime = obsScheme['obsTime'] # times of switches btw subpopulations
        obsPops = obsScheme['obsPops'] # gives active subpops btw switches

        # an index group is defined as a collection of components of y that
        # are always observed together. If the observed subpopulations have
        # no overlap, index groups are identical to subpopulations. With 
        # overlap, the components of y within an overlap between two sub-
        # populations however overall are observed more often than those 
        # belonging only to a single subpopulation. This will make a 
        # difference in the computations below. Hence we have to break 
        # down the subpopulations into smaller parts - in the two-subpop.
        # situation e.g. into those y_i that occur only in the first 
        # subpopulation, those only in the second subpopulation and those
        # occuring in the overlap. In principle, there are 2^m -1 many
        # index groups if the number of subpopulations is m ! (The -1 stems 
        # from assuming that each observed variable y_i is in at least one
        # subpopulation). In practice, only those index groups that have
        # any observed variables y_i are listed and described in the 
        # variables obsIdxG and idxgrps.
        obsIdxG = obsScheme['obsIdxG'] # gives currently observed index groups  
        idxgrps = obsScheme['idxgrps'] # gives members of each index group
    except:
        print('obsScheme:')
        print(obsScheme)
        raise Exception(('provided obsScheme dictionary does not have '
                         'the required fields: subpops, obsTimes, '
                         'obsPops, obsIdxG and idxgrps.'))

    rangeidxgrps = range(len(idxgrps))
    rangeobstime = range(len(obsTime))
    # count occurence of each observed index group (for normalisations)
    if Ti is None:
        Ti = np.zeros(len(idxgrps));
        for tr in range(Trial):
            for i in rangeobstime:
                if i == 0:
                    Ti[obsIdxG[i]] += obsTime[0] # - 0, for t = 0                                                
                else:
                    Ti[obsIdxG[i]] += obsTime[i] - obsTime[i-1]

    # compute sum and (diagonal of) scatter matrix for observed states    
    if sy is None:
        sy    = np.zeros(yDim)
        for tr in range(Trial):
            ytr = y[:,:,tr]            
            for i in rangeobstime:
                idx = subpops[obsPops[i]]
                if i == 0:
                    ts  = range(0, obsTime[i])                                            
                else:
                    ts  = range(obsTime[i-1],obsTime[i])   
                if len(idx)>0:
                    sy[idx] += np.sum(ytr[np.ix_(idx,ts)],1)          
    if syy is None:
        syy   = np.zeros(yDim) # sum over outer product y_t y_t'
        for tr in range(Trial):
            ytr = y[:,:,tr]            
            for i in rangeobstime:
                idx = subpops[obsPops[i]]
                if i == 0:
                    ts  = range(0, obsTime[i])                                            
                else:
                    ts  = range(obsTime[i-1],obsTime[i])                 
                if len(idx)>0:
                    ytmp = ytr[np.ix_(idx,ts)]
                    syy[idx] += np.sum(ytmp*ytmp,1) 
        del ytmp      
    
    # compute (diagonal of) scatter matrix accros observed and latent states
    # compute scatter matrices from posterior means for the latent states
    sExt    = np.zeros(xDim)            # sums of expected values of   
    sExtxt1toN = np.zeros([xDim, xDim]) # x_t, x_t * x_t', 
    syExt      = np.zeros([yDim, xDim]) # y_t * x_t' (only for observed y_i)

    # versions of sums exclusively over data points where individual index
    # groups are observed:
    sExts   = np.zeros([xDim, len(idxgrps)])       
    sExtxts = np.zeros([xDim, xDim, len(idxgrps)])            

    for tr in range(Trial):              # collapse over trials ...
        ytr = y[:,:,tr]
        for i in rangeobstime:         # ... but keep  apart
            idx = subpops[obsPops[i]]  # list of currently observed y_i
            if i == 0:
                ts  = range(0, obsTime[i])                                            
            else:
                ts  = range(obsTime[i-1],obsTime[i])           

            tsExt   = np.sum(Ext[:,ts,tr],1)
            tsExtxt = np.sum(Extxt[:,:,ts,tr], 2)
            sExt         += tsExt           # these sum over 
            sExtxt1toN   += tsExtxt         # all times 
            for j in obsIdxG[i]: 
                sExts[:,j]      += tsExt    # these only sum entries 
                sExtxts[:,:,j]  += tsExtxt  # seen by their index group

            if len(idx)>0:     
                syExt[idx,:] += np.einsum('in,jn->ij',   # index groups are non-overlapping, i.e.
                                    ytr[np.ix_(idx,ts)], # can store outer products y(i)_t x_t'
                                    Ext[:,ts,tr])        # for all i in the same matrix. 

    del ytr

    sysExt = np.outer(sy, sExt)                                                             

    sExtxt2toN   = sExtxt1toN - np.sum(Extxt[:,:,0 , :],2)  
    sExtxt1toNm1 = sExtxt1toN - np.sum(Extxt[:,:,T-1,:],2)           

    sExtxtm1 = np.sum(Extxtm1[:,:,1:T,:], (2,3)) # sum over E[x_t x_{t-1}']        

    
    # Start computing (closed-form) updated parameters
    
    # initial distribution parameters
    mu0 = 1/Trial * np.sum( Ext[:,0,:], 1 )                                    # still blatantly
    V0  = 1/Trial * np.sum( Extxt[:,:,0,:], 2) - np.outer(mu0, mu0)            # wrong

    # latent dynamics paramters
    if uDim > 0:

        # compute scatter matrix for input states
        if suu is None:
            suu   = np.zeros([uDim,uDim]) # sum over outer product u_t u_t'
            for tr in range(Trial):
                utr = u[:,range(1,T),tr]            
                suu += np.dot(utr, utr.transpose())        
        if suuinv is None:
            suuinv = sp.linalg.inv(suu)   # also need matrix inverse  

        # compute scatter matrix accros input and latent states
        sExtu = np.zeros([xDim, uDim]) # sum over outer product u_t x_t'
        for tr in range(Trial):        # collapse over trials ...
            sExtu += np.einsum('in,jn->ij', 
                                Ext[:,range(1,T),tr], # omit the first
                                u[:,range(1,T),tr]) # time step!                              
        suExtm1 = np.zeros([uDim, xDim]) # sum over outer product u_t x_t-1'
        for tr in range(Trial):          # collapse over trials ...
            suExtm1 += np.einsum('in,jn->ij', 
                                  u[:,range(1,T),tr], 
                                  Ext[:,range(0,T-1),tr])                         

        suuvinvsuExtm1 = np.dot(suuinv, suExtm1)
        sExsuuusuExm1  = np.dot(sExtu,               suuvinvsuExtm1)
        sExm1suusuExm1 = np.dot(suExtm1.transpose(), suuvinvsuExtm1)
                      

        A = np.dot(sExtxtm1-sExsuuusuExm1, 
                   sp.linalg.inv(sExtxt1toNm1-sExm1suusuExm1))                                    
        Atr = A.transpose()

        B = np.dot(sExtu - np.dot(A, suExtm1.transpose()), suuinv)
        Btr = B.transpose()
        sExtxtm1Atr = np.dot(sExtxtm1, Atr)
        sExtuBtr   = np.dot(sExtu, Btr)
        BsuExtm1Atr = np.dot(B, np.dot(suExtm1, Atr))                
        Q = 1/(Trial*(T-1)) * (  sExtxt2toN   
                               - sExtxtm1Atr.transpose()
                               - sExtxtm1Atr 
                               + np.dot(np.dot(A, sExtxt1toNm1), Atr) 
                               - sExtuBtr.transpose()
                               - sExtuBtr
                               + BsuExtm1Atr.transpose()
                               + BsuExtm1Atr
                               + np.dot(np.dot(B, suu), Btr)
                              )

    else: # reduce to non-input LDS equations

        A = np.dot(sExtxtm1, 
                   sp.linalg.inv(sExtxt1toNm1))                                    
        Atr = A.transpose()
        sExtxtm1Atr = np.dot(sExtxtm1, Atr)
        B = np.zeros([xDim,uDim])                
        Q = 1/(Trial*(T-1)) * (  sExtxt2toN   
                               - sExtxtm1Atr.transpose()
                               - sExtxtm1Atr 
                               + np.dot(np.dot(A, sExtxt1toNm1), Atr) 
                              )
       
    # observed state parameters C, d    
    # The maximum likelihood solution for C, d in the context of missing data
    # are different from the standard solutions and given by 
    # C[i,:] = (sum_{t:ti} (y(i)_t-d(i)) x_t')  (sum_{t:ti} x_t x_'t)^-1
    # where {ti} is the set of time points t where variable y(i) is observed
    # d[i]   = 1/|ti| sum_{t:ti} y(i)_t - C[i,:] * x_t

    C   = np.zeros([yDim,xDim])    
    for i in rangeidxgrps:
        ixg  = idxgrps[i]
        C[ixg,:] = np.dot(syExt[ixg,:]
                          - np.outer(sy[ixg],sExts[:,i])/Ti[i], 
                          sp.linalg.inv(
                                    sExtxts[:,:,i]
                                  - np.outer(sExts[:,i],sExts[:,i])/Ti[i]
                                        )
                          )        

    d = np.zeros(yDim)
    for i in rangeidxgrps:    
        ixg  = idxgrps[i]
        d[ixg] = (sy[ixg] - np.dot(C[ixg,:], sExts[:,i]))/ Ti[i]

    # now use C, d to compute key terms of the residual noise
    CsExtxtCtr = np.zeros(yDim)
    sdExt      = np.zeros([yDim,xDim])
    for i in rangeidxgrps:    
        ixg  = idxgrps[i]
        sdExt[ixg,:] += np.outer(d[ixg],sExts[:,i])
        Cj  = C[ixg,:]
        CsExtxtCtr[ixg] += np.einsum('ij,ik,jk->i', Cj, Cj,sExtxts[:,:,i])

    # compute observation noise parameter R
    R = ( syy - 2 * sy * d
         + CsExtxtCtr
         + 2 * np.sum(C * (sdExt-syExt),1)
         ) 
    for i in rangeidxgrps:
        R[idxgrps[i]] /= Ti[i]   # normalise sums by actual number of data
    R += d*d  # normalisation of dd' cancels out with summation, so do neither 

    return [A,B,Q,mu0,V0,C,d,R,sy,syy,suu,suuinv,Ti]           

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _unpackInitPars(initPars, uDim, yDim=None, xDim=None, 
                    ifRDiagonal=True):   
    """ OUT = _unpackInitPars(initPars*,uDim,yDim,xDim,ifRDiagonal)
        initPars:    list of parameter arrays. Set None to get default setting 
        uDim:        dimensionality of input variables u
        yDim:        dimensionality of observed variables y
        xDim:        dimensionality of latent variables x
        ifRDiagonal: boolean, specifying if observed covariance R is diagonal
    Extracts and formats state-space model parameter for the EM algorithm.

    """
    if not isinstance(uDim, numbers.Integral) or uDim < 0:
        print('uDim:')
        print(uDim)
        raise Exception('argument uDim has to be a non-negative integer')    

    
    # extended LDS, with input matrix B and constant observation offset d
    if initPars is None:
        initPars = [None,None,None,None,None,None,None,None]
    elif (not ((isinstance(initPars, list)       and len(initPars)==8) or
               (isinstance(initPars, np.ndarray) and initPars.size==8))):
        print(initPars)
        raise Exception(('argument initPars for fitting a LDS with input '
                         'to data has to be a list or an ndarray with '
                         ' exactly 8 elemnts: {A,BQ,mu0,V0,C,dR}. '
                         'Alternatively, it is possible to hand over '
                         'initPars = None to obtain a default input-LDS '
                         'EM-algorithm initialization.'))            
    
    # if not provided, figure out latent dimensionality from parameters
    if xDim is None:
        if not initPars[0] is None and isinstance(initPars[0], np.ndarray):
            xDim = initPars[0].shape[0] # we can get xDim from A
        if not initPars[1] is None and isinstance(initPars[1], np.ndarray):
            xDim = initPars[1].shape[0] # we can get xDim from B
        elif  not initPars[2] is None and isinstance(initPars[2],np.ndarray):
            xDim = initPars[2].shape[0] # ... or from Q
        elif  not initPars[3] is None and isinstance(initPars[3],np.ndarray):
            xDim = initPars[3].size     # ... or from mu0
        elif  not initPars[4] is None and isinstance(initPars[4],np.ndarray):
            xDim = initPars[4].shape[0] # ... or from V0
        elif  not initPars[5] is None and isinstance(initPars[5],np.ndarray):
            xDim = initPars[5].shape[1] # ... or from C
        else: 
            raise Exception(('could not obtain xDim. Need to provide '
                             'either xDim, or initializations for at '
                             'least one of the following: '
                             'A, B, Q, mu0, V0 or C. None was provided.'))
    elif not (isinstance(xDim, numbers.Integral) and xDim > 0):
        print('xDim:')
        print(xDim)
        raise Exception('argument xDim has to be a positive integer')
    if yDim is None:
        if  not initPars[5] is None and isinstance(initPars[5],np.ndarray):
            yDim = initPars[5].shape[0] # ... or from C
        elif  not initPars[6] is None and isinstance(initPars[6],np.ndarray):
            yDim = initPars[6].shape[0] # ... or from d
        elif  not initPars[7] is None and isinstance(initPars[7],np.ndarray):
            yDim = initPars[7].shape[0] # ... or from R
        else: 
            raise Exception(('could not obtain yDim. Need to provide '
                             'either yDim, or initializations for at '
                             'least one of the following: '
                             'C, d, R. None was provided.'))
    elif not (isinstance(yDim, numbers.Integral) and yDim > 0):
        print('yDim:')
        print(yDim)
        raise Exception('argument yDim has to be a positive integer')            

    if initPars[0] is None:
        A   = 0.9 * np.identity(xDim)            
    elif np.all(initPars[0].shape==(xDim,xDim)): 
        A   = initPars[0].copy()
    else:
        print('xDim:')
        print(xDim)
        print('A.shape:')
        print(initPars[0].shape)
        raise Exception(('Bad initialization for LDS parameter A.'
                         'Shape not matching dimensionality of x'))    
    if uDim > 0:        
        if initPars[1] is None:
            B   =       np.identity(xDim)            
        elif np.all(initPars[1].shape==(xDim,uDim)): 
            B   = initPars[1].copy()
        else:
            print('xDim:')
            print(xDim)
            print('uDim:')
            print(uDim)
            print('B.shape:')
            print(initPars[1].shape)
            raise Exception(('Bad initialization for LDS parameter B.'
                             'Shape not matching dimensionality of x,u'))
    else: # if we're not going to use B anyway, ...
        B = np.array([0]) 

    if initPars[2] is None:
        Q   =       np.identity(xDim)            
    elif np.all(initPars[2].shape==(xDim,xDim)): 
        Q   = initPars[2].copy()
    else:
        print('xDim:')
        print(xDim)
        print('Q.shape:')
        print(initPars[2].shape)
        raise Exception(('Bad initialization for LDS parameter Q.'
                         'Shape not matching dimensionality of x'))
    if initPars[3] is None:
        mu0 =       np.zeros(xDim)            
    elif initPars[3].size==xDim: 
        mu0 = initPars[3].copy()
    else:
        print('xDim:')
        print(xDim)
        print('mu0.shape:')
        print(initPars[3].shape)
        raise Exception(('Bad initialization for LDS parameter mu0.'
                         'Shape not matching dimensionality of x'))
    if initPars[4] is None:
        V0  =       np.identity(xDim)            
    elif np.all(initPars[4].shape==(xDim,xDim)): 
        V0  = initPars[4].copy()
    else:
        print('xDim:')
        print(xDim)
        print('V0.shape:')
        print(initPars[3].shape)
        raise Exception(('Bad initialization for LDS parameter V0.'
                         'Shape not matching dimensionality of x'))
    if initPars[5] is None:
        C = np.random.normal(size=[yDim, xDim]) 
    elif np.all(initPars[5].shape==(yDim,xDim)):
        C = initPars[5].copy()
    else:
        print('xDim:')
        print(xDim)
        print('yDim:')
        print(yDim)
        print('C.shape:')
        print(initPars[5].shape)     
        raise Exception(('Bad initialization for LDS parameter C.'
                         'Shape not matching dimensionality of y, x'))  
    if initPars[6] is None:
        d = np.random.normal(size=[yDim]) 
    elif np.all(initPars[6].shape==(yDim,)):
        d = initPars[6].copy()
    else:
        print('yDim:')
        print(yDim)
        print('d.shape:')
        print(initPars[6].shape)  
        raise Exception(('Bad initialization for LDS parameter C.'
                         'Shape not matching dimensionality of y, x'))              
    if ifRDiagonal:
        if initPars[7] is None:
            R = np.ones(yDim)
        elif np.all(initPars[7].shape==(yDim,)):
            R = initPars[7].copy()
        elif np.all(initPars[7].shape==(yDim,yDim)):
            R = initPars[7].copy().diagonal()
        else:
            print('yDim:')
            print(yDim)
            print('R.shape:')
            print(initPars[7].shape) 
            raise Exception(('Bad initialization for LDS '
                             'parameter C. Shape not matching '
                             'dimensionality of y'))        
            
    else:
        if initPars[7] is None:
            R = np.identity(yDim)                
        elif np.all(initPars[7].shape==(yDim,yDim)):
            R = initPars[7].diagonal().copy()
        else:
            print('yDim:')
            print(yDim)
            print('R.shape:')
            print(initPars[7].shape) 
            raise Exception(('Bad initialization for LDS '
                             'parameter C. Shape not matching '
                             'dimensionality of y'))             
    return [A,B,Q,mu0,V0,C,d,R,xDim]
        
#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _getResultsFirstEMCycle(initPars, obsScheme, y, u, eps=1e-30,
                            ifFitA=True):
    """ OUT = _getNumericsFirstEMCycle(initPars*,obsScheme*,y*,u*,eps)
        initPars: collection of initial parameters for LDS
        obsScheme: observation scheme for given data, stored in dictionary
                   with keys 'subpops', 'obsTimes', 'obsPops'
        y:         data array of observed variables
        u:         data array of input variables
        eps:       precision (stopping criterion) for deciding on convergence
                   of latent covariance estimates durgin the E-step                   
        This function serves to quickly get the results of one EM-cycle. It is
        mostly intended to generate results that can quickly be compared against
        other EM implementations or different parameter initialisation methods. 
    """
    [A_0,B_0,Q_0,mu0_0,V0_0,C_0,d_0,R_0] = initPars
    # do one E-step
    [Ext_0, Extxt_0, Extxtm1_0,LL_0, tCovConvFt, tCovConvSm]    = \
      _LDS_E_step(A_0,B_0,Q_0,mu0_0,V0_0,C_0,d_0,R_0,y,u,obsScheme,eps)
    # do one M-step      
    [A_1,B_1,Q_1,mu0_1,V0_1,C_1,d_1,R_1,my,syy,suu,suuinv,Ti] = \
      _LDS_M_step(Ext_0,Extxt_0,Extxtm1_0,y,u,obsScheme) 

    if not ifFitA:
        A_1 = A_0.copy() # simply discard the update!

    # do another E-step
    [Ext_1, Extxt1_1, Extxtm1_1,LL_1, tCovConvFt, tCovConvSm] = \
      _LDS_E_step(A_1,B_1,Q_1,mu0_1,V0_1,C_1,d_1,R_1,y,u,obsScheme, eps)

    return [Ext_0, Extxt_0, Extxtm1_0,LL_0, 
            A_1,B_1,Q_1,mu0_1,V0_1,C_1,d_1,R_1,my,syy,suu,suuinv,Ti,
            Ext_1, Extxt1_1, Extxtm1_1,LL_1]

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _computeObsIndexGroups(obsScheme,yDim):
    """ OUT = _computeObsIndexGroups(obsScheme,yDim)
        obsScheme: observation scheme for given data, stored in dictionary
                   with keys 'subpops', 'obsTimes', 'obsPops'
        yDim:        dimensionality of observed variables y
    Computes index groups for given observation scheme. 

    """
    try:
        subpops = obsScheme['subpops'];
        obsTime = obsScheme['obsTime'];
        obsPops = obsScheme['obsPops'];
    except:
        print('obsScheme:')
        print(obsScheme)
        raise Exception(('provided obsScheme dictionary does not have '
                         'the required fields: subpops, obsTimes, '
                         'and obsPops.'))        

    J = np.zeros([yDim, len(subpops)]) # binary matrix, each row gives which 
    for i in range(len(subpops)):      # subpopulations the observed variable
        J[subpops[i],i] = 1            # y_i is part of

    twoexp = np.power(2,np.arange(len(subpops))) # we encode the binary rows 
    hsh = np.sum(J*twoexp,1)                     # of J using binary numbers

    lbls = np.unique(hsh)         # each row of J gets a unique label 
                                     
    idxgrps = [] # list of arrays that define the index groups
    for i in range(lbls.size):
        idxgrps.append(np.where(hsh==lbls[i])[0])

    obsIdxG = [] # list f arrays giving the index groups observed at each
                 # given time interval
    for i in range(len(obsPops)):
        obsIdxG.append([])
        for j in np.unique(hsh[np.where(J[:,obsPops[i]]==1)]):
            obsIdxG[i].append(np.where(lbls==j)[0][0])            
    # note that we only store *where* the entry was found, i.e. its 
    # position in labels, not the actual label itself - hence we re-defined
    # the labels to range from 0 to len(idxgrps)

    return [obsIdxG, idxgrps]
                      
#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _evaluateFit(A, B, Q, C, d, R, y, u, Ext, Extxt, Extxtm1, LLs):
    """ TO BE EXTENDED """

    T    = y.shape[1]
    Trial= y.shape[2]
    xDim = A.shape[0]
    yDim = C.shape[0]
    uDim = u.shape[0]

    Pi_h    = np.array([sp.linalg.solve_discrete_lyapunov(A, Q)])[0,:,:]
    Pi_t_h  = np.dot(A.transpose(), Pi_h)


    dataCov  = np.cov(y[:,0:T-1,0], y[:,1:T,0])
    covyy    = dataCov[np.ix_(np.arange(0, yDim), np.arange(0,     yDim))]
    covyy_m1 = dataCov[np.ix_(np.arange(0, yDim), np.arange(yDim,2*yDim))]

    plt.figure(1)
    cmap = matplotlib.cm.get_cmap('brg')
    clrs = [cmap(i) for i in np.linspace(0, 1, xDim)]
    for i in range(xDim):
        plt.subplot(xDim,1,i)
        plt.plot(x[i,:,0], color=clrs[i])
        plt.hold(True)
        if np.mean( np.square(x[i,:,0] - Ext_h[i,:,0]) ) < np.mean( np.square(x[i,:,0] + Ext_h[i,:,0]) ):
            plt.plot( Ext_h[i,:,0], color=clrs[i], ls=':')
        else:
            plt.plot(-Ext_h[i,:,0], color=clrs[i], ls=':')

    m = np.min([Pi_h.min(), covyy.min()])
    M = np.max([Pi_h.max(), covyy.max()])       
    plt.figure(1)
    plt.subplot(1,3,1)
    plt.imshow(np.dot(np.dot(C_h, Pi_h), C_h.transpose()) + R_h, interpolation='none')
    plt.title('cov_hat(y_t,y_t)')
    plt.clim(m,M)
    plt.subplot(1,3,2)
    plt.imshow(covyy,    interpolation='none')
    plt.title('cov_emp(y_t,y_t)')
    plt.clim(m,M)
    plt.subplot(1,3,3)
    plt.imshow(np.dot(np.dot(C, Pi), C.transpose()) + R, interpolation='none')
    plt.title('cov_true(y_t,y_t)')
    plt.clim(m,M)
    plt.figure(3)

    m = np.min([covyy_m1.min(), Pi_t_h.min()])
    M = np.max([covyy_m1.max(), Pi_t_h.max()])
    plt.subplot(1,3,1)
    plt.imshow(np.dot(np.dot(C_h, Pi_t_h), C_h.transpose()), interpolation='none')
    plt.title('cov_hat(y_t,y_{t-1})')
    plt.clim(m,M)
    plt.subplot(1,3,2)
    plt.imshow(covyy_m1,    interpolation='none')
    plt.title('cov(y_t,y_{t-1})')
    plt.clim(m,M)
    plt.subplot(1,3,3)
    plt.imshow(np.dot(np.dot(C, Pi_t), C.transpose()), interpolation='none')
    plt.title('cov_true(y_t,y_{t-1})')
    plt.clim(m,M)
    plt.figure(4)
    plt.plot(np.sort(np.linalg.eig(A)[0]), 'r')
    plt.hold(True)
    plt.plot(np.sort(np.linalg.eig(A_h)[0]), 'b')
    plt.legend(['true', 'est'])


import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import time

import ssm_timeSeries as ts  # library for time series overhead
import ssm_fit               # library for state-space model fitting
import numbers

import random
from datetime import datetime     # generate random seed for 
random.seed(datetime.now())       # np.random. Once this is fixed, all 
rngSeed = random.randint(0, 1000) # other 'randomness' is also fixed

from scipy.io import savemat # store results for comparison with Matlab code   

def run(xDim, 
        yDim, 
        uDim, 
        T, 
        obsScheme, 

        maxIter=10, 
        epsilon=np.log(1.001),
        covConvEps=0,        
        ifPlotProgress=False,
        ifTraceParamHist=False,
        ifRDiagonal=True,
        ifUseA=True,
        ifUseB=False,

        truePars=None,                          
        trueAgen='diagonal', 
        truelts=None,
        trueBgen='random', 
        trueQgen='identity', 
        truemu0gen='random', 
        trueV0gen='identity', 
        trueCgen='random', 
        truedgen='scaled', 
        trueRgen='fraction',

        initPars=None,
        initAgen='diagonal', 
        initlts=None,
        initBgen='random',  
        initQgen='identity', 
        initmu0gen='random', 
        initV0gen='identity', 
        initCgen='random', 
        initdgen='mean', 
        initRgen='fractionObserved',

        u=None, inputType='pwconst',constInputLngth=1,
        y=None, 
        x=None,
        ifStoreIntermediateResults = False,
        saveFile='LDS_data.mat'):

    if not isinstance(ifUseB,bool):
        print('ifUseB:')
        print( ifUseB  )
        raise Exception(('ifUseB has to be a boolean'))

    if not isinstance(ifUseA,bool):
        print('ifUseA:')
        print( ifUseA  )
        raise Exception(('ifUseA has to be a boolean'))     

    if not isinstance(ifRDiagonal,bool):
        print('ifRDiagonal:')
        print( ifRDiagonal  )
        raise Exception(('ifRDiagonal has to be a boolean'))

    if not isinstance(ifStoreIntermediateResults,bool):
        print('ifStoreIntermediateResults:')
        print( ifStoreIntermediateResults  )
        raise Exception(('ifStoreIntermediateResults has to be a boolean'))

    obsScheme = ssm_fit._checkObsScheme(obsScheme,yDim,T) 
       
    if y is None:
        if truelts is None:
            truelts = np.linspace(0.9,0.98,xDim)
        truePars = generateModelPars(xDim, yDim, uDim, 
                          parsIn=truePars,
                          obsScheme=obsScheme,
                          Agen=trueAgen, 
                          lts=truelts,
                          Bgen=trueBgen, 
                          Qgen=trueQgen, 
                          mu0gen=truemu0gen, 
                          V0gen=trueV0gen, 
                          Cgen=trueCgen,
                          dgen=truedgen, 
                          Rgen=trueRgen)
        Trial = 1 # fix to always just one repetition for now        

        # generate data from model
        print('generating data from model with ground-truth parameters')
        [x,y,u] = simulateExperiment(truePars,T,Trial,obsScheme,
                                     u,inputType,constInputLngth)

        [A,B,Q,mu0, V0, C,d,R] = truePars
        Pi      = np.array([sp.linalg.solve_discrete_lyapunov(A,Q)])[0,:,:]
        Pi_t    = np.dot(A.transpose(),   Pi  )

        [Ext_true, Extxt_true, Extxtm1_true, LLtr, tCovConvFt, tCovConvSm] = \
                        computeEstep(truePars, y, u, obsScheme, covConvEps)

    else:  # i.e. if data provided
        truePars = [0,0,0,0,0,0,0,0]
        [A,B,Q,mu0, V0, C,d,R] = truePars
        Pi   = 0
        Pi_t = 0
        Ext_true = 0
        Extxt_true = 0
        Extxtm1_true = 0
        LLtr = 0


    # get initial parameters
    if not ifUseA: # overwrites any other parameter choices for A! Set A = 0 
        if isinstance(initPars, dict) and ('A' in initPars):
            initPars['A'] = np.zeros([xDim,xDim])
        elif (isinstance(initPars,(list,np.ndarray)) and 
              not initPars[0] is None):
            iniPars[0] = np.zeros([xDim,xDim])
        elif not initAgen == 'zero': 
            print(('Warning: set flag ifUseA=False, but did not set initAgen '
                   'to zero. Will overwrite initAgen to zero now.'))
            initAgen = 'zero'
    if not ifUseB: # overwrites any other parameter choices for B! Set B = 0
        if isinstance(initPars, dict) and ('B' in initPars):
            initPars['B'] = np.zeros([xDim,uDim])
        elif (isinstance(initPars,(list,np.ndarray)) and 
              not initPars[1] is None):
            iniPars[1] = np.zeros([xDim,uDim])
        elif not initBgen == 'zero': 
            print(('Warning: set flag ifBseA=False, but did not set initBgen '
                   'to zero. Will overwrite initBgen to zero now.'))
            initBgen = 'zero'

    if initlts is None:
        initlts = np.random.uniform(size=[xDim])

    initPars = generateModelPars(xDim, yDim, uDim, 
                      parsIn=initPars, 
                      obsScheme=obsScheme,
                      Agen=initAgen, 
                      lts=initlts,
                      Bgen=initBgen, 
                      Qgen=initQgen, 
                      mu0gen=initmu0gen, 
                      V0gen=initV0gen, 
                      Cgen=initCgen,
                      dgen=initdgen, 
                      Rgen=initRgen,
                      x=x, y=y, u=u)

    # check initial goodness of fit for initial parameters
    [Ext_0,Extxt_0,Extxtm1_0,LL_0, 
     A_1,B_1,Q_1,mu0_1,V0_1,C_1,d_1,R_1,
     my,syy,suu,suuinv,Ti,
     Ext_1,Extxt_1,Extxtm1_1,LL_1] = getResultsFirstEMCycle(
                                                    initPars, 
                                                    obsScheme, 
                                                    y, 
                                                    u, 
                                                    covConvEps,
                                                    ifUseA,
                                                    ifUseB)
    if ifStoreIntermediateResults:
        intermediateSaveFile = saveFile
    else: 
        intermediateSaveFile = None

    # fit the model to data          
    print('fitting model to data')
    t = time.time()
    [[A_h],[B_h],[Q_h],[mu0_h],[V0_h],[C_h],[d_h],[R_h],LL] = ssm_fit._fitLDS(
                y, 
                u,
                obsScheme,
                initPars, 
                maxIter,
                epsilon,
                covConvEps,
                ifPlotProgress, 
                ifTraceParamHist, 
                ifRDiagonal, 
                ifUseA, 
                ifUseB,
                xDim,
                intermediateSaveFile)

    elapsedTime = time.time() - t
    print('elapsed time for fitting is')
    print(elapsedTime)

    if ifUseB:
        [Ext_h, Extxt_h, Extxtm1_h, LL_h, tCovConvFt, tCovConvSm] = \
         ssm_fit._LDS_E_step(A_h,B_h,Q_h,mu0_h,V0_h,C_h,d_h,R_h, 
                             y,u,obsScheme, covConvEps)
    else:
        [Ext_h, Extxt_h, Extxtm1_h, LL_h, tCovConvFt, tCovConvSm] = \
         ssm_fit._LDS_E_step(A_h,B_h,Q_h,mu0_h,V0_h,C_h,d_h,R_h, 
                             y,None,obsScheme, covConvEps)
            
    learnedPars = [A_h.copy(),B_h.copy(),Q_h.copy(),mu0_h.copy(),
                   V0_h.copy(),C_h.copy(),d_h.copy(),R_h.copy()]
    [A_0,B_0,Q_0,mu0_0, V0_0, C_0,d_0,R_0] = initPars


    Pi_h    = np.array([sp.linalg.solve_discrete_lyapunov(A_h, Q_h)])[0,:,:]
    Pi_t_h  = np.dot(A_h.transpose(), Pi_h)

    # save results for visualisation (with Matlab code)
    if u is None:
        u = 0
        B_h = 0
        B_hs = [0]
    if B_h is None:
        B_h = 0
        B_hs = [0]
    matlabSaveFile = {'x': x, 'y': y, 'u' : u, 'LL' : LL, 
                      'T' : T, 'Trial':Trial, 'elapsedTime' : elapsedTime,
                      'inputType' : inputType,
                      'constantInputLength' : constInputLngth,
                      'ifUseB':ifUseB, 'ifUseA':ifUseA, 
                      'epsilon':epsilon,
                      'ifPlotProgress':ifPlotProgress,
                      'ifTraceParamHist':ifTraceParamHist,
                      'ifRDiagonal':ifRDiagonal,'ifUseA':ifUseA,
                      'ifUseB':ifUseB,
                      'covConvEps':covConvEps,        
                      'A':A, 'B':B, 'Q':Q, 'mu0':mu0,'V0':V0,'C':C,'d':d,'R':R,
                      'A_0':A_0, 'B_0':B_0, 'Q_0':Q_0, 'mu0_0':mu0_0,
                      'V0_0':V0_0,'C_0':C_0,'d_0':d_0, 'R_0':R_0,
                      'A_1':A_1, 'B_1':B_1, 'Q_1':Q_1, 'mu0_1':mu0_1,
                      'V0_1':V0_1,'C_1':C_1,'d_1':d_1, 'R_1':R_1,
                      'A_h':A_h, 'B_h':B_h, 'Q_h':Q_h, 'mu0_h':mu0_h,
                      'V0_h':V0_h,'C_h':C_h,'d_h':d_h, 'R_h':R_h,
                      #'A_hs':A_hs, 'B_hs':B_hs, 'Q_hs':Q_hs, 'mu0_hs':mu0_hs,
                      #'V0_hs':V0_hs,'C_hs':C_hs,'d_hs':d_hs, 'R_hs':R_hs,
                      'Ext':Ext_0, 'Extxt':Extxt_0, 'Extxtm1':Extxtm1_0,
                      'Ext1':Ext_1, 'Extxt1':Extxt_1, 'Extxtm11':Extxtm1_1,
                      'Ext_h':Ext_h, 'Extxt_h':Extxt_h, 'Extxtm1_h':Extxtm1_h,
                      'Ext_true':Ext_true, 'Extxt_true':Extxt_true, 
                      'Extxtm1_true':Extxtm1_true,
                      'Pi':Pi,'Pi_h':Pi_h,'Pi_t':Pi_t,'Pi_t_h': Pi_t_h,
                      'obsScheme' : obsScheme}

    savemat(saveFile,matlabSaveFile) # does the actual saving

    return [y,x,u,learnedPars,initPars,truePars]

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def generateModelPars(xDim, yDim, uDim=0, 
                      parsIn=None, obsScheme=None,
                      Agen='diagonal', lts=None,
                      Bgen='random', 
                      Qgen='identity', 
                      mu0gen='random', 
                      V0gen='identity', 
                      Cgen='random', 
                      dgen='scaled', 
                      Rgen='fraction',
                      x=None, y=None, u=None): 

    """ OUT = generateModelPars(xDim*,yDim*,uDim,parsIn, 
                                Agen,lts,Bgen,Qgen,mu0gen,V0gen, 
                                Cgen, dgen, Rgen,x,y,u,ifVisualiseInitPars)
        xDim : dimensionality of latent states x
        yDim : dimensionality of observed states y
        uDim : dimensionality of input states u
        parsIn:    None, or list/np.ndarray/dict containing no, some or all
                   of the desired parameters. This function will identify which
                   parameters were not handed over and will fill in the rest
                   according to selected paramters below.
        obsScheme: observation scheme for given data, stored in dictionary
                   with keys 'subpops', 'obsTimes', 'obsPops'
        Agen   : string specifying methods of parameter generation
        lts    : ndarray with one entry per latent time scale (i.e. xDim many)
        Bgen   :  ""
        Qgen   :  ""
        mu0gen :  "" 
        V0gen  :  ""
        Cgen   :  "" 
        dgen   :  ""
        Rgen   : (see below for details)
        x: data array of latent variables
        y: data array of observed variables
        u: data array of input variables
        ifVisualisePars : boolean, specifying whether to plot the outputs
                          of this function for inspection
        Initialises parameters of an LDS, potentially by looking at the data.
    """

    totNumParams = 8 # an input-LDS model has 8 parameters: A,B,Q,mu0,V0,C,d,R

    """ pars optional inputs to this function """
    if (not (isinstance(lts, np.ndarray) and 
             (np.all(lts.shape==(xDim,)) or np.all(lts.shape==(xDim,1)))
            ) ):
        print('lts (latent time scales)')
        print(lts)
        raise Exception('variable lts has to be an ndarray of shape (xDim,)')


    if y is None:
        if not x is None:
            raise Exception(('provided latent state sequence x but not '
                             'observed data y.')) 
        if not u is None:
            raise Exception(('provided input sequence u but not '
                             'observed data y.')) 

    else: # i.e. if y is provided:
        if not (isinstance(y,np.ndarray) 
                and len(y.shape)==3 and y.shape[0]==yDim):
            print('y:')
            print(y)
            raise Exception(('When providing optional input y, it has to be '
                             'an np.ndarray of dimensions (yDim,T,Trial)'))
        else:
            T     = y.shape[1] # take these values from y and compare with 
            Trial = y.shape[2] # x, u, as we need consistency

    if not (x is None or (isinstance(x,np.ndarray) 
                          and len(x.shape)==3 
                          and x.shape[0]==xDim
                          and x.shape[1]==T
                          and x.shape[2]==Trial) ):
        print('x:')
        print(x)
        raise Exception(('When providing optional input x, it has to be an '
                         'np.ndarray of dimensions (xDim,T,Trial)'))

    if not (u is None or (isinstance(u,np.ndarray) 
                          and len(u.shape)==3 
                          and u.shape[0]==uDim
                          and u.shape[1]==T
                          and u.shape[2]==Trial) ):
        print('x:')
        print(x)
        raise Exception(('When providing optional input x, it has to be an '
                         'np.ndarray of dimensions (xDim,T,Trial)'))


    if (Cgen == 'PCA') or (Rgen == 'fractionObserved'):  
        covy = np.cov(y[:,:,0]-np.mean(y, (1,2)).reshape(yDim,1)) 
        # Depending on the observation scheme, not all entries of the data
        # covariance are also interpretable, and the entries of covy for 
        # variable pairs (y_i,y_j) that were not observed together may indeed
        # contain NaN's of Inf's depending on the choice of representation of
        # missing data entries. Keep this in mind when selecting parameter 
        # initialisation methods such as Cgen=='PCA', which will work with 
        # the full matrix covy.
        # Note that the diagonal of covy should also be safe to use. 

    genPars = np.ones(totNumParams, dtype=bool) # flag: which pars to generate

    parsOut = []                   # initialise output parameters
    for i in range(totNumParams):  # (as list to be filled with
        parsOut.append(None)       #  np.ndarrays)

    """ parse (optional) user-provided true model parameters: """
    # allow comfortable use of dictionaries:
    if isinstance(parsIn, dict):

        if 'A' in parsIn:
            parsOut[0] = parsIn['A']
            genPars[0] = False
        if 'B' in parsIn:
            parsOut[1] = parsIn['B']
            genPars[1] = False
        if 'Q' in parsIn:
            parsOut[2] = parsIn['Q']
            genPars[2] = False
        if 'mu0' in parsIn:
            parsOut[3] = parsIn['mu0']
            genPars[3] = False
        if 'V0' in parsIn:
            parsOut[4] = parsIn['V0']
            genPars[4] = False
        if 'C' in parsIn:
            parsOut[5] = parsIn['C']
            genPars[5] = False
        if 'd' in parsIn:
            parsOut[6] = parsIn['d']
            genPars[6] = False
        if 'R' in parsIn:
            parsOut[7] = parsIn['R']
            genPars[7] = False
    # also support handing over lists or np.ndarrays:
    elif ((isinstance(parsIn,list) and len(parsIn)==8) or 
          (isinstance(parsIn,np.ndarray) and parsIn.size==8)):
        for i in range(8):
            if isinstance(parsIn[i],np.ndarray): # just checking for type, 
                parsOut[i] = parsIn[i]           # not fordimensionality! 
                genPars[i] = False                  

    """ fill in missing parameters (could be none, some, or all) """
    # generate latent state tranition matrix A
    if genPars[0]:
        if lts is None:
            lts = np.random.uniform(size=[xDim])
        if Agen == 'diagonal':
            parsOut[0] = np.diag(lts) # lts = latent time scales
        elif Agen == 'full':
            parsOut[0] = np.diag(lts) # lts = latent time scales
            while True:
                W    = np.random.normal(size=[xDim,xDim])
                if np.abs(np.linalg.det(W)) > 0.001:
                    break
            parsOut[0] = np.dot(np.dot(W, parsOut[0]), np.linalg.inv(W))
        elif Agen == 'random':
            parsOut[0] = np.random.normal(size=[xDim,xDim])            
        elif Agen == 'zero':  # e.g. when fitting without dynamics
            parsOut[0] = np.zeros([xDim,xDim])            
        else:
            print('Agen:')
            print(Agen)
            raise Exception('selected type for generating A not supported')
    # There is inherent degeneracy in any LDS regarding the basis in the latent
    # space. Any rotation of A can be corrected for by rightmultiplying C with
    # the inverse rotation matrix. We do not wish to limit A to any certain
    # basis in latent space, but in a first approach may still initialise A as
    # diagonal matrix .     

    # generate latent state input matrix B
    if genPars[1]:
        if Bgen == 'random':
            parsOut[1] = np.random.normal(size=[xDim,uDim])            
        elif Bgen == 'zero': # make sure is default if ifUseB=False
            parsOut[1] = np.zeros([xDim,uDim])            
        else:
            print('Bgen:')
            print(Bgen)
            raise Exception('selected type for generating B not supported')
    # Parameter B is never touched within the code unless ifUseB == True,
    # hence we don't need to ensure its correct dimensionality if ifUseB==False

    # generate latent state innovation noise matrix Q
    if genPars[2]:                             # only one implemented standard 
        if Qgen == 'identity':     # case: we can *always* rotate x
            parsOut[2]    = np.identity(xDim)  # so that Q is the identity 
        else:
            print('Qgen:')
            print(Qgen)
            raise Exception('selected type for generating Q not supported')
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
    if genPars[3]:
        if mu0gen == 'random':
            parsOut[3]  = np.random.normal(size=[xDim])
        elif mu0gen == 'zero': 
            parsOut[3]  = np.zeros(xDim)
        else:
            print('mu0gen:')
            print(mu0gen)
            raise Exception('selected type for generating mu0 not supported')
    # generate initial latent state covariance matrix V0
    if genPars[4]:
        if V0gen == 'identity': 
            parsOut[4]    = np.identity(xDim)  
        else:
            print('V0gen:')
            print(V0gen)
            raise Exception('selected type for generating V0 not supported')
    # Assuming long time series lengths, parameters for the very first time
    # step are usually of minor importance for the overall fitting result
    # unless they are overly restrictive. We by default initialise V0 
    # non-commitingly to the identity matrix (same as Q) and mu0 either
    # to all zero or with a slight random perturbation on that.   

    # generate emission matrix C
    if genPars[5]:
        if Cgen == 'random': 
            parsOut[5] = np.random.normal(size=[yDim, xDim])
        elif Cgen == 'PCA':
            if y is None:
                raise Exception(('tried to set emission matrix C from results '
                                 'of a PCA on the observed data without '
                                 'providing any data y'))            
            w, v = np.linalg.eig(covy-np.diag(R_0))                           
            w = np.sort(w)[::-1]                 
            # note that we also enforce equal variance for each latent dim. :
            parsOut[5] = np.dot(v[:, range(xDim)], 
                                np.diag(np.sqrt(w[range(xDim)])))  
        else:
            print('Cgen:')
            print(Cgen)
            raise Exception('selected type for generating C not supported')
    # C in many cases is the single-most important parameter to properly 
    # initialise. If the data is fully observed, a basic and powerful solution
    # is to use PCA on the full data covariance (after attributing a certain 
    # fraction of variance to R). In stitching contexts, this however is not
    # possible. Finding a good initialisation in the context of incomplete data
    # observation is not trivial. 

    # check for resulting stationary covariance of latent states x
    Pi    = np.array([sp.linalg.solve_discrete_lyapunov(parsOut[0], 
                                                        parsOut[2])])[0,:,:]
    Pi_t  = np.dot(parsOut[0].transpose(), Pi)  # time-lagged cov(y_t, y_{t-1})
    CPiC = np.dot(parsOut[5], np.dot(Pi, parsOut[5].transpose())) 

    # generate emission noise covariance matrix R
    if genPars[7]:
        if Rgen == 'fraction':
            # set R_ii as 25% to 125% of total variance of y_i
            parsOut[7] = (0.25+np.random.uniform(size=[yDim]))*CPiC.diagonal() 
        elif Rgen == 'fractionObserved':
            if y is None:
                raise Exception(('tried to set emission noise covariance R as '
                                 'a fraction of data variance without '
                                 'providing any data y'))
            parsOut[7]   = 0.1 * covy.diagonal()
        elif Rgen == 'identity':
            Rgen = np.ones(yDim)
        elif Rgen == 'zero':                        # very extreme case!
            Rgen = np.zeros(yDim)
        else:
            print('Rgen:')
            print(Rgen)
            raise Exception('selected type for generating R not supported')
    # C and R should not be initialised independently! Following on the idea
    # of (diagonal) R being additive private noise for the individual variables
    # y_i, we can initialise R as being a certain fraction of the observed 
    # noise. When initialising R from data, we have to be carefull not to
    # attribute too much noise to R, as otherwise the remaining covariance 
    # matrix cov(y)-np.diag(R) might no longer be positive definite!

    # generate emission noise covariance matrix d
    if genPars[6]:
        if dgen == 'scaled':
            parsOut[6] = (np.sqrt(np.mean(np.diag(CPiC+np.diag(parsOut[7])))) 
                                             * np.random.normal(size=yDim))
        elif dgen == 'random':
            parsOut[6] = np.random.normal(size=yDim)
        elif dgen == 'zero':
            parsOut[6] = np.zeros(yDim)
        elif dgen == 'mean':
            if y is None:
                raise Exception(('tried to set observation offset d as the '
                                 'data mean without providing any data y'))
            parsOut[6] = np.mean(y,(1,2)) 
        else:
            print('dgen:')
            print(dgen)
            raise Exception('selected type for generating d not supported')
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
    initOptions = {
                 'Agen'   : Agen,
                 'lts'    : lts,
                 'Bgen'   : Bgen,
                 'Qgen'   : Qgen,
                 'mu0gen' : mu0gen,
                 'V0gen'  : V0gen,
                 'Cgen'   : Cgen,
                 'dgen'   : dgen,
                 'Rgen'   : Rgen
                    }

    """ check validity (esp. of user-provided parameters), return results """
    checkPars(parsOut, xDim, yDim, uDim)

    return parsOut

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def checkPars(pars, xDim, yDim, uDim):


    # check latent state tranition matrix A
    if not np.all(pars[0].shape == (xDim,xDim)):
        print(     'A.shape')
        print(pars[0].shape )
        raise Exception(('Variable Q has to be a '
                         ' (xDim,xDim)-array'))

    # check latent state input matrix B
    if not np.all(pars[1].shape == (xDim,uDim)):
        print(     'B.shape')
        print(pars[1].shape )
        raise Exception(('Variable B has to be a '
                         ' (xDim,uDim)-array'))

    # check latent state innovation noise matrix Q
    if not np.all(pars[2].shape == (xDim,xDim)):
        print(     'Q.shape')
        print(pars[2].shape )
        raise Exception(('Variable Q has to be a '
                         ' (xDim,xDim)-array'))

    # check initial latent state mean mu0
    if np.all(pars[3].shape == (xDim,1)) or np.all(pars[3].shape == (1,xDim)):
        pars[3] = pars[3].reshape(xDim)     
    if not np.all(pars[3].shape == (xDim,)):
        print(   'mu0.shape')
        print(pars[3].shape )
        raise Exception(('Variable mu0 has to be a '
                         ' (xDim,)-array')) 

    # check initial latent state covariance matrix V0
    if not np.all(pars[4].shape == (xDim,xDim)):
        print(    'V0.shape')
        print(pars[4].shape )
        raise Exception(('Variable V0 has to be a '
                         ' (xDim,xDim)-array'))

    # check emission matrix C
    if not np.all(pars[5].shape == (yDim,xDim)):
        print(     'C.shape')
        print(pars[5].shape )
        raise Exception(('Variable C has to be a '
                         ' (yDim,xDim)-array')) 

    # check emission noise covariance matrix d
    if np.all(pars[6].shape == (yDim,1)) or np.all(pars[6].shape == (1,yDim)):
        pars[6] = pars[6].reshape(yDim)     
    if not np.all(pars[6].shape == (yDim,)):
        print(     'd.shape')
        print(pars[6].shape )
        raise Exception(('Variable d has to be a '
                         ' (yDim,)-array')) 

    # check emission noise covariance matrix R
    if (not (np.all(pars[7].shape == (yDim,yDim)) or
             np.all(pars[7].shape == (yDim, )   )) ):
        print(     'R.shape')
        print(pars[7].shape )
        raise Exception(('Variable R is assumed to be diagonal. '
                         'Please provide the diagonal entries as'
                         ' (yDim,)-array')) 

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def simulateExperiment(pars,T,Trial=1,obsScheme=None,
                       u=None,inputType='pwconst',constInputLngth=1):

    xDim = pars[0].shape[0] # get xDim from A
    yDim = pars[5].shape[0] # get yDim from C

    if np.all(pars[7].shape==(yDim,)): # ssm_timeSeries assumes R to be a 
        pars = pars.copy()             # full yDim-by-yDim covariance matrix.
        pars[7] = np.diag(pars[7])     # Need to reshape

    if u is None:
        uDim = 0
        ifGenInput = False
    elif isinstance(u, np.ndarray) and u.shape[1]==T and u.shape[2]==Trial:
        uDim = u.shape[0]
        ifGenInput = False
    elif isinstance(u, numbers.Integral):
        uDim = u
        ifGenInput = True        

    if isinstance(pars[1],np.ndarray) and pars[1].size>0 and uDim==0:
        print(('Warning: Parameter B is initialised, but uDim = 0. '
               'Algorithm will ignore input for the LDS, outcomes may be '
               'not as expected.'))

    if ifGenInput:
        if inputType=='pwconst': # generate piecewise constant input
            u = np.zeros([uDim,T,Trial])
            for tr in range(Trial):
                for i in range(int(np.floor(T/constInputLngth))):
                    idxRange = range((i-1)*constInputLngth, i*constInputLngth)
                    u[:,idxRange,tr] = np.random.normal(size=[1])
                u[:,:,tr] -= np.mean(u[:,:,tr])
        elif inputType=='random': # generate random Gaussian input
            u = np.random.normal(size=[uDim,T,Trial])
            for tr in range(Trial):
                u[:,:,tr] -= np.mean(u[:,:,tr])
        else:
            raise Exception(('selected option for input generation '
                             'not supported. It is possible to directly '
                             'hand over pre-computed inputs u.'))

    if uDim > 0:
      seq = ts.setStateSpaceModel('iLDS',[xDim,yDim,uDim],pars) # initiate 
      seq.giveEmpirical().addData(Trial,T,[u],rngSeed)          # draw data
    else: 
      parsNoInput = pars.copy()
      parsNoInput[1] = np.zeros([xDim,1])
      seq = ts.setStateSpaceModel('iLDS',[xDim,yDim,1],parsNoInput) 
      seq.giveEmpirical().addData(Trial,T,None,rngSeed)          # draw data


    x = seq.giveEmpirical().giveData().giveTracesX()
    y = seq._empirical._data.giveTracesY()        

    return [x,y,u]

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def computeEstep(pars, y, u=None, obsScheme=None, eps=1e-30):

    [A,B,Q,mu0,V0,C,d,R] = pars

    if (u is None and 
        not (B is None or B == [] or
             (isinstance(B,np.ndarray) and B.size==0) or
             (isinstance(B,np.ndarray) and B.size>0 and np.max(abs(B))==0) or
             (isinstance(B,(float,numbers.Integral)) and B==0))):
        print(('Warning: Parameter B is initialised, but input u = None. '
               'Algorithm will ignore input for the LDS, outcomes may be '
               'not as expected.'))

    if obsScheme is None:
        print('creating default observation scheme: Full population observed')
        obsScheme = {'subpops': [list(range(yDim))], # creates default case
                     'obsTime': [T],                 # of fully observed
                     'obsPops': [0]}                 # population

    [Ext_true, Extxt_true, Extxtm1_true, LLtr, tCovConvFt, tCovConvSm] = \
        ssm_fit._LDS_E_step(A,B,Q,mu0,V0,C,d,R,y,u,obsScheme, eps)

    return [Ext_true, Extxt_true, Extxtm1_true, LLtr, tCovConvFt, tCovConvSm]

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def evaluateFit(A, B, Q, C, d, R, y, u, Ext, Extxt, Extxtm1, LLs):
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
        if (np.mean( np.square(x[i,:,0] - Ext_h[i,:,0]) ) < 
            np.mean( np.square(x[i,:,0] + Ext_h[i,:,0]) )  ):
            plt.plot( Ext_h[i,:,0], color=clrs[i], ls=':')
        else:
            plt.plot(-Ext_h[i,:,0], color=clrs[i], ls=':')

    m = np.min([Pi_h.min(), covyy.min()])
    M = np.max([Pi_h.max(), covyy.max()])       
    plt.figure(1)
    plt.subplot(1,3,1)
    plt.imshow(np.dot(np.dot(C_h, Pi_h), C_h.transpose()) + R_h, 
               interpolation='none')
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
    plt.imshow(np.dot(np.dot(C_h, Pi_t_h), C_h.transpose()), 
               interpolation='none')
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

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def getResultsFirstEMCycle(initPars, obsScheme, y, u, eps=1e-30, 
                           ifUseA=True,ifUseB=True):
    """ OUT = _getNumericsFirstEMCycle(initPars*,obsScheme*,y*,u*,eps)
        initPars: collection of initial parameters for LDS
        obsScheme: observation scheme for given data, stored in dictionary
                   with keys 'subpops', 'obsTimes', 'obsPops'
        y:         data array of observed variables
        u:         data array of input variables
        eps:       precision (stopping criterion) for deciding on convergence
                   of latent covariance estimates durgin the E-step                   
        This function serves to quickly get the results of one EM-cycle. It is
        mostly intended to generate results that can quickly be compared with
        other EM implementations or different parameter initialisation methods. 
    """
    [A_0,B_0,Q_0,mu0_0,V0_0,C_0,d_0,R_0] = initPars
    # do one E-step
    [Ext_0, Extxt_0, Extxtm1_0,LL_0, tCovConvFt, tCovConvSm]    = \
      ssm_fit._LDS_E_step(A_0,B_0,Q_0,mu0_0,V0_0,C_0,d_0,R_0,y,u,obsScheme,eps)
    # do one M-step      
    [A_1,B_1,Q_1,mu0_1,V0_1,C_1,d_1,R_1,my,syy,suu,suuinv,Ti] = \
      ssm_fit._LDS_M_step(Ext_0,Extxt_0,Extxtm1_0,y,u,obsScheme,
                          ifUseA=ifUseA,ifUseB=ifUseB) 

    if not ifUseA:
        A_1 = A_0.copy() # simply discard the update!

    # do another E-step
    [Ext_1, Extxt1_1, Extxtm1_1,LL_1, tCovConvFt, tCovConvSm] = \
      ssm_fit._LDS_E_step(A_1,B_1,Q_1,mu0_1,V0_1,C_1,d_1,R_1,y,u,obsScheme,eps)

    return [Ext_0, Extxt_0, Extxtm1_0,LL_0, 
            A_1,B_1,Q_1,mu0_1,V0_1,C_1,d_1,R_1,my,syy,suu,suuinv,Ti,
            Ext_1, Extxt1_1, Extxtm1_1,LL_1]

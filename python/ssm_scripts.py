
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

def run(xDim, yDim, uDim, T, obsScheme, fitOptions=None,
        u=None, inputType='pwconst',constInputLngth=1,
        truePars=None,
	initPars=None,
        y=None, x=None,
        saveFile='LDS_data.mat'):

    if fitOptions is None:                         
        fitOptions = {}

    try: 
        fitOptions['ifUseB']
    except:
          fitOptions['ifUseB'] = False
          print(('Warning: Did not specify whether to use input matrix B.'
                 'Default is NOT to use it.'))
    if not isinstance(fitOptions['ifUseB'],bool):
        print('ifUseB:')
        print(fitOptions['ifUseB'])
        raise Exception(('ifUseB has to be a boolean'))

    try:
        fitOptions['maxIter']
    except:
        fitOptions['maxIter'] = 50
    if (not isinstance(fitOptions['maxIter'],numbers.Integral)
        or not fitOptions['maxIter'] > 0):
        print('maxIter:')
        print(fitOptions['maxIter'])
        raise Exception(('maxIter has to be a positive integer'))

    try:
        fitOptions['epsilon']
    except:
        fitOptions['epsilon'] = np.log(1.001)
    if (not isinstance(fitOptions['epsilon'],(float,numbers.Integral))
        or not fitOptions['epsilon'] >= 0):
        print('epsilon:')
        print(fitOptions['epsilon'])
        raise Exception(('epsilon has to be a non-negative number'))

    try: 
        fitOptions['ifPlotProgress']
    except:
        fitOptions['ifPlotProgress'] = False
    if not isinstance(fitOptions['ifPlotProgress'],bool):
        print('ifPlotProgress:')
        print(fitOptions['ifPlotProgress'])
        raise Exception(('ifPlotProgress has to be a boolean'))

    try:
        fitOptions['ifTraceParamHist']
    except:
        fitOptions['ifTraceParamHist'] = False
    if not isinstance(fitOptions['ifTraceParamHist'],bool):
        print('ifTraceParamHist:')
        print(fitOptions['ifTraceParamHist'])
        raise Exception(('ifTraceParamHist has to be a boolean'))

    try:
        fitOptions['ifRDiagonal']
    except:
        fitOptions['ifRDiagonal'] = True
    if not isinstance(fitOptions['ifRDiagonal'],bool):
        print('ifRDiagonal:')
        print(fitOptions['ifRDiagonal'] )
        raise Exception(('ifRDiagonal has to be a boolean'))

    try:
        fitOptions['ifFitA']
    except:
        fitOptions['ifFitA'] = True
    if not isinstance(fitOptions['ifFitA'],bool):
        print('ifFitA:')
        print(fitOptions['ifFitA'] )
        raise Exception(('ifFitA has to be a boolean'))

    try:
        fitOptions['ifInitCwithPCA']
    except:
        fitOptions['ifInitCwithPCA'] = False
    if not isinstance(fitOptions['ifInitCwithPCA'],bool):
        print('ifInitCwithPCA:')
        print(fitOptions['ifInitCwithPCA'] )
        raise Exception(('ifInitCwithPCA has to be a boolean'))        

    try:
        fitOptions['covConvEps']
    except:
        fitOptions['covConvEps'] = 1e-30
    if (not isinstance(fitOptions['covConvEps'],(float,numbers.Integral))
        or not fitOptions['covConvEps'] >= 0):
        print('covConvEps:')
        print(fitOptions['covConvEps'])
        raise Exception(('covConvEps has to be a non-negative number'))
    if isinstance(fitOptions['covConvEps'],numbers.Integral): # convert!
        fitOptions['covConvEps'] = float(fitOptions['covConvEps'])
    if fitOptions['covConvEps'] > 1:
        print(('Warning: Selected convergence criterion for latent '
               'covariance is very generous. Results of the E-step may '
               'be very imprecise. Default value is 1e-30.'))

    Trial = 1 # fix to always just one repetition for now

    truePars = generatePars(xDim, yDim, uDim, truePars)

    if y is None or x is None:
        [x,y,u] = simulateExperiment(truePars,T,Trial,obsScheme,
                                     u,inputType,constInputLngth)
        [A,B,Q,mu0, V0, C,d,R] = truePars
        Pi      = np.array([sp.linalg.solve_discrete_lyapunov(A,Q)])[0,:,:]
        Pi_t    = np.dot(A.transpose(),   Pi  )

        [Ext_true, Extxt_true, Extxtm1_true, LLtr, tCovConvFt, tCovConvSm] = \
                        computeEstep(truePars, y, u, obsScheme, 
                        fitOptions['covConvEps'])
    else: 

        truePars = [0,0,0,0,0,0,0,0]
        [A,B,Q,mu0, V0, C,d,R] = truePars
        Pi   = 0
        Pi_t = 0
        Ext_true = 0
        Extxt_true = 0
        Extxtm1_true = 0
        LLtr = 0

    try:
        obsScheme['obsIdxG']     # check for addivional  
        obsScheme['idxgrps']     # (derivable) information
    except:                       # can fill in if missing !
        [obsIdxG, idxgrps] = ssm_fit._computeObsIndexGroups(obsScheme,yDim)
        obsScheme['obsIdxG'] = obsIdxG # add index groups and 
        obsScheme['idxgrps'] = idxgrps # their occurences    



    t = time.time()

    # get initial parameters
    if initPars is None:
        if fitOptions['ifFitA']:
            initAType = 'random'
        elif not fitOptions['ifFitA']:
            initAType = 'zero'
        if fitOptions['ifInitCwithPCA']:
            print(('Warning: Using full data information for initialisation of C. '
                   'When doing spatial stitching, this is cheating.'))
            initCType = 'PCA'
        else: 
            initCType = 'random'
        [initPars, initOptions] = ssm_fit._getInitPars(y, u, xDim,
                                                       fitOptions['ifUseB'], 
                                                       obsScheme,
                                                       initA   =initAType,
                                                       initC   =initCType)

    else:
        print('Using provided initial parameters. User is responsible for making sure they work!')
    print('A_0:')
    print(initPars[0])

    # check initial goodness of fit for initial parameters
    [Ext_0,Extxt_0,Extxtm1_0,LL_0, 
     A_1,B_1,Q_1,mu0_1,V0_1,C_1,d_1,R_1,
     my,syy,suu,suuinv,Ti,
     Ext_1,Extxt_1,Extxtm1_1,LL_1] = ssm_fit._getResultsFirstEMCycle(
                                                    initPars, 
                                                    obsScheme, 
                                                    y, 
                                                    u, 
                                                    fitOptions['covConvEps'],
                                                    fitOptions['ifFitA'])
    # fit the model to data           
    [[A_h],[B_h],[Q_h],[mu0_h],[V0_h],[C_h],[d_h],[R_h],LL] = ssm_fit._fitLDS(
                y, 
                u,
                obsScheme,
                initPars, 
                fitOptions,
                xDim,
                saveFile)

    elapsedTime = time.time() - t
    print('elapsed time for fitting is')
    print(elapsedTime)

    if fitOptions['ifUseB']:
        [Ext_h, Extxt_h, Extxtm1_h, LL_h, tCovConvFt, tCovConvSm] = ssm_fit._LDS_E_step(A_h,B_h,Q_h,mu0_h,V0_h,C_h,d_h,R_h, 
                                                                         y,u,obsScheme, 
                                                                         fitOptions['covConvEps'])
    else:
        [Ext_h, Extxt_h, Extxtm1_h, LL_h, tCovConvFt, tCovConvSm] = ssm_fit._LDS_E_step(A_h,B_h,Q_h,mu0_h,V0_h,C_h,d_h,R_h, 
                                                                         y,None,obsScheme, 
                                                                         fitOptions['covConvEps'])
            
    learnedPars = [A_h.copy(),B_h.copy(),Q_h.copy(),mu0_h.copy(),V0_h.copy(),C_h.copy(),d_h.copy(),R_h.copy()]
    [A_0,B_0,Q_0,mu0_0, V0_0, C_0,d_0,R_0] = initPars


    Pi_h    = np.array([sp.linalg.solve_discrete_lyapunov(A_h, Q_h)])[0,:,:]
    Pi_t_h  = np.dot(A_h.transpose(), Pi_h)

    # save results for visualisation (with Matlab code, as my python-plotting still lacks badly)
    if u is None:
        u = 0
        B_h = 0
        B_hs = [0]
    if B_h is None:
      B_h = 0
      B_hs = [0]
    matlabSaveFile = {'x': x, 'y': y, 'u' : u, 'LL' : LL, 'T' : T, 'elapsedTime' : elapsedTime,
                      'inputType' : inputType,
                      'constantInputLength' : constInputLngth,
                      'ifUseB':fitOptions['ifUseB'], # let us safe that extra
                      'A':A, 'B':B, 'Q':Q, 'mu0':mu0,'V0':V0,'C':C,'d':d,'R':R,
                      'A_0':A_0, 'B_0':B_0, 'Q_0':Q_0, 'mu0_0':mu0_0,'V0_0':V0_0,'C_0':C_0,'d_0':d_0, 'R_0':R_0,
                      'A_1':A_1, 'B_1':B_1, 'Q_1':Q_1, 'mu0_1':mu0_1,'V0_1':V0_1,'C_1':C_1,'d_1':d_1, 'R_1':R_1,
                      'A_h':A_h, 'B_h':B_h, 'Q_h':Q_h, 'mu0_h':mu0_h,'V0_h':V0_h,'C_h':C_h,'d_h':d_h, 'R_h':R_h,
                      #'A_hs':A_hs, 'B_hs':B_hs, 'Q_hs':Q_hs, 'mu0_hs':mu0_hs,'V0_hs':V0_hs,'C_hs':C_hs,'d_hs':d_hs, 'R_hs':R_hs,
                      'fitOptions'  : fitOptions,
                      'initOptions' : initOptions, 
                      'Ext':Ext_0, 'Extxt':Extxt_0, 'Extxtm1':Extxtm1_0,
                      'Ext1':Ext_1, 'Extxt1':Extxt_1, 'Extxtm11':Extxtm1_1,
                      'Ext_h':Ext_h, 'Extxt_h':Extxt_h, 'Extxtm1_h':Extxtm1_h,
                      'Ext_true':Ext_true, 'Extxt_true':Extxt_true, 'Extxtm1_true':Extxtm1_true,
                      'Pi' : Pi, 'Pi_h' : Pi_h, 'Pi_t' : Pi_t, 'Pi_t_h': Pi_t_h,
                      'obsScheme' : obsScheme}

    savemat(saveFile,matlabSaveFile) # does the actual saving

    return [y,x,u,learnedPars,initPars,truePars]


def generatePars(xDim, yDim, uDim,truePars): 

    genPars = np.ones(8, dtype=bool)

    if isinstance(truePars, dict):

        if 'A' in truePars:
            A = truePars['A']
            genPars[0] = False
        if 'B' in truePars:
            B = truePars['B']
            genPars[1] = False
        if 'Q' in truePars:
            Q = truePars['Q']
            genPars[2] = False
        if 'mu0' in truePars:
            mu0 = truePars['mu0']
            genPars[3] = False
        if 'V0' in truePars:
            V0 = truePars['V0']
            genPars[4] = False
        if 'C' in truePars:
            C = truePars['C']
            genPars[5] = False
        if 'd' in truePars:
            d = truePars['d']
            genPars[6] = False
        if 'R' in truePars:
            R = truePars['R']
            genPars[7] = False
    elif ((isinstance(truePars,list) and len(truePars)==8) or 
          (isinstance(truePars,np.ndarray) and truePars.size==8)):
        for i in range(8):
            if isinstance(truePars[i],np.ndarray): # just checking for type,   
                genPars[i] = False                 # not fordimensionality! 
        [A,B,Q,mu0,V0,C,d,R] = truePars # potentially overwrite some of those

    if genPars[0]:
        while True:
            W    = np.random.normal(size=[xDim,xDim])
            if np.abs(np.linalg.det(W)) > 0.001:
                break
        A    = np.diag(np.linspace(0.9,0.98,xDim)) 
        #A    = np.dot(np.dot(W, A), np.linalg.inv(W))

    if genPars[1]:
        B = np.random.normal(size=[xDim,uDim])            

    if genPars[2]:
        Q    = np.identity(xDim)

    if genPars[3]:
        mu0  = np.random.normal(size=[xDim]) #np.random.normal(size=[xDim])

    if genPars[4]:
        V0   = np.identity(xDim)

    if genPars[5]:
        C = np.random.normal(size=[yDim, xDim])

    Pi    = np.array([sp.linalg.solve_discrete_lyapunov(A, Q)])[0,:,:]
    Pi_t  = np.dot(A.transpose(), Pi)
    CPiC = np.dot(C, np.dot(Pi, C.transpose()))

    print(CPiC.shape)
    if genPars[7]:
        # set R_ii as 25% to 125% of total variance of y_i
        R = (0.25 + np.random.uniform(size=[yDim])) * CPiC.diagonal() 
        R = np.diag(R)

    if np.all(R.shape==(yDim,)):
        R = np.diag(R)

    if genPars[6]:
        d = np.sqrt(np.mean(np.diag(CPiC+np.diag(R)))) * np.random.normal(size=yDim)

    return [A.copy(),B.copy(),Q.copy(),
            mu0.copy(),V0.copy(),
            C.copy(),d.copy(),R.copy()]


def simulateExperiment(pars,T,Trial=1,obsScheme=None,
                       u=None,inputType='pwconst',constInputLngth=1):

    xDim = pars[0].shape[0] # get xDim from A
    yDim = pars[5].shape[0] # get yDim from C

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
      seq = ts.setStateSpaceModel('iLDS',[xDim,yDim,uDim],pars) # initiate model
      seq.giveEmpirical().addData(Trial,T,[u],rngSeed)          # draw data
    else: 
      parsNoInput = pars.copy()
      parsNoInput[1] = np.zeros([xDim,1])
      seq = ts.setStateSpaceModel('iLDS',[xDim,yDim,1],parsNoInput) 
      seq.giveEmpirical().addData(Trial,T,None,rngSeed)          # draw data


    x = seq.giveEmpirical().giveData().giveTracesX()
    y = seq._empirical._data.giveTracesY()        

    return [x,y,u]


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

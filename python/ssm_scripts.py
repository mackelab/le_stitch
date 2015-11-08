%matplotlib inline
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import time

import ssm_timeSeries as ts  # library for time series overhead
import ssm_fit               # library for state-space model fitting

import random
from datetime import datetime     # generate random seed for 
random.seed(datetime.now())       # np.random. Once this is fixed, all 
rngSeed = random.randint(0, 1000) # other 'randomness' is also fixed

from scipy.io import savemat # store results for comparison with Matlab code


    subpops = [list(range(0,30)), list(range(20,50)), list(range(40,70)), list(range(60,90))]
    obsTime = [int(round(T/4)), int(round(T/2)), int(round(2*T/3)), int(T)]
    obsPops = [0,1,2,3]
    obsScheme = {'subpops': subpops,
                 'obsTime': obsTime,
                 'obsPops': obsPops}
    [obsIdxG, idxgrps] = ssm_fit._computeObsIndexGroups(obsScheme,yDim)
    obsScheme['obsIdxG'] = obsIdxG # add index groups and 
    obsScheme['idxgrps'] = idxgrps # their occurences        

def generatePars(xDim, yDim, uDim): 

    while True:
        W    = np.random.normal(size=[xDim,xDim])
        if np.abs(np.linalg.det(W)) > 0.001:
            break
    A    = np.diag(np.linspace(0.1,0.95,xDim))  #np.diag(np.random.uniform(size=[xDim]))
    A    = np.dot(np.dot(W, A), np.linalg.inv(W))
    Qnoise = np.random.normal(size=[xDim,xDim])/9
    Qnoise = np.dot(Qnoise, Qnoise.transpose())
    Q    = np.identity(xDim) + Qnoise
    mu0  = np.random.normal(size=[xDim]) #np.random.normal(size=[xDim])
    V0   = np.identity(xDim)
    C = np.random.normal(size=[yDim, xDim])

    Pi    = np.array([sp.linalg.solve_discrete_lyapunov(A, Q)])[0,:,:]
    Pi_t  = np.dot(A.transpose(), Pi)
    CPiC = np.dot(C, np.dot(Pi, C.transpose()))

    R = (0.1 + np.random.uniform(size=[yDim])) * CPiC.diagonal() # set R_ii as 5% to 55% of total variance of y_i
    R = np.diag(R)
    print('R is of shape')
    print(R.shape)
    d  = np.arange(yDim)
    d -= 10
    
    uDim = 1
    B = np.random.normal(size=[xDim,uDim])            
    return [A,B,Q,mu0,V0,C,d,R]

def simulateExperiment(pars,T,Trial=1,obsScheme=None,
                       u=None,inputType='pwconst',constInputLngth=1):
"""  Set latent dimensionality    """

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

    if isinstance(B,np.ndarray) and B.size>0 and uDim==0:
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



    seq = ts.setStateSpaceModel('iLDS',[xDim,yDim,uDim],pars) # initiate model 
    seq.giveEmpirical().addData(Trial,T,[u],rngSeed)          # draw data

    x = seq.giveEmpirical().giveData().giveTracesX()
    y = seq._empirical._data.giveTracesY()          

    return [y,x,u]

def computeEstep(pars, y, u=None, obsScheme=None, eps=1e-30):

    [A,B,Q,mu0,V0,C,d,R] = pars

    if (u is None and not (B is None or B == [] or
                           isinstance(B,np.ndarray) and np.max(abs(B))==0 or
                           isinstance(B,(float,numbers.Integral)) and B==0)):
        print(('Warning: Parameter B is initialised, but input u = None. '
               'Algorithm will ignore input for the LDS, outcomes may be '
               'not as expected.'))

    if obsScheme is None:
        print('creating default observation scheme: Full population observed')
        obsScheme = {'subpops': [list(range(yDim))], # creates default case
                     'obsTime': [T],                 # of fully observed
                     'obsPops': [0]}                 # population

    [Ext_true, Extxt_true, Extxtm1_true, LLtr] = \
        ssm_fit._LDS_E_step(A,B,Q,mu0,V0,C,d,R,y,u,obsScheme, eps)

    return [Ext_true, Extxt_true, Extxtm1_true, LLtr]


def load    
""" ALTERNATIVELY, LOAD SOME ACTUAL DATA """
"""

data = np.loadtxt('/home/mackelab/Desktop/Projects/Stitching/data/ChaLearn_Challenge_Connectomics/' 
                 +'valid/fluorescence_valid.txt',
                   delimiter=',')
#                + 'fluorescence_iNet1_Size100_CC05inh.txt', 

y = np.zeros([data.shape[1], data.shape[0], 1])
y[:,:,0] = data.transpose()
yDim  = y.shape[0]
T     = y.shape[1]
Trial = y.shape[2]
uDim  = 0
u     = []
x     = []
ifDataGeneratedWithInput = False
ifInputPiecewiseConstant = False
constantInputLength = 1

[A,B,Q,mu0,V0,C,d,R,Ext_true,Extxt_true,Extxtm1_true,LLtr]= [0,0,0,0,0,0,0,0,0,0,0,0]
Pi           = np.zeros([xDim, xDim])
Pi_t         = np.zeros([xDim, xDim])

ifUseB = False
subpops = [list(range(0,yDim)), list(range(0,yDim))]
obsTime = [int(round(T/4)), int(round(2*T/3)), T]
obsPops = [0,1,0]
obsScheme = {'subpops': subpops,
             'obsTime': obsTime,
             'obsPops': obsPops}
[obsIdxG, idxgrps] = ssm_fit._computeObsIndexGroups(obsScheme,yDim)
obsScheme['obsIdxG'] = obsIdxG # add index groups and 
obsScheme['idxgrps'] = idxgrps # their occurences        


"""

[initPars, initOptions] = ssm_fit._getInitPars(y, u, xDim, ifUseB, obsScheme, initC = 'PCA')

tmp =  ssm_fit._getResultsFirstEMCycle(initPars, obsScheme, y, u, eps=1e-30)
[Ext_0,Extxt_0,Extxtm1_0,LL_0,A_1,B_1,Q_1,mu0_1,V0_1,C_1,d_1,R_1,my,syy,suu,suuinv,Ti,Ext_1,Extxt1_1,Extxtm1_1,LL_1] = tmp

# fit the model to data                               

[A_0,B_0,Q_0,mu0_0,V0_0,C_0,d_0,R_0] = initPars
fitoptions = {'maxIter' : 50,
              'epsilon' : np.log(1.01),
              'ifPlotProgress' : True,
              'ifTraceParamHist' : False,
              'ifRDiagonal' : True,
              'ifUseB' : ifUseB,
              'covConvEps': 1e-30}

t = time.time()
%matplotlib inline

[[A_h],[B_h],[Q_h],[mu0_h],[V0_h],[C_h],[d_h],[R_h],LL] = ssm_fit._fitLDS(
            y, 
            u,
            obsScheme,
            initPars, 
            fitoptions,
            xDim)

elapsedTime = time.time() - t
print('elapsed time for fitting is')
print(elapsedTime)

if ifUseB:
    [Ext_h, Extxt_h, Extxtm1_h, LL_h]          = ssm_fit._LDS_E_step(A_h,B_h,Q_h,mu0_h,V0_h,C_h,d_h,R_h, 
                                                                     y,u,obsScheme, 1e-30)
else:
    [Ext_h, Extxt_h, Extxtm1_h, LL_h]          = ssm_fit._LDS_E_step(A_h,B_h,Q_h,mu0_h,V0_h,C_h,d_h,R_h, 
                                                                     y,None,obsScheme, 1e-30)
    

Pi_h    = np.array([sp.linalg.solve_discrete_lyapunov(A_h, Q_h)])[0,:,:]
Pi_t_h  = np.dot(A_h.transpose(), Pi_h)

# save results for numerical comparison with Matlab-generated results
matlabSaveFile = {'x': x, 'y': y, 'u' : u, 
                  'A':A, 'B':B, 'Q':Q, 'mu0':mu0,'V0':V0,'C':C,'d':d,'R':R,
                  'A_0':A_0, 'B_0':B_0, 'Q_0':Q_0, 'mu0_0':mu0_0,'V0_0':V0_0,'C_0':C_0,'d_0':d_0,'R_0':R_0,
                  'A_1':A_1, 'B_1':B_1, 'Q_1':Q_1, 'mu0_1':mu0_1,'V0_1':V0_1,'C_1':C_1,'d_1':d_1,'R_1':R_1,
                  'A_h':A_h, 'Q_h':Q_h, 'mu0_h':mu0_h,'V0_h':V0_h,'C_h':C_h,'d_h': d_h, 'R_h':R_h,
                  'Ext':Ext_0, 'Extxt':Extxt_0, 'Extxtm1':Extxtm1_0,
                  'T':T,
                  'LL0': LL_0, 'LL1': LL_1}
savemat('LDS_data.mat',matlabSaveFile)

# save results for visualisation (with Matlab code, as my python-plotting still lacks badly)
if u is None:
    u = 0
matlabSaveFile = {'x': x, 'y': y, 'u' : u, 'LL' : LL,
                  'ifDataGeneratedWithInput' : ifDataGeneratedWithInput, 
                  'ifInputPiecewiseConstant' : ifInputPiecewiseConstant,
                  'constantInputLength' : constantInputLength,
                  'ifUseB':ifUseB,
                  'A':A, 'B':B, 'Q':Q, 'mu0':mu0,'V0':V0,'C':C,'d':d,'R':R,
                  'A_0':A_0, 'B_0':B_0, 'Q_0':Q_0, 'mu0_0':mu0_0,'V0_0':V0_0,'C_0':C_0,'d_0':d_0, 'R_0':R_0,
                  'A_h':A_h, 'B_h':B_h, 'Q_h':Q_h, 'mu0_h':mu0_h,'V0_h':V0_h,'C_h':C_h,'d_h':d_h, 'R_h':R_h,
                  'Ext_h':Ext_h, 'Extxt_h':Extxt_h, 'Extxtm1_h':Extxtm1_h,
                  'Ext_true':Ext_true, 'Extxt_true':Extxt_true, 'Extxtm1':Extxtm1_true,
                  'Pi' : Pi, 'Pi_h' : Pi_h, 'Pi_t' : Pi_t, 'Pi_t_h': Pi_t_h,
                  'obsScheme' : obsScheme}
savemat('LDS_data_to_visualise.mat',matlabSaveFile)

import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import savemat # store results for comparison with Matlab code   

import ssm_timeSeries as ts  # my self-written time series overhead
import ssm_fit               # my self-written library for state-space model fitting
import ssm_scripts

import random
from datetime import datetime     # generate random seed for 
random.seed(datetime.now())       # np.random. Once this is fixed, all 
rngSeed = random.randint(0, 1000) # other 'randomness' is also fixed

%matplotlib inline

yDim = 6
xDim = 1
uDim = 0

T = 1000
Trial = 1

subpops = [list(range(0,yDim)), list(range(0,yDim))] # generate some non-stitched 
obsTime = [int(round(T/2)), int(round(T))]           # scenario which has transitions between
obsPops = [0,1]                                      # (full) subpopulations
obsScheme = {'subpops': subpops,
             'obsTime': obsTime,
             'obsPops': obsPops}
try:
    obsScheme['obsIdxG']     # check for addivional  
    obsScheme['idxgrps']     # (derivable) information
except:                       # can fill in if missing !
    [obsIdxG, idxgrps] = ssm_fit._computeObsIndexGroups(obsScheme,yDim)
    obsScheme['obsIdxG'] = obsIdxG # add index groups and 
    obsScheme['idxgrps'] = idxgrps # their occurences   

fitOptions = {'ifUseB' : False,  
              'maxIter': 50, 
              'ifPlotProgress' : True,
              'covConvEps' :0 
             }
pars = ssm_scripts.generatePars(xDim, yDim, uDim)
[x,y,u] = ssm_scripts.simulateExperiment(pars,T,Trial,obsScheme)
[A,B,Q,mu0, V0, C,d,R] = pars
Bu = np.zeros([A.shape[0], y.shape[1], y.shape[2]])

[mu,V,P,Pinv,logc,tCovConvFt] = ssm_fit._KalmanFilter(A,Bu,Q,mu0,V0,C,d,R,y,obsScheme,eps=0)
[mu_h,V_h,J,tCovConvSm] = ssm_fit._KalmanSmoother(A, Bu, mu.copy(), V, P, Pinv, obsTime, tCovConvFt, eps=0)
[Ext, Extxt, Extxtm1]  = ssm_fit._KalmanParsToMoments(mu_h, V_h, J,obsTime,tCovConvFt,tCovConvSm)
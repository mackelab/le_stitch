import numpy as np
import scipy as sp
from scipy import stats
import numbers       # to check for numbers.Integral, for isinstance(x, int)

import matplotlib
import matplotlib.pyplot as plt
from IPython import display  # for live plotting in jupyter

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _fitLDS(y, 
            u = None,
            obsScheme = None,
            initPars=None, 
            maxIter=1000, 
            epsilon=np.log(1.05), # stop if likelihood change < 5%
            ifPlotProgress=False, 
            ifTraceParamHist=False,
            ifRDiagonal=True,
            xDim=None):
    """ OUT = _fitLDS(y*,obsScheme*,initPars, maxIter, epsilon, 
                    ifPlotProgress,xDim)
        y:         data array of observed variables
        u:         data array of input variables
        obsScheme: observation scheme for given data, stored in dictionary
                   with keys 'subpops', 'obsTimes', 'obsPops'
        initPars:  set of parameters to start fitting. If == None, the 
                   parameters currently stored in the model will be used,
                   otherwise needs initPars = [A,Q,mu0,V0,C,R]
        maxIter:   maximum allowed iterations for iterative fitting (e.g. EM)
        epsilon:  convergence criterion, e.g. difference of log-likelihoods
        ifPlotProgress: boolean, specifying if fitting progress is visualized
        ifTraceParamHist: boolean, specifying if entire parameter updates 
                          history or only the current state is kept track of 
        xDim:      dimensionality of (sole subgroup of) latent state X
        ifRDiagonal: boolean, specifying diagonality of observation noise
        Fits an LDS model to data.

    """
    yDim  = y.shape[0] # Data organisation convention (also true for input u): 
    T     = y.shape[1] # yDim is dimensionality of observed data, T is trial
    Trial = y.shape[2] # length (in bins), Trial is number of trials (with
                       # idential trial length, observation structure etc.)

    if u is None or u == []:
        uDim = 0
    elif isinstance(u, np.ndarray) and len(u.shape)==3:
        uDim = u.shape[0]
    else:
        if isinstance(u, np.ndarray):
            print('u.shape')
            print( u.shape )
        else:
            print('u')
            print( u )
        raise Exception(('If provided, input data u has to be an array with '
                         'three dimensions, (uDim,T,Trial). To not include '
                         'any input data, set u = None or u = [].'))

    if obsScheme is None:
        subpops = [list(range(yDim))]
        obsTime = [T]
        obsPops = [0]

        obsScheme = {'subpops': subpops,
                     'obsTime': obsTime,
                     'obsPops': obsPops}     

        [obsIdxG, idxgrps] = _computeObsIndexGroups(obsScheme,yDim)
        obsScheme['obsIdxG'] = obsIdxG # add index groups and 
        obsScheme['idxgrps'] = idxgrps # their occurences                                
                        
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

    [A,B,Q,mu0,V0,C,d,R,xDim] = _unpackInitPars(initPars,
                                                uDim,yDim,None,
                                                ifRDiagonal)   

    E_step = _iLDS_E_step 
    M_step = _iLDS_M_step 

    # evaluate initial state       
    [Ext, Extxt, Extxtm1, LLtr] = E_step(A,B,Q,mu0,V0,C,d,R,y,u, 
                                         obsScheme)
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
        Rs   = [R]
        Exts    = [Ext]
        Extxts  = [Extxt]
        Extxtm1s= [Extxtm1]
    
    #while LL_new - LL_old > epsilon and stepCount < maxIter:
    while stepCount < maxIter:                                 # will need to change this back in the future!

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
        
        [Ext, Extxt, Extxtm1, LLtr] = E_step(A, 
                                             B,
                                             Q, 
                                             mu0, 
                                             V0, 
                                             C,
                                             d,
                                             R,
                                             y,
                                             u,
                                             obsScheme)

        LL_new = np.sum(LLtr) # discarding distinction between trials
        LLs.append(LL_new.copy())

        if ifPlotProgress:
            # dynamically plot log of log-likelihood difference
            plt.subplot(1,2,1)
            plt.plot(LLs)
            plt.xlabel('#iter')
            plt.ylabel('log-likelihood')
            plt.subplot(1,2,2)
            plt.plot(dLL)
            plt.xlabel('#iter')
            plt.ylabel('LL_{new} - LL_{old}')
            display.display(plt.gcf())
            display.clear_output(wait=True)

        if LL_new < LL_old:
            print('LL_new - LL_old')
            print( LL_new - LL_old )
            print(('WARNING! Lower bound decreased during EM '
                   'algorithm. This is impossible for an LDS. '
                   'Continue?'))
            print('Press Y to continue or N to cancel')
            #inp = input("Enter (y)es or (n)o: ")
            #if inp == "no" or inp.lower() == "n":
            #    return None # break EM loop because st. is wrong
        dLL.append(LL_new - LL_old)
            
    LLs = np.array(LLs)     
    
    if ifTraceParamHist:    
        return [As, Bs, Qs, mu0s, V0s, Cs, ds, Rs, LLs]
    else:
        return [[A], [B], [Q], [mu0], [V0], [C], [d], [R], LLs]

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _iLDS_E_step(A,B,Q,mu0,V0,C,d,R,y,u,obsScheme): 
    """ OUT = _LDS_E_step(A*,Q*,mu0*,V0*,C*,R*,y*,obsScheme*,ifRDiagonal)
    see Bishop, 'Pattern Recognition and Machine Learning', ch. 13
    for formulas and input/output naming conventions
    The variable obsScheme is a dictionary that contains information
    on observed subpopulations of y that are observed at each time point

    """ 
    Bu = np.zeros([B.shape[0], y.shape[1], y.shape[2]])
    if (isinstance(u, np.ndarray) and u.size>0 
        and u.shape[1]==y.shape[1] and u.shape[2]==y.shape[2]):
        for tr in range(y.shape[2]):
            Bu[:,:,tr] = np.dot(B, u[:,:,tr])

    [mu,V,P,logc]   = _iKalmanFilter(A,B,Q,mu0,V0,C,d,R,y,Bu,obsScheme)    
    [mu_h,V_h,J]    = _iKalmanSmoother(A,B,mu,V,P,Bu)
    [Ext,Extxt,Extxtm1] = _KalmanParsToMoments(mu_h,V_h,J)
        
    LL = np.sum(logc,axis=0) # sum over times, get Trial-dim. vector
    
    return [Ext, Extxt, Extxtm1, LL]

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _iKalmanFilter(A,B,Q,mu0,V0,C,d,R,y,Bu,obsScheme):        
    """ OUT = _KalmanFilter(A*,Q*,mu0*,V0*,C*,R*,y*,obsScheme*)
    see Bishop, 'Pattern Recognition and Machine Learning', ch. 13
    for formulas and input/output naming conventions   
    The variable obsScheme is a dictionary that contains information
    on observed subpopulations of y that are observed at each time point
    """
    xDim  = A.shape[0]
    yDim  = y.shape[0]
    uDim  = B.shape[1]
    T     = y.shape[1]
    Trial = y.shape[2]
    mu    = np.zeros([xDim,     T,Trial])
    V     = np.zeros([xDim,xDim,T,Trial])
    P     = np.zeros([xDim,xDim,T,Trial])
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
    

    for tr in range(Trial):

        # first time step: [mu0,V0] -> [mu1,V1]
        idx = subpops[obsPops[0]]

        # pre-compute for this group of observed variables
        Cj   = C[np.ix_(idx,xRange)]                    # all these
        Rinv = 1/R[idx]                                 # operations    
        CtrRinv = Cj.transpose() * Rinv                 # are order
        CtrRinvC = np.dot(CtrRinv, Cj)                  # O(yDim) !  

        # pre-compute for this time step                                           
        Cmu0B0 = np.dot(Cj,mu0+Bu[:,0,tr]) # O(yDim)
        yDiff  = y[idx,0,tr] - d[idx] - Cmu0B0          # O(yDim)   

        CtrRyDiff_Cmu0 = np.dot(CtrRinv, yDiff)         # O(yDim)
        P0   = V0 # = np.dot(np.dot(A, V0), Atr) + Q
                
        # compute Kalman gain components
        Pinv   = sp.linalg.inv(P0)                
        Kcore  = sp.linalg.inv(CtrRinvC+Pinv)                                                      
        Kshrt  = Iq  - np.dot(CtrRinvC, Kcore)
        PKsht  = np.dot(P0,    Kshrt) 
        KC     = np.dot(PKsht, CtrRinvC)        
        
        # update posterior estimates
        mu[ :,0,tr] = mu0 + np.dot(PKsht,CtrRyDiff_Cmu0)
        V[:,:,0,tr] = np.dot(Iq - KC, P0)
        P[:,:,0,tr] = np.dot(np.dot(A,V[:,:,0,tr]), Atr) + Q

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
                
        t = 1 # now start with second time step ...
        for i in range(len(obsTime)):
            idx = subpops[obsPops[i]]
                                                   
            # pre-compute for this group of observed variables
            Cj   = C[np.ix_(idx,xRange)]                    # all these
            Rinv = 1/R[idx]                                 # operations 
            CtrRinv = Cj.transpose() * Rinv                 # are order
            CtrRinvC = np.dot(CtrRinv, Cj)                  # O(yDim) !
                                                   
            while t < obsTime[i]: 
                                                   
                # pre-compute for this time step                                   
                AmuBu  = np.dot(A,mu[:,t-1,tr]) + Bu[:,t, tr] 
                yDiff  = y[idx,t,tr] - d[idx] - np.dot(Cj,AmuBu) # O(yDim)                                              
                CtrRyDiff_CAmu = np.dot(CtrRinv, yDiff)          # O(yDim)
                                                   
                # compute Kalman gain components                                       
                Pinv   = sp.linalg.inv(P[:,:,t-1,tr])       
                Kcore  = sp.linalg.inv(CtrRinvC+Pinv)                                        
                Kshrt  = Iq  - np.dot(CtrRinvC, Kcore)
                PKsht  = np.dot(P[:,:,t-1,tr],  Kshrt) 
                KC     = np.dot(PKsht, CtrRinvC)

                # update posterior estimates
                mu[ :,t,tr] = AmuBu + np.dot(PKsht,CtrRyDiff_CAmu)
                V[:,:,t,tr] = np.dot(Iq - KC,P[:,:,t-1,tr])
                P[:,:,t,tr] = np.dot(np.dot(A,V[:,:,t,tr]), Atr) + Q
                                                   
                # compute marginal probability y_t | y_0, ..., y_{t-1}
                M    = sp.linalg.cholesky(P[:,:,t-1,tr])                                 
                logdetCPCR = (  np.sum(np.log(R[idx]))                                  
                               + np.log(sp.linalg.det(Iq+np.dot(M.transpose(),
                                                         np.dot(CtrRinvC,M))))
                             )

                logc[ t,tr] = (  np.sum(Rinv * yDiff * yDiff)   
                               - np.dot(CtrRyDiff_CAmu, np.dot(Kcore, 
                                        CtrRyDiff_CAmu))
                               + logdetCPCR
                              )
                                                   
                t += 1
                                     
    logc = -1/2 * (logc + yDim * np.log(2*np.pi))
    
    return [mu,V,P,logc]

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _iKalmanSmoother(A, B, mu, V, P, Bu):        
    """ OUT = _KalmanSmoother(A*,B*,mu*,V*,P*,u*)
    see Bishop, 'Pattern Recognition and Machine Learning', ch. 13
    for formulas and input/output naming conventions   
    """        
    T     = mu.shape[1]
    Trial = mu.shape[2]
    mu_h = mu
    V_h  = V
    J    = np.zeros([mu.shape[0],mu.shape[0],T,Trial])
    Atr = A.transpose()

    for tr in range(Trial):
        AmuBu = np.dot(A, mu[:,:,tr]) + Bu[:,:,tr]
        t = T-2 # T-1 already correct by initialisation of mu_h, V_h
        while t >= 0:
            J[:,:,t,tr] = np.dot(np.dot(V[:,:,t,tr], Atr),
                                 sp.linalg.inv(P[:,:,t,tr]))
            mu_h[ :,t,tr] += np.dot(J[:,:,t,tr], mu_h[:,t+1,tr] - AmuBu[:,t]) 
            V_h[:,:,t,tr] += np.dot(np.dot(J[:,:,t,tr], 
                                            V_h[:,:,t+1,tr] - P[:,:,t,tr]),
                                    J[:,:,t,tr].transpose()) 
            t -= 1
    return [mu_h,V_h,J]

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _iLDS_M_step(Ext, Extxt, Extxtm1, y, u, obsScheme, 
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
                j   = obsPops[i]
                idx = subpops[j]
                if i == 0:
                    ts  = range(0, obsTime[i])                                            
                else:
                    ts  = range(obsTime[i-1],obsTime[i])                 
                sy[idx] += np.sum(ytr[np.ix_(idx,ts)],1)          
    if syy is None:
        syy   = np.zeros(yDim) # sum over outer product y_t y_t'
        for tr in range(Trial):
            ytr = y[:,:,tr]            
            for i in rangeobstime:
                j   = obsPops[i]
                idx = subpops[j]
                if i == 0:
                    ts  = range(0, obsTime[i])                                            
                else:
                    ts  = range(obsTime[i-1],obsTime[i])                 
                ytmp = ytr[np.ix_(idx,ts)]
                syy[idx] += np.sum(ytmp*ytmp,1) 
        del ytmp

    # compute scatter matrix for input states
    if suu is None:
        suu   = np.zeros([uDim,uDim]) # sum over outer product u_t u_t'
        for tr in range(Trial):
            utr = u[:,range(1,T),tr]            
            suu += np.dot(utr, utr.transpose())        
    if suuinv is None:
        suuinv = sp.linalg.inv(suu)   # also need matrix inverse        
    
    # compute (diagonal of) scatter matrix accros observed and latent states
    # compute scatter matrices from posterior means for the latent states
    sExt    = np.zeros(xDim)            # sums of expected values of   
    sExtxt1toN = np.zeros([xDim, xDim]) # x_t, x_t * x_t', 
    syExt      = np.zeros([yDim, xDim]) # y_t * x_t' (only for observed y_i)

    # versions of sums exclusively over data points where individual index
    # groups are observed:
    sExts   = np.zeros([xDim, len(idxgrps)])       
    sExtxts = np.zeros([xDim, xDim, len(idxgrps)]) 
    syExts  = np.zeros([yDim, xDim]) # index groups have no overlap, can store
                                     # results for y_t x_t' in a singe matrix!                

    for tr in range(Trial):              # collapse over trials ...
        ytr = y[:,:,tr]
        for i in rangeobstime:         # ... but keep  apart
            idg = obsIdxG[i]           # list of currently observed idx groups
            idx = subpops[obsPops[i]]  # list of currently observed y_i
            if i == 0:
                ts  = range(0, obsTime[i])                                            
            else:
                ts  = range(obsTime[i-1],obsTime[i])           

            tsExt   = np.sum(Ext[:,ts,tr],1)
            tsExtxt = np.sum(Extxt[:,:,ts,tr], 2)
            tsyExt  = np.einsum('in,jn->ij', 
                                ytr[np.ix_(idx,ts)], 
                                Ext[:,ts,tr])     

            sExt         += tsExt
            sExtxt1toN   += tsExtxt
            syExt[idx,:] += tsyExt

            for j in obsIdxG[i]: # for each currently observed index group:
                sExts[:,j]      += tsExt
                sExtxts[:,:,j]  += tsExtxt
                syExts[idxgrps[j],:] += tsyExt  
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

        suuvinvsuExtm1 = np.dot(suuinv, suExtm1)
        sExsuuusuExm1  = np.dot(sExtu,               suuvinvsuExtm1)
        sExm1suusuExm1 = np.dot(suExtm1.transpose(), suuvinvsuExtm1)

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
        C[ixg,:] = np.dot(syExts[ixg,:]
                          - np.outer(sy[ixg],sExts[:,i])/Ti[i], 
                          sp.linalg.inv(
                                    sExtxts[:,:,i]
                                  - np.outer(sExts[:,i],sExts[:,i])/Ti[i]
                                        )
                          )        


    # In an approximation to the ML solution for C, which is tempting if there
    # are many subpopulations (or if there is additional missing data without
    # much structure in what is missing), we may want to take the inverse of
    # the sum over all x_t x_' instead of just those outer products of latent
    # states that occured while we observed a given observed variable y(i)_t.
    # That leaves us with only one matrix inversion instead of up to yDim many
    # In terms of equations, for this approximation we assume 
    # C = (Tau^-1 sum_t I_t (y_t-d) x_t')  (1/T sum_t x_t x_'t)^-1
    # for diagonal observation index matrices I_t and Tau = sum_t I_t being
    # the new normaliser matrix. 
    # and whish to solve C * sExtxtinv  - 1/Ti * C * (sExt*sExt') = rhs for C 
    # The following may be outdated in terms of variable conventions!
    # Ti1 = Ti.reshape(yDim,1)
    # Ti2 = Ti1 * Ti1
    # rhs = syExt/Ti1 - sysExt/Ti2
    # mExtxt1toN = sExtxt1toN / (T*Trial)
    # mExtxtinv = sp.linalg.inv(mExtxt1toN)
    # z   = np.dot(mExtxtinv, sExt)
    # zz  = np.outer(z, z)
    # wii = Ti2 - np.dot(sExt, z) # Ti + trace(sExt*sExt' * inv(sExtxt))
    # C = np.dot(rhs,mExtxtinv) + np.dot(rhs/wii, zz)

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

    print('subpops')
    print(subpops)
    print('A')
    print(A)
    print('B')
    print(B)
    print('Q')
    print(Q)
    print('C')
    print(C)  
    print('d')
    print(d)  
    print('R')
    print(R)
    if not np.min(R) > 0:
        print('CsExtxtCtr')
        print(CsExtxtCtr)
        print('- 2 * syExtCtr')
        print(- 2 * syExtCtr)
        print('+ 2 * CsExtdtr')
        print(+ 2 * CsExtdtr)
        print('syy')
        print(syy)
        print('- 2 * (sy * d)')
        print(- 2 * (sy * d))
        print('+ Ti * (d * d)')         
        print(+ Ti * (d * d))      
        print('Ti')
        print(Ti)   
        print('R')
        print(R)        
        raise Exception('stop, R is ill-behaved')

    return [A,B,Q,mu0,V0,C,d,R,sy,syy,suu,suuinv,Ti]        

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _KalmanParsToMoments(mu_h, V_h, J):
    """ OUT = _KalmanParsToMoments(mu)h*,V_h*,J*)
    see Bishop, 'Pattern Recognition and Machine Learning', ch. 13
    for formulas and input/output naming conventions  
    The variable obsScheme is a dictionary that contains information
    on observed subpopulations of y that are observed at each time point 
    """                
    T    = mu_h.shape[1]
    Trial= mu_h.shape[2]

    Ext   = mu_h.copy()             # E[x_t]                        
    Extxt = V_h.copy()              # E[x_t, x_t]
    Extxtm1 = np.zeros(V_h.shape)   # E[x_t x_{t-1}'] 

    for tr in range(Trial):
        for t in range(T):
            Extxt[:,:,t,tr] += np.outer(mu_h[:,t,tr], mu_h[:,t,tr]) 
    for tr in range(Trial):
        for t in range(1,T): # t=0 stays all zeros !
            Extxtm1[:,:,t,tr] =  (np.dot(V_h[:,:, t, tr], 
                                         J[:,:,t-1,tr].transpose()) 
                                + np.outer(mu_h[:,t,tr], mu_h[:,t-1,tr]) ) 

    return [Ext, Extxt, Extxtm1]    

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _unpackInitPars(initPars, uDim, yDim=None, xDim=None, ifRDiagonal=True):   
    """ OUT = _unpackInitPars(initPars*,uDim,yDim,xDim,ifRDiagonal)
        initPars:    list of parameter arrays. Set None to get default setting 
        uDim:        dimensionality of input variables u
        yDim:        dimensionality of observed variables y
        xDim:        dimensionality of latent variables x
        ifRDiagonal: boolean, specifying if observed covariance R is diagonal
    Extracts and formats state-space model parameter for the EM algorithm.

    """
    if not isinstance(yDim, numbers.Integral) and uDim >= 0:
        print('uDim:')
        print(uDim)
        raise Exception('argument uDim has to be a non-negative integer')    

    if uDim == 0:
    # classic LDS (no inputs, no constant offsets, zero noise means):
        if initPars is None:
            initPars = [None,None,None,None,None,None,None]
        elif (not ((isinstance(initPars, list)       and len(initPars)==7) or
                   (isinstance(initPars, np.ndarray) and initPars.size==7))):
            if isinstance(initPars, list):
                print('len(initPars)')
                print(len(initPars))
            elif isinstance(initPars, np.ndarray):
                print('initPars.size')
                print(initPars.size)
            else:
                print(initPars)
            raise Exception(('argument initPars for fitting a LDS to data has '
                             'to be a list or an ndarray with exactly 7 '
                             'elements: {A,Q,mu0,V0,C,d,R}. Alternatively, it '
                             'is possible to hand over initPars = None to '
                             'get a default LDS EM-algorithm initialization.'))            

        # if not provided, figure out latent dimensionality from parameters
        if xDim is None:
            if not initPars[0] is None and isinstance(initPars[0], np.ndarray):
                xDim = initPars[0].shape[0] # we can get xDim from A
            elif  not initPars[1] is None and isinstance(initPars[1],np.ndarray):
                xDim = initPars[1].shape[0] # ... or from Q
            elif  not initPars[2] is None and isinstance(initPars[2],np.ndarray):
                xDim = initPars[2].size     # ... or from mu0
            elif  not initPars[3] is None and isinstance(initPars[3],np.ndarray):
                xDim = initPars[3].shape[0] # ... or from V0
            elif  not initPars[4] is None and isinstance(initPars[4],np.ndarray):
                xDim = initPars[4].shape[1] # ... or from C
            else: 
                raise Exception(('could not obtain xDim. Need to provide '
                                 'either xDim, or initializations for at '
                                 'least one of the following: '
                                 'A, Q, mu0, V0 or C. None was provided.'))
        elif not (isinstance(xDim, numbers.Integral) and xDim > 0):
            print('xDim:')
            print(xDim)
            raise Exception('argument xDim has to be a positive integer')
        if yDim is None:
            if  not initPars[4] is None and isinstance(initPars[4],np.ndarray):
                yDim = initPars[4].shape[0] # ... or from C
            elif  not initPars[5] is None and isinstance(initPars[5],np.ndarray):
                yDim = initPars[5].shape[0] # ... or from C
            elif  not initPars[6] is None and isinstance(initPars[6],np.ndarray):
                yDim = initPars[6].shape[0] # ... or from C
            else: 
                raise Exception(('could not obtain yDim. Need to provide '
                                 'either yDim, or initializations for at '
                                 'least one of the following: '
                                 'C, d, R. None was provided.'))
        elif not (isinstance(yDim, numbers.Integral) and yDim > 0):
            print('yDim:')
            print(yDim)
            raise Exception('argument yDim has to be a positive integer')            
        # else: we're fine

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
        if initPars[1] is None:
            Q   =       np.identity(xDim)            
        elif np.all(initPars[1].shape==(xDim,xDim)): 
            Q   = initPars[1].copy()
        else:
            print('xDim:')
            print(xDim)
            print('Q.shape:')
            print(initPars[1].shape)
            raise Exception(('Bad initialization for LDS parameter Q.'
                             'Shape not matching dimensionality of x'))
        if initPars[2] is None:
            mu0 =       np.zeros(xDim)            
        elif initPars[2].size==xDim: 
            mu0 = initPars[2].copy()
        else:
            print('xDim:')
            print(xDim)
            print('mu0.shape:')
            print(initPars[2].shape)
            raise Exception(('Bad initialization for LDS parameter mu0.'
                             'Shape not matching dimensionality of x'))
        if initPars[3] is None:
            V0  =       np.identity(xDim)            
        elif np.all(initPars[3].shape==(xDim,xDim)): 
            V0  = initPars[3].copy()
        else:
            print('xDim:')
            print(xDim)
            print('V0.shape:')
            print(initPars[3].shape)
            raise Exception(('Bad initialization for LDS parameter V0.'
                             'Shape not matching dimensionality of x'))
        if initPars[4] is None:
            C = np.random.normal(size=[yDim, xDim]) 
        elif np.all(initPars[4].shape==(yDim,xDim)):
            C = initPars[4].copy()
        else:
            print('xDim:')
            print(xDim)
            print('yDim:')
            print(yDim)
            print('C.shape:')
            print(initPars[4].shape)    
            raise Exception(('Bad initialization for LDS parameter C.'
                             'Shape not matching dimensionality of y, x'))  
        if initPars[5] is None:
            d = np.random.normal(size=[yDim]) 
        elif np.all(initPars[5].shape==(yDim,)):
            d = initPars[5].copy()
        else:
            print('yDim:')
            print(yDim)
            print('d.shape:')
            print(initPars[5].shape)    
            raise Exception(('Bad initialization for LDS parameter d.'
                             'Shape not matching dimensionality of y'))  

        if ifRDiagonal:
            if initPars[6] is None:
                R = np.ones(yDim)
            elif np.all(initPars[6].shape==(yDim,)):
                R = initPars[6].copy()
            elif np.all(initPars[6].shape==(yDim,yDim)):
                R = initPars[6].copy().diagonal()
            else:
                print('yDim:')
                print(yDim)
                print('R.shape:')
                print(initPars[6].shape) 
                raise Exception(('Bad initialization for LDS '
                                 'parameter C. Shape not matching '
                                 'dimensionality of y'))        
                
        else:
            if initPars[6] is None:
                R = np.identity(yDim)                
            elif np.all(initPars[6].shape==(yDim,yDim)):
                R = initPars[6].diagonal().copy()
            else:
                print('yDim:')
                print(yDim)
                print('R.shape:')
                print(initPars[6].shape) 
                raise Exception(('Bad initialization for LDS '
                                 'parameter C. Shape not matching '
                                 'dimensionality of y'))       
                
        return [A,None,Q,mu0,V0,C,d,R,xDim]                      
    
    elif uDim > 0:
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

    else: 
        raise Exception('Variable uDim has to be larger or equal to zero')
        
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
                      
        
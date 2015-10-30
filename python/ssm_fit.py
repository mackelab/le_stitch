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
            obsScheme,
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

    yDim = y.shape[0] 
    if not u is None:
        uDim = u.shape[0]
    else:
        uDim = 0

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
        
        
    [A,B,Q,mu0,V0,C,d,R] = _unpackInitPars(initPars, ifRDiagonal, uDim)   

    E_step = _LDS_E_step # greatly improves
    M_step = _LDS_M_step # readability ...
    
    # evaluate initial state       
    [Ext, Extxt, Extxtm1, LLtr] = E_step(A,B,Q,mu0,V0,C,d,R,y,u, 
                                         obsScheme, ifRDiagonal)
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
    
    while LL_new - LL_old > epsilon and stepCount < maxIter:

        LL_old = LL_new
        
        syy = None # initialize, then copy results, as
        Ti  = None # there is no need to compute these twice                                 
        
        [A,B,Q,mu0,V0,C,d,R,syy,Ti] = M_step(Ext, 
                                             Extxt, 
                                             Extxtm1,
                                             y, 
                                             u,
                                             obsScheme,
                                             syy,
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
                                             obsScheme,
                                             ifRDiagonal)
        LL_new = np.sum(LLtr) # discarding distinction between trials
        LLs.append(LL_new.copy())

        if ifPlotProgress:
            # dynamically plot log of log-likelihood difference
            plt.plot(dLL)
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
        dLL.append(np.log(LL_new - LL_old)/log10)
            
    LLs = np.array(LLs)     
    
    if ifTraceParamHist:    
        return [As, Qs, mu0s, V0s, Cs, Rs, LLs]
    else:
        return [[A],  [Q],  [mu0],  [V0],  [C],  [R],  LLs]
        
#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _LDS_E_step(A,B,Q,mu0,V0,C,d,R,y,u,obsScheme,ifRDiagonal=True): 
    """ OUT = _LDS_E_step(A*,Q*,mu0*,V0*,C*,R*,y*,obsScheme*,ifRDiagonal)
    see Bishop, 'Pattern Recognition and Machine Learning', ch. 13
    for formulas and input/output naming conventions
    The variable obsScheme is a dictionary that contains information
    on observed subpopulations of y that are observed at each time point
    The boolean ifRDiagonal gives if the observation noise is diagonal or not.
    """
    if not u is None:
        [mu,V,P,logc]   = _KalmanFilter(A,B,Q,mu0,V0,C,d,R,y,u,obsScheme)    
        [mu_h,V_h,J]    = _KalmanSmoother(A,mu,V,P)
        [Ext,Extxt,Extxtm1] = _KalmanParsToMoments(mu_h,V_h,J)

    else:
        print('Marcel has to work on this.')
        raise Exception('support for classical LDS currently not given')
        
    LL = np.sum(logc,axis=0) # sum over times, get Trial-dim. vector
    
    return [Ext, Extxt, Extxtm1, LL]

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _KalmanFilter(A,B,Q,mu0,V0,C,d,R,y,u,obsScheme):        
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
    Id = np.identity(xDim)

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
    
    y = y.copy - d.reshape([yDim,1,1]) # potentially heavy burden on memory!
    
    for tr in range(Trial):
        # first time step: [mu0,V0] -> [mu1,V1]
        j   = obsPops[0]
        idx = subpops[j]

        # pre-compute for this group of observed variables
        Cj   = C[np.ix_(idx,xRange)]                    # all these
        Rinv = 1/R[idx]                                 # operations    
        CtrRinv = Cj.transpose() * Rinv                 # are order
        CtrRinvC = np.dot(CtrRinv, Cj)                  # O(yDim) !            
        
        # pre-compute for this time step                                   
        Cmu0 = np.dot(Cj,mu0)
        yDiff  = y[idx,0,tr] - Cmu0                     # O(yDim)                                       
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
        V[:,:,0,tr] = np.dot(Id - KC, P0)
        P[:,:,0,tr] = np.dot(np.dot(A,V[:,:,0,tr]), Atr) + Q

        # compute marginal probability y_0
        M    = sp.linalg.cholesky(P0)
        logdetCPCR    = (  np.sum(np.log(R[idx])) 
                         + np.log(sp.linalg.det(Iq+np.dot(M.transpose(),np.dot(CtrRinvC,M))))
                        )
        logc[ 0,tr] = (  np.dot(yDiff * Rinv, yDiff)       
                       - np.dot(CtrRyDiff_Cmu0, np.dot(Kcore, CtrRyDiff_Cmu0)) 
                       + logdetCPCR
                      )
                
        t = 1 # now start with second time step ...
        for i in range(len(obsTime)):
            j   = obsPops[i]
            idx = subpops[j]
                                                   
            # pre-compute for this group of observed variables
            Cj   = C[np.ix_(idx,xRange)]                    # all these
            Rinv = 1/R[idx]                                 # operations 
            CtrRinv = Cj.transpose() * Rinv                 # are order
            CtrRinvC = np.dot(CtrRinv, Cj)                  # O(yDim) !
                                                   
            while t < obsTime[i]: 
                                                   
                # pre-compute for this time step                                   
                Amu    = np.dot(A,mu[:,t-1,tr])
                yDiff  = y[idx,t,tr] - np.dot(Cj,Amu)       # O(yDim)                                              
                CtrRyDiff_CAmu = np.dot(CtrRinv, yDiff)     # O(yDim)
                                                   
                # compute Kalman gain components                                       
                Pinv   = sp.linalg.inv(P[:,:,t-1,tr])       
                Kcore  = sp.linalg.inv(CtrRinvC+Pinv)                                        
                Kshrt  = Iq  - np.dot(CtrRinvC, Kcore)
                PKsht  = np.dot(P[:,:,t-1,tr],  Kshrt) 
                KC     = np.dot(PKsht, CtrRinvC)

                # update posterior estimates
                mu[ :,t,tr] = Amu + np.dot(PKsht,CtrRyDiff_CAmu)
                V[:,:,t,tr] = np.dot(Id - KC,P[:,:,t-1,tr])
                P[:,:,t,tr] = np.dot(np.dot(A,V[:,:,t,tr]), Atr) + Q
                                                   
                # compute marginal probability y_t | y_0, ..., y_{t-1}
                M    = sp.linalg.cholesky(P[:,:,t-1,tr])                                 
                logdetCPCR = (  np.sum(np.log(R[idx]))                                  
                                 + np.log(sp.linalg.det(Iq+np.dot(M.transpose(),np.dot(CtrRinvC,M))))
                             )

                logc[ t,tr] = (  np.dot(yDiff * Rinv, yDiff) 
                               - np.dot(CtrRyDiff_CAmu, np.dot(Kcore, CtrRyDiff_CAmu))
                               + logdetCPCR
                              )
                                                   
                t += 1
                                     
    logc = -1/2 * (logc + yDim * np.log(2*np.pi))
    
    return [mu,V,P,logc]

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _KalmanSmoother(A, mu, V, P):        
    """ OUT = _KalmanSmoother(A*,mu*,V*,P*)
    see Bishop, 'Pattern Recognition and Machine Learning', ch. 13
    for formulas and input/output naming conventions   
    """        
    xDim  = mu.shape[0]
    T     = mu.shape[1]
    Trial = mu.shape[2]
    mu_h = np.zeros([xDim,     T,Trial])
    V_h  = np.zeros([xDim,xDim,T,Trial])
    J    = np.zeros([xDim,xDim,T,Trial])
    tr = 0
    Atr = A.transpose()
    for tr in range(Trial):
        mu_h[ :,T-1,tr] = mu[ :,T-1,tr] # \beta(x_N) = 1, i.e. 
        V_h[:,:,T-1,tr] = V[:,:,T-1,tr] # \alpha(x_N) = \gamma(x_N)
        t = T-2
        while t >= 0:
            Amu         = np.dot(A,mu[:,t,tr])             
            J[:,:,t,tr] = np.dot(np.dot(V[:,:,t,tr], Atr),
                                 sp.linalg.inv(P[:,:,t,tr]))
            mu_h[ :,t,tr] = ( mu[:,t,tr] 
                            + np.dot(J[:,:,t,tr],mu_h[:,t+1,tr] - Amu) )
            V_h[:,:,t,tr] = (V[:,:,t,tr] 
                            + np.dot(np.dot(J[:,:,t,tr], 
                                            V_h[:,:,t+1,tr] - P[:,:,t,tr]),
                                     J[:,:,t,tr].transpose()) )
            t -= 1
    return [mu_h,V_h,J]

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _KalmanParsToMoments(mu_h, V_h, J):
    """ OUT = _KalmanParsToMoments(mu)h*,V_h*,J*)
    see Bishop, 'Pattern Recognition and Machine Learning', ch. 13
    for formulas and input/output naming conventions  
    The variable obsScheme is a dictionary that contains information
    on observed subpopulations of y that are observed at each time point 
    """                
    xDim = mu_h.shape[0]
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
    
    xRange = range(xDim)
    
    # unpack observation scheme
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
        
    # compute sum and (diagonal of) scatter matrix for observed states    
    if sy is None:
        sy    = np.zeros(yDim)
        for tr in range(Trial):
            ytr = y[:,:,tr]            
            for i in range(len(obsTime)):
                j   = obsPops[i]
                idx = subpops[j]
                if i == 0:
                    ts  = range(0, obsTime[i])                                            
                else:
                    ts  = range(obsTime[i-1],obsTime[i])                 
                sy[idx] += np.sum(ytr[np.ix_(idx,ts,1)])          
    if syy is None:
        syy   = np.zeros(yDim) # sum over outer product y_t y_t'
        for tr in range(Trial):
            ytr = y[:,:,tr]            
            for i in range(len(obsTime)):
                j   = obsPops[i]
                idx = subpops[j]
                if i == 0:
                    ts  = range(0, obsTime[i])                                            
                else:
                    ts  = range(obsTime[i-1],obsTime[i])                 
                ytmp = ytr[np.ix_(idx,ts)]
                syy[idx] += np.sum(ytmp*ytmp,1)  
                
    # count occurence of each observed component (for normalisation purposes)
    if Ti is None:
        Ti = np.zeros(yDim);
        for tr in range(Trial):
            for i in range(len(obsTime)):
                j   = obsPops[i]
                idx = subpops[j]
                if i == 0:
                    Ti[idx] += obsTime[0] # - 0, for t = 0                                                
                else:
                    Ti[idx] += obsTime[i] - obsTime[i-1]    
                    
    # compute scatter matrix for input states
    if suu is None:
        suu   = np.zeros([uDim,uDim]) # sum over outer product u_t u_t'
        for tr in range(Trial):
            utr = u[:,range(1,T),tr]            
            suu += np.dot(utr, utr.transpose())        
    if suuinv is None:
        suuinv = sp.linalg.inv(suu)   # also need matrix inverse        
    
    # compute scatter matrices from posterior means for the latent states
    sExt     = np.sum(Ext[:,:,:], (1,2))         # sum over E[x_t]
    sExtxtm1 = np.sum(Extxtm1[:,:,1:T,:], (2,3)) # sum over E[x_t x_{t-1}']        
    sExtxt2toN   = np.sum(             Extxt[:,:,1:T,:], (2,3) )  # sums 
    sExtxt1toN   = sExtxt2toN + np.sum(Extxt[:,:,  0,:],2)        # over 
    sExtxt1toNm1 = sExtxt1toN - np.sum(Extxt[:,:,T-1,:],2)        # E[x_t x_t']                                 
                
    # compute (diagonal of) scatter matrix accros observed and latent states
    syExts = np.zeros([yDim, xDim, len(obsTime)]) # sum over outer product y_t x_t'
    for tr in range(Trial):              # collapse over trials ...
        ytr = y[:,:,tr]
        for i in range(len(obsTime)):    # ... but keep subpopulations apart
            j   = obsPops[i]
            idx = subpops[j]         
            if i == 0:
                ts  = range(0, obsTime[i])                                            
            else:
                ts  = range(obsTime[i-1],obsTime[i])                    
            syExts[idx,:,i] += np.einsum('in,jn->ij', 
                                         ytr[np.ix_(idx,ts)], 
                                         Ext[:,ts,tr])                   
    mymExt = np.dot(sy, sExt.transpose())    
    myExt  = np.sum(syExts,2)              # normalize sum_t( y_t * x_t')
    for i in range(yDim):                  # row by row with the number 
        myExt[i,:]  = myExt[i,:] / Ti[i]   # of times that y(i) was observed       
        mymExt[i,:] = mymExt[i,:]/ Ti[i]
    my = my/Ti
        
    # compute scatter matrix accros input and latent states
    suExt = np.zeros([uDim, xDim]) # sum over outer product u_t x_t'
    for tr in range(Trial):        # collapse over trials ...
        suExt += np.einsum('in,jn->ij', 
                            utr[:,range(1,T),tr], # omit the first
                            Ext[:,range(1,T),tr]) # time step!                              
    suExtm1 = np.zeros([uDim, xDim]) # sum over outer product u_t x_t-1'
    for tr in range(Trial):          # collapse over trials ...
        suExtm1 += np.einsum('in,jn->ij', 
                              utr[:,range(1,T),tr], 
                              Ext[:,range(0,T-1),tr])                         
    
    # Start computing (closed-form) updated parameters
    
    # initial distribution parameters
    mu0 = 1/Trial * np.sum( Ext[:,0,:], 1 )                              
    V0  = 1/Trial * np.sum( Extxt[:,:,0,:], 2) - np.outer(mu0, mu0)      

    # latent dynamics paramters
    sExm1suusuExm1 = np.dot(suExtm1.transpose(), np.dot(suuinv, suExtm1)
    sExsuuusuExm1  = np.dot(suExt.transpose(),   np.dot(suuinv, suExtm1)
                      
    A = np.dot( sExtxtm1-sExm1suusuExm1, sp.linalg.inv(sExtxt1toNm1-sExsuuusuExm1) )                                    
    Atr = A.transpose()
    B = np.dot(suExt.transpose() - np.dot(A, suExtm1.transpose(), suuinv)
    sExtxtm1Atr = np.dot(sExtxtm1, Atr)
    BsuExt      = np.dot(B, suExt)
    BsuExtm1Atr = np.dot(B, np.dot(suExtm1, Atr))                
    Q = 1/(Trial*(T-1)) * (  sExtxt2toN   
                           - sExtxtm1Atr.transpose()
                           - sExtxtm1Atr 
                           + np.dot(np.dot(A, sExtxt1toNm1), Atr) 
                           - BsuExt.tranpose()
                           - BsuExt
                           + BsuExtm1Atr.transpose()
                           + BsuExtm1Atr
                           + np.dot(np.dot(B, suu), B.transpose())
                          )
    
       
    # observed state parameters
    C = (Trial*T) * np.dot(myExt-mymExt, sp.linalg.inv(sExtxt1toN-1/(Trial*T)*np.dot(sExt,sExt.transpose()))     
    d = my - 1/(Trial*T) * np.dot(C, sExt)                        
                           
    sydtr       = 
    syExtCtr    = np.zeros(yDim)
    CsExtxtCtr  = np.zeros(yDim)
    CsExtdtr    = np.zeros(yDim)
    for tr in range(Trial):
        for i in range(0,len(obsTime)):
            j   = obsPops[i]
            idx = subpops[j]
            if i == 0:
                ts  = range(0, obsTime[i])
            else:
                ts  = range(obsTime[i-1],obsTime[i])
            Cj  = C[np.ix_(idx,xRange)]
            Ctr = Cj.transpose()                                                  
            CsExtxtCtr[idx] += np.einsum('ij,ik,jk->i', Cj, Cj, np.sum(Extxt[:,:,ts,tr], 2)) 
            syExtCtr[idx]   += np.sum(syExts[idx,:,i] * Cj, 1)    
            CsExtdtr[idx]   += np.sum(np.dot(Cj, sExt)) * d
    R = (  syy - 2 * (sy * d) + d * d
         + CsExtxtCtr - 2 * syExtCtr  + 2 * CsExtdtr) / Ti
    
    return [A,Q,mu0,V0,C,R,syy,Ti]        

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _LDS_M_step(Ext, Extxt, Extxtm1, y, u, obsScheme, syy=None, Ti=None):   
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
    
    xRange = range(xDim)
    
    # unpack observation scheme
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
    
    # compute scatter matrices from posterior means for the latent states
    sExtxtm1 = np.sum(Extxtm1[:,:,1:T,:], (2,3)) # sum over E[x_t x_{t-1}']        
    sExtxt2toN   = np.sum(             Extxt[:,:,1:T,:], (2,3) )  # sums 
    sExtxt1toN   = sExtxt2toN + np.sum(Extxt[:,:,  0,:],2)        # over 
    sExtxt1toNm1 = sExtxt1toN - np.sum(Extxt[:,:,T-1,:],2)        # E[x_t x_t']  

    # compute (diagonal of) scatter matrix for observed states
    if syy is None:
        syy   = np.zeros(yDim) # sum over outer product y_t y_t'
        for tr in range(Trial):
            ytr = y[:,:,tr]            
            for i in range(len(obsTime)):
                j   = obsPops[i]
                idx = subpops[j]
                if i == 0:
                    ts  = range(0, obsTime[i])                                            
                else:
                    ts  = range(obsTime[i-1],obsTime[i])                 
                ytmp = ytr[np.ix_(idx,ts)]
                syy[idx] += np.sum(ytmp*ytmp,1)  
    
    # count occurence of each observed component (for normalisation purposes)
    if Ti is None:
        Ti = np.zeros([yDim]);
        for tr in range(Trial):
            for i in range(len(obsTime)):
                j   = obsPops[i]
                idx = subpops[j]
                if i == 0:
                    Ti[idx] += obsTime[0] # - 0, for t = 0                                                
                else:
                    Ti[idx] += obsTime[i] - obsTime[i-1]                                       
                
    # compute (diagonal of) scatter matrix accros observed and latent states
    syExts = np.zeros([yDim, xDim, len(obsTime)]) # sum over outer product y_t x_t'
    for tr in range(Trial):              # collapse over trials ...
        ytr = y[:,:,tr]
        for i in range(len(obsTime)):    # ... but keep subpopulations apart
            j   = obsPops[i]
            idx = subpops[j]         
            if i == 0:
                ts  = range(0, obsTime[i])                                            
            else:
                ts  = range(obsTime[i-1],obsTime[i])                    
            syExts[idx,:,i] += np.einsum('in,jn->ij', 
                                         ytr[np.ix_(idx,ts)], 
                                         Ext[:,ts,tr])                         
    myExt = np.sum(syExts,2)               # normalize sum_t( y_t * x_t')
    for i in range(yDim):                  # row by row with the number 
        myExt[i,:] = myExt[i,:] / Ti[i]    # of times that y(i) was observed                                  
    
    
    # Start computing (closed-form) updated parameters
    
    # initial distribution parameters
    mu0 = 1/Trial * np.sum( Ext[:,0,:], 1 )                              
    V0  = 1/Trial * np.sum( Extxt[:,:,0,:], 2) - np.outer(mu0, mu0)      

    # latent dynamics paramters
    A = np.dot( sExtxtm1, sp.linalg.inv(sExtxt1toNm1) )                                    
    Atr = A.transpose()
    sExtxtm1Atr = np.dot(sExtxtm1, Atr)
    Q = 1/(Trial*(T-1)) * (  sExtxt2toN   
                           - sExtxtm1Atr.transpose()
                           - sExtxtm1Atr 
                           + np.dot(np.dot(A, sExtxt1toNm1), Atr) ) 
    
       
    # observed state parameters
    C = (Trial*T) * np.dot(myExt, sp.linalg.inv(sExtxt1toN))                    
    
    syExtCtr    = np.zeros(yDim)
    CsExtxtCtr  = np.zeros(yDim)
    for tr in range(Trial):
        for i in range(0,len(obsTime)):
            j   = obsPops[i]
            idx = subpops[j]
            if i == 0:
                ts  = range(0, obsTime[i])
            else:
                ts  = range(obsTime[i-1],obsTime[i])
            Cj  = C[np.ix_(idx,xRange)]
            Ctr = Cj.transpose()                                                  
            CsExtxtCtr[idx] += np.einsum('ij,ik,jk->i', Cj, Cj, np.sum(Extxt[:,:,ts,tr], 2)) 
            syExtCtr[idx]   += np.sum(syExts[idx,:,i] * Cj, 1)    
    R = (  syy
         - 2 * syExtCtr
         + CsExtxtCtr ) / Ti
    
    return [A,Q,mu0,V0,C,R,syy,Ti]        

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def _unpackInitPars(initPars, ifRDiagonal=True, uDim=0, xDim=None):   

    # we will work with continuously changing working copies of
    # the parameter initializations (no memory track along updates):
    
    if uDim == 0:
    # classic LDS (no inputs, no constant offsets, zero noise means):
        if initPars is None:
        initPars = [None,None,None,None,None,None]
        elif (not ((isinstance(initPars, list)       and len(initPars)==6) or
                   (isinstance(initPars, np.ndarray) and initPars.size==6))):
            print(initPars)
            raise Exception(('argument initPars for fitting a LDS to data has '
                             'to be a list or an ndarray with exactly 6 '
                             'elemnts: {A,Q,mu0,V0,C,R}. Alternatively, it is '
                             'possible to hand over initPars = None to get '
                             'a default LDS EM-algorithm initialization.'))            

        # if not provided, figure out latent dimensionality from parameters
        if xDim is None:
            if not initPars[0] is None and isinstance(initPars[0], np.array):
                xDim = initPars[0].shape[0] # we can get xDim from A
            elif  not initPars[1] is None and isinstance(initPars[1],np.array):
                xDim = initPars[1].shape[0] # ... or from Q
            elif  not initPars[2] is None and isinstance(initPars[2],np.array):
                xDim = initPars[2].size     # ... or from mu0
            elif  not initPars[3] is None and isinstance(initPars[3],np.array):
                xDim = initPars[3].shape[0] # ... or from V0
            elif  not initPars[4] is None and isinstance(initPars[4],np.array):
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
        if ifRDiagonal:
            if initPars[5] is None:
                R = np.ones(yDim)
            elif np.all(initPars[5].shape==(yDim,)):
                R = initPars[5].copy()
            else:
                print('yDim:')
                print(yDim)
                print('R.shape:')
                print(initPars[5].shape) 
                raise Exception(('Bad initialization for LDS '
                                 'parameter C. Shape not matching '
                                 'dimensionality of y'))        
                
        else:
            if initPars[5] is None:
                R = np.identity(yDim)                
            elif np.all(initPars[5].shape==(yDim,yDim)):
                R = initPars[5].diagonal().copy()
            else:
                print('yDim:')
                print(yDim)
                print('R.shape:')
                print(initPars[5].shape) 
                raise Exception(('Bad initialization for LDS '
                                 'parameter C. Shape not matching '
                                 'dimensionality of y'))       
                
    B = None
    d = None
    return [A,B,Q,mu0,V0,C,d,R]                      
    
    elif uDim > 0:
    # extended LDS, with input matrix B and constant observation offset d
        if initPars is None:
        initPars = [None,None,None,None,None,None]
        elif (not ((isinstance(initPars, list)       and len(initPars)==8) or
                   (isinstance(initPars, np.ndarray) and initPars.size==8))):
            print(initPars)
            raise Exception(('argument initPars for fitting a LDS with input '
                             'to data has to be a list or an ndarray with '
                             ' exactly 8 elemnts: {A,BQ,mu0,V0,C,dR}. '
                             'Alternatively, it is possible to hand over '
                             'initPars = None to obtain a default input-LDS '
                             'EM-algorithm initialization.'))            

    
    elif ((isinstance(initPars, list)       and len(initPars)==8) or
          (isinstance(initPars, np.ndarray) and initPars.size==8)):
        
        # if not provided, figure out latent dimensionality from parameters
        if xDim is None:
            if not initPars[0] is None and isinstance(initPars[0], np.array):
                xDim = initPars[0].shape[0] # we can get xDim from A
            if not initPars[1] is None and isinstance(initPars[1], np.array):
                xDim = initPars[1].shape[0] # we can get xDim from B
            elif  not initPars[2] is None and isinstance(initPars[2],np.array):
                xDim = initPars[2].shape[0] # ... or from Q
            elif  not initPars[3] is None and isinstance(initPars[3],np.array):
                xDim = initPars[3].size     # ... or from mu0
            elif  not initPars[4] is None and isinstance(initPars[4],np.array):
                xDim = initPars[4].shape[0] # ... or from V0
            elif  not initPars[5] is None and isinstance(initPars[5],np.array):
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
    else: 
        raise Exception('Variable uDim has to be larger or equal to zero')
        
    return [A,B,Q,mu0,V0,C,d,R]                  
        
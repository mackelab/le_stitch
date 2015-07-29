
#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def logLikelihood(A,Q,mu0,V0,C,R,Ext,Extxt,Extxtm1,y):

    xDim  = Ext.shape[0]
    T     = Ext.shape[1]
    Trial = Ext.shape[2]    
    yDim  = y.shape[0]

    # pre-compute statistics important for the terms related to x_1    
    V0inv = np.linalg.inv(V0)
    sMu0mMu1V0Inv = 0 # sum over (mu0 - mu1_h)' V0^-1 (mu0 - mu1_h)
    sV0invV1_h    = 0 # sum over trace(V0^-1 V1_h)
    tr = 0
    while tr < Trial:
        V1_h  = Extxt[:,:,0,tr] - np.outer(Ext[:,0,tr], Ext[:,0,tr])
        sV0invV1_h    += np.trace(np.dot(V0inv, V1_h))
        sMu0mMu1V0Inv += np.inner(mu0 - Ext[:,0,tr], np.dot(V0inv, mu0-Ext[:,0,tr])) 
        tr += 1
    
    # pre-compute statistics important for the terms related to x_t | x_{t-1}
    sExtxtm1 = np.sum(Extxtm1[:,:,1:T,:], (2,3)) # sum over E[ x_t x_{t-1}' ]        
    sExtxt2toN   = np.sum(             Extxt[:,:,1:T,:], (2,3) )  # sums over
    sExtxt1toN   = sExtxt2toN + np.sum(Extxt[:,:,  0,:],2)  # E[x_t x_t'] for 
    sExtxt1toNm1 = sExtxt1toN - np.sum(Extxt[:,:,T-1,:],2)  # different indexes                                                                     
        
    # pre-compute statistics important for the terms related to y_t | x_t
    Rinv  = np.linalg.inv(R)
    syRinvCExt = 0 # sum over quadratics y_t' R^-1 E[x_t]                                 
    syRinvy    = 0 # sum over quadratics y_t' R^-1 y_t'                                  
    tr = 0
    while tr < Trial:
        t = 0
        while t < T:
            Rinvy = np.dot(Rinv, y[:,t,tr])
            syRinvCExt += np.inner(  y[:,t,tr],           Rinvy)
            syRinvy    += np.inner(np.dot(C,Ext[:,t,tr]), Rinvy)                                   
            t += 1
        tr += 1    
    
    LL = 0
    
    # add E_q[log p(x_1 | Y, \theta)], i.e. cross-entropy of q(x_1), p(x_1)
    LL -= 1/2 * ( Trial * np.log(np.linalg.det(V0))
                + sV0invV1_h
                + sMu0mMu1V0Inv
                + Trial * xDim * np.log(2*sp.pi)          # + constant in mu0, V0 
                ) 
    # add E_q[log p(x_t, x_{t-1} | Y, \theta)] for t = 2, ..., T
    Qinv   = np.linalg.inv(Q)    
    QinvA  = np.dot(Qinv, A)
    AQinvA = np.dot(A.transpose(), QinvA)    
    LL -= 1/2 * ( Trial * (T-1) * np.log(np.linalg.det(Q))
                +     np.sum(  Qinv  * sExtxt2toN ) 
                - 2 * np.sum(  QinvA * sExtxtm1 )
                +     np.sum( AQinvA * sExtxt1toNm1 )
                + Trial * (T-1) * xDim * np.log(2*sp.pi)  # + constant in A, Q 
                ) 
    # add E_q[log p(y_t, x_t     |  \theta)  ] for t = 1, ..., T
    CRinvC = np.dot(np.dot( C.transpose(), Rinv), C)    
    LL -= 1/2 * ( Trial * T * np.log(np.linalg.det(R)) 
                +     syRinvy 
                - 2 * syRinvCExt 
                +     np.sum( CRinvC * sExtxt1toN ) 
                + Trial * T * yDim * np.log(2*sp.pi)      # + constant in C, R
                )
    # add - E_q[log q( x )], i.e. the entropy of q(x) 
    LL += 1/2 * ( Trial * np.log(np.linalg.det(V0))       # from H[x_1]
                + Trial *(T-1)* np.log(np.linalg.det(Q))  # from H[x_n|x_{n-1}] 
                + Trial * T * xDim * np.log(2*sp.pi)      # + constant in A, Q 
                )
    
    return LL

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def KalmanFilter(A,Q,mu0,V0,C,R,y):
    xDim  = A.shape[0]
    yDim  = y.shape[0]
    T     = y.shape[1]
    Trial = y.shape[2]
    mu = np.zeros([xDim,     T,Trial])
    V  = np.zeros([xDim,xDim,T,Trial])
    P  = np.zeros([xDim,xDim,T,Trial])
    K  = np.zeros([xDim,yDim,T,Trial])
    Id = np.identity(xDim)
    tr = 0
    
    Atr = A.transpose()
    Ctr = C.transpose()
    while tr < Trial:
        # first time step: [mu0,V0] -> [mu1,V1]
        P0      = V0 # = np.dot(np.dot(A, V0), Atr) + Q
        CPCRinv = np.linalg.inv(np.dot(np.dot(C,P0), Ctr) + R)
        K[:,:,0,tr] = np.dot(np.dot(P0,Ctr),CPCRinv)
        mu[ :,0,tr] = mu0 + np.dot(K[:,:,0,tr],y[:,0,tr]-np.dot(C,mu0))
        V[:,:,0,tr] = np.dot(Id - np.dot(K[:,:,0,tr],C),P0)
        P[:,:,0,tr] = np.dot(np.dot(A,V[:,:,0,tr]), Atr) + Q
        t = 1 # now start with second time step ...
        while t < T:
            Amu  = np.dot(A,mu[:,t-1,tr])
            PCtr = np.dot(P[:,:,t-1,tr], Ctr)
            CPCRinv = np.linalg.inv(np.dot(C, PCtr) + R)
            K[:,:,t,tr] = np.dot(PCtr,CPCRinv)
            mu[ :,t,tr] = Amu + np.dot(K[:,:,t,tr],y[:,t,tr]-np.dot(C,Amu))
            V[:,:,t,tr] = np.dot(Id - np.dot(K[:,:,t,tr],C),P[:,:,t-1,tr])
            P[:,:,t,tr] = np.dot(np.dot(A,V[:,:,t,tr]), Atr) + Q
            t += 1
        tr += 1
    return [mu,V,P,K]

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def KalmanSmoother(A, mu, V, P):
    xDim  = mu.shape[0]
    T     = mu.shape[1]
    Trial = mu.shape[2]
    mu_h = np.zeros([xDim,     T,Trial])
    V_h  = np.zeros([xDim,xDim,T,Trial])
    J    = np.zeros([xDim,xDim,T,Trial])
    tr = 0
    Atr = A.transpose()
    while tr < Trial:
        mu_h[ :,T-1,tr] = mu[ :,T-1,tr] # \beta(x_N) = 1, i.e. 
        V_h[:,:,T-1,tr] = V[:,:,T-1,tr] # \alpha(x_N) = \gamma(x_N)
        t = T-2
        while t >= 0:
            Amu         = np.dot(A,mu[:,t,tr])             
            J[:,:,t,tr] = np.dot(np.dot(V[:,:,t,tr], Atr),
                                 np.linalg.inv(P[:,:,t,tr]))
            mu_h[ :,t,tr] = ( mu[:,t,tr] 
                            + np.dot(J[:,:,t,tr],mu_h[:,t+1,tr] - Amu) )
            V_h[:,:,t,tr] = (V[:,:,t,tr] 
                            + np.dot(np.dot(J[:,:,t,tr], 
                                            V_h[:,:,t+1,tr] - P[:,:,t,tr]),
                                     J[:,:,t,tr].transpose()) )
            t -= 1
        tr += 1
    return [mu_h,V_h,J]

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def E_step(A,Q,mu0,V0,C,R,y): 
    
    [mu,V,P,K]   = KalmanFilter(  A, Q, mu0, V0, C, R, y)
    [mu_h,V_h,J] = KalmanSmoother(A, mu, V, P)
    
    [Ext, Extxt, Extxtm1] = KalmanParsToMoments(mu_h, V_h, J)
    
    return [Ext, Extxt, Extxtm1]

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def KalmanParsToMoments(mu_h, V_h, J):
    
    xDim = mu_h.shape[0]
    T    = mu_h.shape[1]
    Trial= mu_h.shape[2]

    Ext   = mu_h.copy()             # E[x_t]                        
    Extxt = V_h.copy()              # E[x_t, x_t]
    tr = 0
    while tr < Trial:
        t = 0
        while t < T:
            Extxt[:,:,t,tr] += np.outer(mu_h[:,t,tr], mu_h[:,t,tr]) 
            t += 1
        tr += 1
    Extxtm1 = np.zeros(V_h.shape)   # E[x_t x_{t-1}'] 
    tr = 0
    while tr < Trial:
        t = 1 # t=0 stays all zeros !
        while t < T:
            Extxtm1[:,:,t,tr] =  (np.dot(V_h[:,:, t, tr], 
                                         J[:,:,t-1,tr].transpose()) 
                                + np.outer(mu_h[:,t,tr], mu_h[:,t-1,tr]) ) 
            t += 1
        tr += 1                        
                        
    return [Ext, Extxt, Extxtm1]

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def M_step(Ext, Extxt, Extxtm1, y):
    xDim  = Ext.shape[0]
    T     = Ext.shape[1]
    Trial = Ext.shape[2]    
    yDim  = y.shape[0]
                                  
    sExtxt2toN   = np.sum(             Extxt[:,:,1:T,:], (2,3) )  # sums over
    sExtxt1toN   = sExtxt2toN + np.sum(Extxt[:,:,  0,:],2)  # E[x_t x_t'] for 
    sExtxt1toNm1 = sExtxt1toN - np.sum(Extxt[:,:,T-1,:],2)  # different indexes                                                                     
    sExtxtm1 = np.sum(Extxtm1[:,:,1:T,:], (2,3)) # sum over E[ x_t x_{t-1}' ]
                                                 # starting at t = 2                                  
    syExt = np.zeros([yDim,xDim]) # sum over outer product y_t x_t'                                 
    syy   = np.zeros([yDim,yDim]) # sum over outer product y_t y_t'                                  
    tr = 0
    while tr < Trial:
        t = 0
        while t < T:
            syExt += np.outer(y[:,t,tr], Ext[:,t,tr])
            syy   += np.outer(y[:,t,tr],   y[:,t,tr])                                   
            t += 1
        tr += 1

    mu0 = 1/Trial * np.sum( Ext[:,0,:], 1 ) 
    V0  = 1/Trial * np.sum( Extxt[:,:,0,:], 2) - np.outer(mu0, mu0) 
                                  
    A = np.dot(  sExtxtm1, np.linalg.inv(sExtxt1toNm1) )                                    
    Atr = A.transpose()
    sExtxtm1Atr = np.dot(sExtxtm1, Atr)
    Q = 1/(Trial*(T-1)) * (  sExtxt2toN  
                           - sExtxtm1Atr.transpose()
                           - sExtxtm1Atr 
                           + np.dot(np.dot(A, sExtxt1toNm1), Atr) ) 
                                  
    C = np.dot(syExt, np.linalg.inv(sExtxt1toN))
    Ctr = C.transpose()
    syExtCtr = np.dot(syExt, Ctr)
    R = 1/(Trial*T) * (  syy 
                       - syExtCtr.transpose()
                       - syExtCtr
                       + np.dot(np.dot(C, sExtxt1toN), Ctr) )
               
    
    return [A,Q,mu0,V0,C,R]


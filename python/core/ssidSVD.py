# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 18:25:27 2016

@author: artur
"""

import numpy as np
import warnings
import control

def ssidSVD(SIGfp,SIGyy,xDim,varargin = None):
    
    usePiHack = 0
    minVar    = 1e-5
    minVarPi  = 1e-5   
    
    '''
    assignopts(who,varargin);
    '''
    
    yDim = np.size(SIGyy,0)
    
    UU,SS,VV = np.linalg.svd(SIGfp)
    VV = VV.T
    SS = np.diag(SS)
    UU = UU[:,:xDim]
    SS = SS[:xDim,:xDim]
    VV = VV[:,:xDim]
    SSs = np.sqrt(SS)
    
    Obs = np.dot(UU,SS)
    Con = VV.T
    A = np.linalg.lstsq(Obs[:-yDim,:],Obs[yDim:,:])[0]
    C = Obs[:yDim,:]
    Chat = Con[:xDim,:yDim].T
    
#    Pi = control.matlab.dare(A.T,-C.T,np.zeros([xDim,xDim]),-SIGyy, Chat.T)
    try:
        
        Pi = control.matlab.dare(A.T,-C.T,np.zeros([xDim,xDim]),-SIGyy, Chat.T)
        
    except:
         
        warnings.warn('Cannot solve DARE, using heuristics; this might lead to poor estimates of Q and Q0')
        
        Pi = np.linalg.lstsq(A,np.dot(Chat.T,np.linalg.pinv(C.T)))[0]
        
        if usePiHack:
            print(' -----try new Pi hack----- ')
            Pi    = np.dot(np.dot(np.linalg.pinv(C),SIGyy),np.linalg.pinv(C.T))
       
    D, V = np.linalg.eig(Pi)
    D[D < minVarPi] = minVarPi
    Pi = np.dot(np.dot(V,np.diag(D)),V.T)
    Pi = np.real((Pi+Pi.T)/2)
    Q = Pi - np.dot(np.dot(A,Pi),A.T)
    D, V = np.linalg.eig(Q); 
    D[D<minVar] = minVar

    Q = np.dot(np.dot(V,np.diag(D)),V.T)
    Q = (Q+Q.T)/2
    
    R = np.diag(SIGyy-np.dot(np.dot(C,Pi),C.T))
    R.flags.writeable = True
    R[R<minVar] = minVar
    R = np.diag(R)   
    
    return A, C, Q, R, Pi
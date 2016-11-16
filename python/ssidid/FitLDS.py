# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:31:09 2016

@author: artur
"""

import numpy as np
import generateCovariancesFP
import ssidSVD
import control

def FitLTSParamsSSID(seq, xDim, varagin = None):

    algo            = 'SVD'
    hS              = xDim
    minFanoFactor   = 1.01
    minEig          = 0.0001
    params          = dict
    saveHankel      = 0
    doNonlinTransf  = 0
    useB            = 0
    
    yall            = seq['y']
    yDim, allT      = yall.shape
    minMoment       = 5/allT
    
    '''
    extraOptsIn     = assignopts(who, varargin)
    '''
    
    print('Fitting data with LDS-SSID')
    print('---------------------------')
    print('using HankelSize = %i\n',hS);
    print('useB = %i \n',useB)
    print('doNonlinTransform = %i \n',doNonlinTransf);    
    
    if useB: algo = 'n4SID'
    
    SIGfp, SIGff, SIGpp, _ = generateCovariancesFP.generateCovariancesFP(seq, hS)
    
    mu       = np.mean(yall,axis = 1)
    muBig    = np.tile(mu, 2*hS)
    gammaBig = muBig
    
    SIGBig   = np.r_[np.c_[np.copy(SIGff), np.copy(SIGfp)], np.c_[np.copy(SIGfp.T), np.copy(SIGpp)]]
    
    SIGyy    = SIGBig[:yDim,:yDim]
    SIGfp    = SIGBig[:yDim*hS,yDim*hS:]
    SIGff    = SIGBig[:yDim*hS,:yDim*hS]
    SIGpp    = SIGBig[yDim*hS:,yDim*hS:]

    SIGBig   = np.r_[np.c_[SIGff, SIGfp], np.c_[SIGfp.T, SIGpp]]
    params = {'d':gammaBig[:yDim]}
    
    if algo == 'SVD':
        
        print('PPLDSID: Using SVD with hS %d \n',hS);
        
        A, C, Q, R, Q0 = ssidSVD.ssidSVD(SIGfp,SIGyy,xDim);
        params.update({'A': A,'C': C,'Q': Q,'R': R,'Q0': Q0,})
        
    else:
        print('Not implemented')
        
    if isinstance(params['Q'], complex):
        params['Q'] = np.real(params['Q'])
        
    params['Q'] = (params['Q'] + params['Q'].T)/2
    
    if np.min(np.linalg.eig(params['Q'])[0]) < 0:
        
        a,b = np.linalg.eig(params['Q'])
        params['Q'] = a*np.max(b,10**-10)*a.T
    
    if 'Q0' not in params:
        params['Q0'] = np.real(control.dlyap(params['A'],params['Q']))
    
    params.update({'x0':np.zeros(xDim)}) 
    params['R']  = np.diag(np.diag(params['R']))
    
    if saveHankel:
        params.SIGBig = SIGBig;
      
    print('Done!')
    return params, SIGBig
    
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:29:56 2016

@author: artur
"""

import numpy as np
import warnings

def generateCovariancesFP(seqIn,hankelSize,varargin = None):
    
    checkEVFlag = False
    circFlag    = True
    trialFlag   = False
    
    '''
    assignopts(who, varargin)
    '''
    
    yDim = np.size(seqIn['y'], axis = 0)
    
    SIGfp  = np.zeros([yDim*hankelSize,yDim*hankelSize])
    SIGff  = np.zeros([yDim*hankelSize,yDim*hankelSize])
    SIGpp  = np.zeros([yDim*hankelSize,yDim*hankelSize])
    
    if circFlag:
        
        Ytot    = seqIn['y']
        Ttot    = np.size(Ytot, axis = 1)
        Ytot    = Ytot - np.tile(np.mean(Ytot, axis = 1),(Ttot,1)).T
        Yshift  = Ytot
        
        lamK    = np.dot(Ytot,Yshift.T)/2
        
        for k in range(hankelSize):

            SIGff[k*yDim:yDim+k*yDim, k*yDim:yDim+k*yDim] = lamK
            
        SIGpp = np.copy(SIGff)
    
        for k in range(2*hankelSize-1):
            
            Yshift  = np.roll(Yshift, 1, axis = 1)
            lamK    = np.dot(Ytot,Yshift.T)

            if k < hankelSize-1.5:
                
                for kk in range(k,hankelSize-1):

                    SIGff[yDim+kk*yDim:yDim+(kk+1)*yDim,(kk-k)*yDim:yDim+(kk-k)*yDim] = lamK
                    SIGpp[yDim+kk*yDim:yDim+(kk+1)*yDim,(kk-k)*yDim:yDim+(kk-k)*yDim] = lamK.T

            if k < hankelSize-0.5:
                
                for kk in range(k+1):

                    SIGfp[(k-kk)*yDim:yDim+(k-kk)*yDim,(kk)*yDim:yDim+(kk)*yDim] = lamK
                    
            else:
                
                for kk in range(2*hankelSize - k -1):

                    SIGfp[(hankelSize-kk-1)*yDim:yDim+(hankelSize-kk-1)*yDim,(kk+k+1-hankelSize)*yDim:yDim+(kk+k+1-hankelSize)*yDim] = lamK
                 
        SIGfp = SIGfp/Ttot
        SIGff = (SIGff+SIGff.T)/Ttot
        SIGpp = (SIGpp+SIGpp.T)/Ttot       
    
    else:
        
        if not trialFlag:
            
            Ytot = seqIn['y']
            """?????"""
            #[0]['y'] = Ytot       
            seq['y'] = Ytot 
        else:
            
            seq = seqIn
            
        Trials = np.size(seq, 1)
        DhsTot = 0
        
        for tr in range(Trials):
            
            T = np.size(seq[0]['y'], 1)
            Dhs = T - 2*hankelSize + 1
            
            if Dhs>2:
                
                DhsTot  = DhsTot+Dhs
                Yf = np.zeros[hankelSize*yDim,Dhs]
                Yp = np.zeros[hankelSize*yDim,T-2*hankelSize+1]
            
                for kk in range(hankelSize):
                    
                    seq[tr]['y'] = seq[tr]['y'] - np.mean[seq[tr]['y'],1]
                	
                    Yf[(kk-1)*yDim+1:kk*yDim,:] = seq[tr]['y'][:,hankelSize+kk:T-hankelSize+kk]
                    Yp[(kk-1)*yDim+1:kk*yDim,:] = seq[tr]['y'][:,hankelSize+1-kk:T-hankelSize-kk+1]
                    
                SIGfp = SIGfp + Yf*Yp.T
                SIGff = SIGff + Yf*Yf.T
                SIGpp = SIGpp + Yp*Yp.T    

        if DhsTot>2:
            SIGfp = SIGfp/DhsTot 
            SIGff = SIGff/DhsTot
            SIGpp = SIGpp/DhsTot
        else:
            warnings.warn('hankelSize too large, cannot estimate Hankel matrix!')

    SIGtot = np.c_[np.r_[SIGff, SIGfp], np.r_[SIGfp.T, SIGpp]]
    
    if checkEVFlag:
        
       V, D = np.linalg.eig(SIGtot) 
       D = np.diag(D)
       
       if min(np.real(D))<0:
           
            warnings.warn('smth wrong, future-past cov not psd; fixing it')
            print('\n min EV: %d \n \n', np.min(np.real(D)))
            D[D < 0] = 0;
            SIGtot = np.dot(np.dot(V,np.linalg.diag(D)),V.T)
            SIGff = SIGtot[:hankelSize*yDim,:hankelSize*yDim]
            SIGfp = SIGtot[:hankelSize*yDim,+hankelSize*yDim:]
            SIGpp = SIGtot[1+hankelSize*yDim:,1+hankelSize*yDim:]
         
    return SIGfp, SIGff, SIGpp, SIGtot
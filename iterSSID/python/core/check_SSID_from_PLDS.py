# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 18:58:37 2016

@author: artur
"""

import numpy as np
import FitLDS

data = np.load('data/LDS_save.npz')
seq = dict

seq = {'x':data['stateseq'].T, 'y':data['data'].T, 'T':data['T']}
xDim = np.size(seq['x'],0)

params,_ = FitLDS.FitLTSParamsSSID(seq,xDim)
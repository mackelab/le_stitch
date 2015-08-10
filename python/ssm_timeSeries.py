
import numpy as np
import scipy as sp
from scipy import stats
import numbers       # to check for numbers.Integral, for isinstance(x, int)
import matplotlib
import matplotlib.pyplot as plt
from IPython import display  # for live plotting in jupyter


""" *TBD* : - DEBUG
            -Replace lists with arrays where appropriate-arrays can 
             be more memory overhead, but also be much faster!
            -Many arguments would be nice to be also providable as 
             dictionaries: instead of e.g. always having to update ALL
             parameters, just give a dictionary - the keys will tell
             which parameters to update
            -Store the time at which each data trace was generated and
             update accordingly at construction and at _reinitData()
            -Allow covariates such as physical locations of variable nodes
            -Extend _seediidNoise() to include any desired distribution.
            -Update/complete class and method descriptions
            -Write sampling code, not just allocate and fill with noise!
            -Redirect simple queries on the level of .data and .model to
             be answerable also for .tsobject (write a bunch of methods)
            -Write a GUI to 'draw' basic state-space models by ticking
             dependency relations and selecting link functions and noise
             distributions (e.g. linear-Gaussian) from dropdown menus.
            -overhaul setStateSpaceModel() to better allow adding .models()
            -get pandas objects to replace much of the more prominent arrays
            -JHM: rename 'experiment' into 'trial' (one rarely has same-length
                  trials in reality, so most of what currently is termed as
                  'experiments' would have to be represented as single-trial).
                  Will have to rename 'trials' into trial_repets or something.
                  Alternatively: just hide the naming somewhere...
                  
"""             





#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def setStateSpaceModel(ssmType, dims, pars=None, seq=None):
    """ Data bank for standard state-space models """

    if isinstance(dims, (list, np.ndarray)):
        dims = toArray(dims)
    else:
        raise Exception(('argument dims has to be list or array'))
                       
    deps = []
    linkFunctions     = [[],[]]
    linkPars          = [[],[]]
    noiseDistrs       = [[],[]]
    noiseIntActTypes  = [[],[]]
    noisePars         = [[],[]]
    initDistrs        = []
    initPars          = []            
            
    if ssmType == 'LDS':
                       
        if dims.size != 2:
            print('dims:')
            print(dims)
            raise Exception(('LDS has two groups of variables, X and Y, and '
                             'dims hence needs to be of size two'))
        [xDim,yDim] = checkInDims(dims)
        
        if pars is None: # populate default parameters
            A    = np.identity(xDim)
            Q    = np.identity(xDim)
            mu0  = np.zeros(xDim)
            V0   = np.identity(xDim)
            C = np.ones([yDim,xDim])
            R = np.identity(yDim)
        elif isinstance(pars, (list,np.ndarray)):
            if   isinstance(pars,np.ndarray): 
                if np.size(pars)!=6:
                    print('number of parameters handed over:')
                    print(np.size(pars))
                    raise Exception(('LDS is defined by exactly 6 parameters: '
                                     '{A,Q,mu0,V0,C,R}. Mismatch!'))            
            elif isinstance(pars,list):
                if len(pars)!=6:
                    print('number of parameters handed over:')
                    print(len(pars))
                    raise Exception(('LDS is defined by exactly 6 parameters: '
                                     '{A,Q,mu0,V0,C,R}. Mismatch!'))
            A    = pars[0]
            Q    = pars[1]
            mu0  = pars[2]
            V0   = pars[3]
            C    = pars[4]
            R    = pars[5]
        else:
            raise Exception(('argument pars has to be list or ndarray '
                             'containing the parameters defining an LDS. '
                             'It is neither.'))
                                   
        xGr = np.zeros(dims[0],dtype=int)
        yGr = np.zeros(dims[1],dtype=int)
        uGr = np.zeros(      0,dtype=int)

        dts = toArray([-1,0])
            
        tmp = np.zeros([3,2], dtype=np.ndarray)         # dt = -1
        tmp[0,0] =  np.ones( [1,1],dtype=bool)  # X|X   read e.g.  | x1 | x2 
        tmp[1,0] =  np.zeros([1,1],dtype=bool)  # X|Y   X|Y as     |--------
        tmp[2,0] =  np.zeros([1,1],dtype=bool)  # X|U           y1 | o  |
        tmp[0,1] =  np.zeros([1,1],dtype=bool)  # Y|X           y2 |    | o 
        tmp[1,1] =  np.zeros([1,1],dtype=bool)  # Y|Y
        tmp[2,1] =  np.zeros([1,1],dtype=bool)  # Y|U
        deps.append( tmp )

        tmp = np.zeros([3,2], dtype=np.ndarray)         # dt = 0
        tmp[0,0] =  np.zeros([1,1],dtype=bool)  # X|X
        tmp[1,0] =  np.zeros([1,1],dtype=bool)  # X|Y
        tmp[2,0] =  np.zeros([1,1],dtype=bool)  # X|U
        tmp[0,1] =  np.ones([ 1,1],dtype=bool)  # Y|X 
        tmp[1,1] =  np.zeros([1,1],dtype=bool)  # Y|Y
        tmp[2,1] =  np.zeros([1,1],dtype=bool)  # Y|U
        deps.append( tmp )
        del tmp

        linkFunctions[0] = ['linear']            # x = f(fa_x)                    
        linkFunctions[1] = ['linear']            # y = f(fa_y)

        noiseDistrs[0] = ['Gaussian']            # noise on x
        noiseDistrs[1] = ['Gaussian']            # noise on y
        
        noiseIntActTypes[0] = [ '+' ] # noise is additive
        noiseIntActTypes[1] = [ '+' ] # noise is additive
        
        initDistrs.append([[],[]]) # variable initializations for  t = 1
        initDistrs[0][0] = ['Gaussian']  # initDistrs[dt][X] = list(...)
        initDistrs[0][1] = [  None    ]  # initDistrs[dt][Y] = list(...)
        
        updateParsList = [A,Q,mu0,V0,C,R]
        
        
    elif ssmType == 'inputARLDS':
                       
        if dims.size != 3:
            print('dims:')
            print(dims)
            raise Exception(('input auto-regressive LDS has three groups of '
                             'variables, X, Y and U, and '
                             'dims hence needs to be of size three'))
        [xDim,yDim,uDim] = checkInDims(dims)
        
        if pars is None:
            A    = np.identity(xDim)    # x_t = Ax_t-1 + Bu_t 
            B    = np.zeros([xDim,uDim])  
            Q    = np.identity(xDim)
            mu0  = np.zeros(xDim)       # x_1 ~ mu0 + N(0, V0)
            V0   = np.identity(xDim)
            C = np.ones( [yDim,xDim])   # y_t = Ey_t-1 + Cx_t + Du_t
            D = np.zeros([yDim,uDim])
            E = np.identity(yDim)
            R = np.identity(yDim)
            nu0  = np.zeros(yDim)
            W0   = np.identity(yDim)
        elif isinstance(pars, (list,np.ndarray)):
            if   isinstance(pars,np.ndarray):
                if np.size(pars)!=11:
                    print('number of parameters handed over:')
                    print(np.size(pars))
                    raise Exception(('input auto-regressive LDS is defined by '
                                     'exactly 11 parameters: '
                                     '{A,B,Q,mu0,V0,C,D,E,R,nu0,W0}. Mismatch!'))
            elif isinstance(pars,list):
                if len(pars)!=11:
                    print('number of parameters handed over:')
                    print(len(pars))
                    raise Exception(('input auto-regressive LDS is defined by '
                                     'exactly 11 parameters: '
                                     '{A,B,Q,mu0,V0,C,D,E,R,nu0,W0}. Mismatch!'))
            A    = pars[0]
            B    = pars[1]
            Q    = pars[2]
            mu0  = pars[3]
            V0   = pars[4]
            C    = pars[5]
            D    = pars[6]
            E    = pars[7]
            R    = pars[8]
            nu0  = pars[9]
            W0   = pars[10]
        else:
            raise Exception(('argument pars has to be list or ndarray '
                             'containing the parameters defining an input '
                             'auto-regressive LDS. It is neither.'))
                                   
        xGr = np.zeros(dims[0],dtype=int)
        yGr = np.zeros(dims[1],dtype=int)
        uGr = np.zeros(dims[2],dtype=int)

        dts = toArray([-1,0])
            
        tmp = np.zeros([3,2], dtype=np.ndarray)         # dt = -1
        tmp[0,0] =  np.ones( [1,1],dtype=bool) # X|X   read e.g.   | x1 | x2 
        tmp[1,0] =  np.zeros([1,1],dtype=bool) # X|Y   X|Y as      |--------
        tmp[2,0] =  np.zeros([1,1],dtype=bool) # X|U            y1 | o  |
        tmp[0,1] =  np.zeros([1,1],dtype=bool) # Y|X            y2 |    | o
        tmp[1,1] =  np.ones( [1,1],dtype=bool) # Y|Y
        tmp[2,1] =  np.zeros([1,1],dtype=bool) # Y|U
        deps.append( tmp )

        tmp = np.zeros([3,2], dtype=np.ndarray)         # dt = 0
        tmp[0,0] =  np.zeros([1,1],dtype=bool) # X|X
        tmp[1,0] =  np.zeros([1,1],dtype=bool) # X|Y
        tmp[2,0] =  np.ones( [1,1],dtype=bool) # X|U
        tmp[0,1] =  np.ones( [1,1],dtype=bool) # Y|X 
        tmp[1,1] =  np.zeros([1,1],dtype=bool) # Y|Y
        tmp[2,1] =  np.ones( [1,1],dtype=bool) # Y|U
        deps.append( tmp )
        del tmp

        linkFunctions[0] = ['linearTwoInputs']            # x = f(fa_x)                    
        linkFunctions[1] = ['linearThreeInputs']          # y = f(fa_y)

        noiseDistrs[0] = ['Gaussian']            # noise on x
        noiseDistrs[1] = ['Gaussian']            # noise on y
        
        noiseIntActTypes[0] = [ '+' ] # noise is additive
        noiseIntActTypes[1] = [ '+' ] # noise is additive
        
        initDistrs.append([[],[]]) # variable initializations for t = 1
        initDistrs[0][0] = ['Gaussian']  # initDistrs[dt][X] = list(...)
        initDistrs[0][1] = ['Gaussian']  # initDistrs[dt][Y] = list(...)
        
        updateParsList = [A,B,Q,mu0,V0,C,D,E,R,nu0,W0]
            
    """ done with model selection, now initiate """
    if seq is None:
        seq = timeSeries() # initiate from scratch
        
        seq.giveEmpirical().setModel( 'stateSpace',      # modelClass
                                 ssmType,                # modelDescr
                                 xGr,                    
                                 yGr,                   
                                 uGr,                   
                                 dts,                    
                                 deps,                 
                                 linkFunctions,          
                                 noiseDistrs,            
                                 noiseIntActTypes,       
                                 initDistrs,             
                                 True)                   # isHomogeneous      

        # get pointers to object lists
        model = seq.giveEmpirical().giveModel()

    else: 
        seq.analysis1 = timeSeriesObject('analysis', seq)  # add new model
        
        seq.analysis1.setModel( 'stateSpace',            # modelClass
                                 ssmType,                # modelDescr
                                 xGr,                    
                                 yGr,                   
                                 uGr,                   
                                 dts,                    
                                 deps,                 
                                 linkFunctions,          
                                 noiseDistrs,            
                                 noiseIntActTypes,       
                                 initDistrs,             
                                 True)                   # isHomogeneous      
        
        # get pointers to object lists
        model = seq.analysis1.giveModel()

    
    """ copy desired parameters for link functions and noise distributions """                
    modelDescr = ('stateSpace_' + ssmType)
    [linkPars,noisePars,initPars]=model.parsToUpdateParsList(updateParsList,
                                                             modelDescr)
    model.updatePars(linkPars, noisePars, initPars)            
    
    noiseDists = model.giveFactorization().giveNoiseDistrList()        
    for i in range(np.amax(xGr)+1):
        if not noiseIntActTypes[0][i] is None:
            noiseDists[0][i].setNoiseInteraction(noiseIntActTypes[0][i])
    for i in range(np.amax(yGr)+1):
        if not noiseIntActTypes[1][i] is None:
            noiseDists[1][i].setNoiseInteraction(noiseIntActTypes[1][i])
    

    return seq


def checkInDims(dims):
    
    if ((isinstance(dims[0], list      ) and len(dims[0]) == 1) or
        (isinstance(dims[0], np.ndarray) and dims[0].size == 1)):
        xDim = dims[0][0]
    elif isinstance(dims[0], numbers.Integral):
        xDim = dims[0]
    else: 
        print('dims[0]:')
        print(dims[0])
        raise Exception(('first element of dims has to be int, list or '
                         'ndarray giving the dimenionality of X'))
        
    if ((isinstance(dims[1], list      ) and len(dims[1]) == 1) or
        (isinstance(dims[1], np.ndarray) and dims[1].size == 1)):
        yDim = dims[1][0]
    elif isinstance(dims[1], numbers.Integral):
        yDim = dims[1]
    else: 
        print('dims[1]:')
        print(dims[1])
        raise Exception(('second element of dims has to be int, list or '
                         'ndarray giving the dimenionality of Y'))
        
    if ((isinstance(dims,list)       and len(dims)>2) or
        (isinstance(dims,np.ndarray) and dims.size>2) ):
        
        if ((isinstance(dims[2], list      ) and len(dims[2]) == 1) or
            (isinstance(dims[2], np.ndarray) and dims[2].size == 1)):
            uDim = dims[2][0]
        elif isinstance(dims[2], numbers.Integral):
            uDim = dims[2]
        else: 
            print('dims[2]:')
            print(dims[2])
            raise Exception(('third element of dims has to be int, list or '
                             'ndarray giving the dimenionality of U'))       
        return [xDim,yDim,uDim]
    else:
        return [xDim,yDim]
    

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

""" this function is mostly convenience stuff and needs some work """
def rotatePars(pars, W=None, sortIdx=None):
    """OUT = rotatePars(pars*, W, sortIdx)
    pars:    list or np.ndarray of (multivariate) parameters to rotate and flip
    W:       base change matrix (has to be invertible)
    sortIdx: permutation index for individual components of the parameters
    OUT: x converted to ndarray 
    function that serves to rotate the parameters of an LDS to reflect a change 
    of base in the latent space. This change of base is desirable to compare 
    the results of an LDS fitting algorithm to ground truth latent states and
    parameters (if available). 
    """    
    pars = pars.copy() # never rotate true pars (remember most giveSomething() 
                       # methods in this code package directly return pointers
                       # to data, parameters etc., instead of making copies).
                       # if results shall apply to the parameters stored in a 
                       # timeSeriesModel object, use updatePars()    
    if not W is None:
        Winv = np.linalg.inv(W)
        Wtr  = W.transpose()
        yDim = pars[5].shape[0]
        pars[0] = np.dot( np.dot( W, pars[0] ), Winv )   #  A'  = WAW^-1
        pars[1] = np.dot( np.dot( W, pars[1] ), Wtr )    #  Q'  = WQW^T
        pars[2] = np.dot( W, pars[2] )                   # mu0' = Wmu0
        pars[3] = np.dot( np.dot( W, pars[3] ), Wtr )    # V0'  = WV0W^T
        pars[4] = np.dot( pars[4], Winv )                #  C'  = CW^-1 
        # pars[5] = pars[5]                                 R'  = R
    
    if not sortIdx is None:
        sortIdx2d = np.ix_(sortIdx, sortIdx)
        pars[0] = pars[0][sortIdx2d]
        pars[1] = pars[1][sortIdx2d]
        pars[2] = pars[2][sortIdx]
        pars[3] = pars[3][sortIdx2d]    
        pars[4] = pars[4][:,sortIdx]
        # pars[5] = pars[5]     
    return pars


#----this -------is ------the -------79 -----char ----compa rison---- ------bar

def toArray(x):
    """OUT = toArray(x*)
    x:   integer, float, list or ndarray
    OUT: x converted to ndarray 
    Convenience function that automatically converts integers and lists into
    corresponding numpy ndarrays. Does not change input that already is ndarray
    """
    if isinstance(x, (numbers.Integral,float)):
        x = np.array([x])
    elif isinstance(x, list):
        x = np.array(x)
    elif not isinstance(x, np.ndarray):
        raise Exception(('method "toArray" only converts from int, float, '
                         'list or ndarray'))
    return x


#----this -------is ------the -------79 -----char ----compa rison---- ------bar

class timeSeries:
    """time series structure for use in the context of state-space models
    
    Assumes the time series to consist of the complete-data series {X, Y} and 
    additionally observed inputs U. 
    Since data models e.g. used for generating toy data and analysis models
    applied to such data may in general differ, all that is hard-imposed onto 
    objects of this class are
    the distinction between observed input u, output y and latent state x, and 
    the disinction between temporal and spatial (i.e. u, y, x) dimensions.

    A state-space model is defined by the structure and functional form of
    the dependencies between U, Y and X. 
    The observed output data Y is usually deemed a noisy function of U and X, 
    but potentially also of past instantiations of Y.
    The latents X are usually functions of past X, and potentially also of U.
    Input U is typically assumed fixed, but may be integrated out via a prior. 
    Each of the variable groups has its own noise added/multiplied to it.

    To allow complex models with e.g. latent state hierachy (X ~ Poiss(exp(Z)),
    the variable groups Y, X and U can be further divided into subgroups.
    
    Constructor is called without arguments - objects of this class mostly
    serve as containers for other objects and as scaffold for object 
    communication
    
    VARIABLES and METHODS: 
    giveEmpirical()    - returns the timeSeriesObject 'empirical' that stores
                         the data model and actual data
    """
    
    objDescr = 'state_space_model_series'  

    # always construct a data structure for each time series object
    def __init__(self):                               

        # every time series needs some 'empirical' data 
        self._addEmpirical()
        
    def _addEmpirical(cls):
        cls._empirical = timeSeriesObject('empirical', cls)
        
    def giveEmpirical(cls):
        """ OUT = .giveEmpirical()
        OUT: returns timeSeriesObject 'empirical' containing training data
        
        """
        return cls._empirical
                

#----this -------is ------the -------79 -----char ----compa rison---- ------bar

class timeSeriesObject:
    """storage variable for time series in the context of state-space models

    Assumes the time series to consist of the complete-data series {X, Y} and 
    additionally observed inputs U. Not all three of these groups of variables
    need to be contained in a timeSeriesData object, however. 
    To not include e.g. U, simply set uDim = 0. 
    
    The class of state-space model from which the data is generated is set by
    the probability distributions connecting U, X and Y. The randomness of 
    these distributions may already be seeded when allocating the memory
    storage as is done when creating an object of this class. 
    Basic state-space models such as Bernoulli-observation HMMs or LDS are
    easily seeded with standard built-in functions such as np.random.normal()
    and np.random.rand(). 
    For more complex models with e.g. latent state hierachy (X ~ Poiss(exp(Z))
    the distinction between subgroups of Y, X and U has to be considered,
    and seeding has to be adjusted potentially also with some non-built-in
    RNG functions.
    
    INPUT to constructor: 
    tag:       string for tagging 'empirical' vs. 'analysis' objects 
    seqObject: pointer to object that called the constructor for this object.
               Possibly == None, but usually a timeSeries() object 
    
    VARIABLES and METHODS: 
    .objectDescr   - always equals 'state_space_model_object' 
    .tag           - tags 'empirical' and 'analysis' timeSeriesObject()s
    .supportedTags - gives supported tags
    .setModel()    - sets the model specifications (type, variables, parameters)
    .fitModel()    - fits the model to data stored in the 'empirical' object
    .resetData()   - sets curent 'empirical' to an empty timeSeriesData object
    .addData()     - adds specified amount of data drawn from the current model
    .loadData()    - loads specified data into the data storage object
    .giveData()    - returns pointer to the attached timeSeriesData object
    .giveModel()   - returns pointer to the attached timeSeriesModel object
    .giveMotherObject() - returns pointer provided at object instantiation, 
                         usually the superordinate timeSeries object
    _initModel()  - initiates a timeSeriesModel() object

    """
    objDescr = 'state_space_model_object'      
    
    supportedTags = ['analysis', 'empirical']
    
    def __init__(self, 
                 tag='analysis',
                 seqObject=None):
        
        if tag in self.supportedTags:
            self.tag   = tag
        else:
            print('tag for timeSeriesObject:')
            print(tag)
            print('supported values: ')
            print(self.supportedTags)
            raise Exception('chosen timeSeriesObject tag type not supported.')
            
        self._seqObject = seqObject # if provided, store 'mother' object
        
        self._model = self._initModel('unknown')
        self._data = timeSeriesData(self)  

    def _initModel(cls, modelDescr='unknown'):
        cls._model = timeSeriesModel().defaultModel(modelDescr)
        
    def setModel( cls,
                  modelClass='unknown',
                  modelDescr='custom',
                  xGr=0, yGr=0, uGr=0, 
                  dts=None, deps=None,
                  linkFunctions=None,
                  noiseDistrs=None,
                  noiseInteractions=None,
                  initDistrs=None,
                  isHomogeneous=True ):     
        """ .setModel(modelClass, modelDescr, xGr, yGr, uGr, dts, deps,
                      linkFunctions,noiseDistrs,noiseInteractions,initDistrs,
                      isHomogeneous)
        modelClass:   string specifying the model class, e.g.'stateSpace'
        modelDescr:   string specifying the model type, e.g. 'LDS'
        xGr:          np.ndarray of size xDim that specifies the assigned 
                      subgroup of each latent variable. Implicitly defines 
                      xDim with xGr.size!
        yGr:          np.ndarray of size yDim that specifies the assigned 
                      subgroup of each observed variable. Defines yDim! 
        uGr:          np.ndarray of size uDim that specifies the assigned 
                      subgroup of each input variable. Defines uDim!
        dts:          np.ndarray that gives the relative time steps that 
                      are relevant for evaluating the variables of the 
                      current step. E.g. for a classic LDS, dt = -1 
                      (dependency of x on x) and dt = 0 (dep. of y on x)
                      are relevant and dts = np.array([-1,0]) in this case
        deps:         list of np.ndarrays. Length of list has to match size
                      of dts. Each entry of the list is a 2D binary array
                      specifying which subgroups of X, Y depend on which
                      other subgroups of X,Y and U. 
        linkFunctions:     list of lists of strings specifying the types and
                           parameters of the deterministic parts of the condi-
                           tional distr.s linking subgroups of X, Y and U
        noiseDistrs:       list of lists of strings specifying the types and
                           parameters of the random parts of the conditional
                           distributionss linking subgroups of X, Y and U
        noiseInteractions: list of lists of strings specifying how the random
                           and deterministic parts of the conditional distri-
                           butions interact. E.g. '+', 'multiplicative', '^'
        initDistrs:        list of lists of lists of strings specifying the
                           probability distributions of the intial time steps.
                           Length of outermost list has to match dts.size 
        isHomogeneous: boolean that specifies whether the model is 
                       homogeneous over time                      
        Sets a pre-stored (but potentially not yet configurated) model to 
        specific model type, variable description and parameter set.
        
        """        
        if deps is None:
            print(('conditional dependency variables "deps" not provided. '
                   'Assuming no dependencies (fully independent model!)'))
            
        cls._model = timeSeriesModel(cls,     # hand over pointer to self
                                     modelClass,
                                     modelDescr,
                                     xGr, yGr, uGr,
                                     dts, 
                                     deps,
                                     linkFunctions,
                                     noiseDistrs,
                                     noiseInteractions, 
                                     initDistrs,
                                     isHomogeneous)        

    def fitModel(cls,
                 maxIter=1000, 
                 epsilon=np.log(1.05), # stop if likelihood change < 5%
                 initPars=None,
                 ifPlotProgress=False,
                 experiment=0, 
                 trials=None, 
                 times=None): 
        """ .fitModel(maxIter,epsilon,initPars, 
                      ifPlotProgress,experiment,trials,times)
        maxIter:  maximum allowed iterations for iterative fitting (e.g. EM)
        epsilon:  convergence criterion, e.g. difference of log-likelihoods
        initPars: set of parameters to start fitting. If == None, the 
                  parameters currently stored in the model will be used 
        ifPlotProgress: boolean, specifying if fitting progress is visualized
        experiment: integer index for stored experiment
        trials:  ndarray of indices for stored trials in selected experiment
        times:   ndarray of indices for time steps in selected experiment
        Fits the stored model to data stored in the 'empirical' object.
        """
        LLs = cls._model.fit(maxIter,epsilon,initPars,ifPlotProgress,
                             experiment,trials,times)
        return LLs

    def resetData(cls):
        """ .resetData()
        Resets the timeSeriesData object .data to zero stored experiments') 
        
        """
        try:
            cls._model
        except:
            raise Exception(('cannot generate artificial data before ' 
                             'specifying a model'))
            
        cls._data = timeSeriesData(cls)  
        
    def loadData(cls, y):
        """ .loadData(y*)
        y: activity of variables stored in an ndarray of shape 
           [xyuDim,T,#Trials], or a list of multiple such ndarrays. 
        'Loads' data into timeSeriesData() object. Primarily intended to get 
        real data (interpreted as observed variables Y) into the framework,
        then preferentially with a corresponding timeSeriesModel() object of
        dedicated model class 'empirical'. Can also be used to load complete 
        data sets (all X,Y,U) for other model classes.
        Input arrays will be COPIED before adding them to the data object.

        """
        try: 
            cls._data
        except:
            cls._resetData()
        
        cls._data.loadExperiment(y)        
        
    def addData(cls,
                numTrials=np.array([1]),
                Ts=np.array([42]),                 
                RNGseed=42):
        """ .addData(numTrials,Ts,RNGseed)
        numTrials: ndarray of number of trials to add for new experiment(s)
        Ts:        ndarray of trial lengths for new experiment(s) 
        RNGseed:   integer specifying desired random seed used for new data
        Adds data of one or several new experiments to the timeSeriesData
        object attached to this timeSeriesObject. Not specifying the random
        seed will always intialize it to the same value! Use an RNG and 
        provide it as third argument if random outcomes are wanted.
        
        """        
        try:
            cls._data
        except:
            cls.resetData()
            
        cls._data.addExperiment(numTrials,Ts,RNGseed)
                    
    def giveModel(cls):
        """ OUT = .giveModel()
        OUT: timeSeriesModel object "model" specifying the time series model

        """        
        try:
            return cls._model
        except:
            raise Exception(('time series model object apparently '
                             'not (yet) initialized'))
            
    def giveData(cls):
        """ OUT = .giveData()
        OUT: timeSeriesData object "data" containing the time series data        
        
        """                
        try:
            return cls._data
        except:
            raise Exception(('time series data object apparently '
                             'not (yet) initialized'))            

    def giveMotherObject(cls):
        """OUT = .returnMotherObject()
        OUT: object that called the constructor for this object (possibly None)
        
        """        
        return cls._seqObject            
                            
#----this -------is ------the -------79 -----char ----compa rison---- ------bar

class timeSeriesData:
    """storage variable for time series in the context of state-space models

    Assumes the time series to consist of the complete-data series {X, Y} and 
    additionally observed inputs U. Not all three of these groups of variables
    have to be contained in a timeSeriesData object, however. 
    To not include e.g. U, simply set uDim = 0. 
    
    The class of state-space model from which the data is generated is set by
    the probability distributions connecting U, X and Y. The randomness of 
    these distributions may already be seeded when allocating the memory
    storage as is done when creating an object of this class. 
    Basic state-space models such as Bernoulli-observation HMMs or LDS are
    easily seeded with standard built-in functions such as np.random.normal()
    and np.random.rand(). 
    For more complex models with e.g. latent state hierachy (X ~ Poiss(exp(Z))
    the distinction between subgroups of Y, X and U has to be considered,
    and seeding has to be adjusted potentially also with some non-built-in
    RNG functions.
    The most important variable of timeSeriesData() objects is of course the
    data array. Internally, the data for all time points, all trials and all 
    variables X, Y and U within one experiment are stored within one and the 
    same np.ndarray. Translation between variable group/subgroup indexing 
    and ndarray indexing happens automatically upon calling any relevant 
    method that deals with the stored data. Data arrays for different 
    experiments are listed for quick access and reference. 
    An experiment is defined as a collection of trials with the same length.
    Note that different-trial-length 'experiments' are also supported through
    a number of single-trial experiments. 
    
    INPUT to constructor:
    tsobject:             pointer to object that called the constructor. 
                          Possibly none, but usually a timeSeriesObject()

    VARIABLES and METHODS:
    .objectDescr          always equals 'state_space_model_data' 
    .loadExperiment()     primary method of loading data into this data store
    .addExperiment()      primary method to generate data for this data store
    .giveNumExperiments() returns number of experiments
    .giveNumTrials()      returns number of trials for selected experiments
    .giveTrialLengths()   returns length of trials in selected experiments
    .giveTracesU()        returns (slices of) data for input variables U
    .giveTracesX()        returns (slices of) data for latent variables X
    .giveTracesY()        returns (slices of) data for output variables Y
    .giveTraces()         returns (slices of) data for full groups X,Y,U
    .giveMotherObject()   returns object one step above in object hierarchy 
    _seediidNoise()       allocates memory for the time series of a particular 
                          (group of) variables, and (optionally) seeds noise
    _sampleModelGivenNoiseSeed() samples from given model given noise-seeded
                                 np.ndarrays for data storage
    _condsToIdx()         translates from lists of conditional dependencies
                          into lists of indexes for quick reference within
                          data storages
    """
    objDescr = 'state_space_model_data'      
    
    def __init__(self, 
                 tsobject=None): # timeSeriesObject  
        
        self._tsobject = tsobject # communication with e.g. "timeSeriesModel"
                                  # objects is organized via "timeSeriesObject"

        self._xyu  = [] # list of real full data {X, Y, U} for each experiment
        self._yObs = [] # observed data Y as seen under some (possibly only
                        # partial) observation scheme
        self._obsScheme = [] # observation scheme for each experiment
        
        self._offsets = np.zeros(4,dtype=int)
        
        self._Ts        = np.zeros(0) # initialize to empty
        self._numTrials = np.zeros(0) # sets resp. 0, then
        self._numExperiments = self._numTrials.size   # add data 
        
        
    def loadExperiment(cls,y,xyu='xyu',experiment=None):
        """ .loadExperiment(y*)
        y: activity of variables stored in an ndarray of shape 
           [xyuDim,T,#Trials], or a list of multiple such ndarrays
        xyu:     string that defines for which of the variable groups X,Y,U
                 data traces are to be returned. Possible values are e.g.  
                 'xyu','xu','y' and other ordered substrings of 'xyu'           
        experiment: integer index for stored experiment. Use the default 
                    'None' to add a completely new experiment.
        'Loads' data into timeSeriesData() object. Primarily intended to get 
        real data (interpreted as observed variables Y) into the framework,
        then preferentially with a corresponding timeSeriesModel() object of
        dedicated model class 'empirical'. Can also be used to load complete 
        data sets (all X,Y,U) for other model classes.
        Input arrays will be COPIED before adding them to the data object.

        """
        if (not experiment is None  # convention: add new experiments
              and not isinstance(experiment, (list, numbers.Integral))):
            raise Exception(('argument experiment has to be None, int or a '
                             'list of ints. It it neither.'))
            
        # try to get acces to .model object:
        try:
            model = cls._tsobject.giveModel()
        except:
            raise Exception(('model description for data model is '
                             'required, but apparently not yet initialized'))
        
        if isinstance(y, list):             # handing over list of experiments
            i = 0
            if not experiment is None and not isinstance(experiment, list):
                raise Exception(('if argument y is a list, argument experiment'
                                 ' also has to be a list. It is not'))
            elif not experiment is None and len(y) != len(experiment):
                print('len(y)')
                print(len(y))
                print('len(experiment)')
                print(len(experiment))
                raise Exception(('lengths of arguments y and experiment have '
                                 'to match. They do not.'))
            xyuDim = y[0].shape[0]
            while i < len(y):
                if not isinstance(y[i],np.ndarray):
                    print(('Error when trying to load several experiments at '
                           'once.'))
                    print('experiment #i, i = ')
                    print(i)
                    raise Exception(('All data sets have to be ndarrays. The '
                                     'i-th data set is not.'))
                elif y[i].shape[0] != xyuDim:
                    print(('Error when trying to load several experiments at '
                           'once.'))
                    print('experiment #i, i = ')
                    print(i)
                    print('data dimensionality of first experiment:')
                    print(xyuDim)
                    print('data dimensionality of i-th experiment:')
                    print(y[i].shape[0])
                    raise Exception(('All data sets must have the same '
                                     'data dimensionality.'))
                elif y[i].size == 0: # e.g. if xyu='u' in a no-input model
                    raise Exception(('trying to load empty data array into '
                                     'timeSeriesData object.'))
                i +=1                
        elif not isinstance(y, np.ndarray): # handing over single experiment
            raise Exception('data has to be of dtype ndarray. It is not.')
        elif y.size == 0:
            raise Exception(('trying to load empty data array into '
                             'timeSeriesData object.'))  
        elif (not (experiment is None or isinstance(experiment, list)) 
               and (experiment < 0 or experiment >= cls._numExperiments)):
            print('experiment:')
            print( experiment  )
            raise Exception('argument experiment has to be non-negative int')
        else:
            xyuDim = y.shape[0]
            
        if model._modelClass == 'empirical': # default case for loading data 
            # 'empirical' models should only know observed variables Y, 
            # and neither X nor U.
            if cls._offsets[3] == 0: # check if offsets initialized 
                cls._offsets[2] += xyuDim                    # for finding 
                cls._offsets[3]  = cls._offsets[2]           # the data
            elif not np.all(cls._offsets[3] == np.array([0,0,xyuDim,xyuDim])):
                print('Data dimensionality error:')
                print('stored variable group index offsets:')
                print(cls._offsets)
                print('new total data dimensionality:')
                print(xyuDim)
                raise Exception('Dimensionality mismatch!')   
            # else: we're fine, i.e. offsets are known and match new data                          
        else: # if not 'empirical' model class, i.e. we actually have a model                           
            try: 
                model = cls._tsobject.giveModel()
            except:
                raise Exception(('model description for data model is '
                                 'required, but apparently not yet '
                                 'initialized'))                                        
            try: 
                varDescr = model.giveVarDescr()
            except:
                raise Exception(('variable description for data model is '
                                 'required, but apparently not yet '
                                 'initialized'))  
            xyuVarDims = varDescr.giveVarDims(xyu) 
            if xyuDim != np.sum(xyuVarDims):
                print('model description:')
                print(model.giveModelDescription())
                print('number of dimensions of provided data:')
                print(xyuDim)
                print('selected variable groups (within X,Y,U)')
                print(xyu)
                print('number of selected variables of current model:')
                print(np.sum(xyuVarDims))
                raise Exception(('dimensionality of provided data does not '
                                 'match dimensionality of model parameters.'))
                                            
            cls._offsets[0] = 0
            cls._offsets[1] = cls._offsets[0] + varDescr.giveVarDims('x')
            cls._offsets[2] = cls._offsets[1] + varDescr.giveVarDims('y')
            cls._offsets[3] = cls._offsets[2] + varDescr.giveVarDims('u')
        try:
            idxDims   = { 'xyu' : np.arange(cls._offsets[0], cls._offsets[3]),
                          'xy'  : np.arange(cls._offsets[0], cls._offsets[2]),
                          'xu'  : np.concatenate(
                                     [np.arange(cls._offsets[0], cls._offsets[1]),
                                      np.arange(cls._offsets[2], cls._offsets[3])]
                                                 ),
                          'yu'  : np.arange(cls._offsets[1], cls._offsets[3]),
                           'x'  : np.arange(cls._offsets[0], cls._offsets[1]),
                           'y'  : np.arange(cls._offsets[1], cls._offsets[2]),
                           'u'  : np.arange(cls._offsets[2], cls._offsets[3])
                             }[xyu]
        except:
            raise Exception(('argument "xyu" has to be an ordered subset '
                           'of x, y and u, e.g. "yu", "x", "xyu"'))        
        
        if isinstance(y, list):
            Ts        = np.zeros(len(y))
            numTrials = np.zeros(len(y))
            i = 0
            while i < len(y):
                Ts[i]        = y[i].shape[1]
                numTrials[i] = y[i].shape[2]
                if experiment is None:
                    cls._xyu.append(np.zeros([cls._offsets[3],
                                              Ts[i],
                                              numTrials[i]]))
                    tmp = cls._numExperiments+i  # note this gives +1 to index  
                    cls._yObs.append([]) # make space for possible observed Y
                    cls._obsScheme.append([])
                else:
                    tmp = experiment[i]
                cls._xyu[tmp][idxDims,:,:] = y[i].copy()
                i += 1
        else: # checked before, has to be ndarray otherwise: 
            Ts        = y.shape[1]
            numTrials = y.shape[2]
            if experiment is None:
                tmp = cls._numExperiments # note this gives +1 to index
                cls._xyu.append(np.zeros([cls._offsets[3], Ts, numTrials]))       
                cls._yObs.append([]) # make space for possible observed Y
                cls._obsScheme.append([])                
            cls._xyu[tmp][idxDims,:,:] = y.copy()
                      
        # update data inventory
        if experiment is None: # only need to update if adding new experiments
            cls._Ts        = np.hstack([cls._Ts.copy(), Ts])
            cls._numTrials = np.hstack([cls._numTrials.copy(), numTrials])
            cls._numExperiments = cls._numTrials.size         

        
    def addExperiment(cls,
                      numTrials=np.array([1]),
                      Ts=np.array([42]), 
                      RNGseed=42):
        """ .addExperiment(Ts,numTrials,RNGseed)
        timeSeriesObject: object that holds timeSeriesModel object
        Ts:        numExperiments-by-1 ndarray of trial lengths
        numTrials: numExperiments-by-1 ndarray of numbers of trials
        RNGseed:   integer that gives a specific random seed
        Adds data of one or several new experiments to the timeSeriesData
        object attached to this timeSeriesObject. Not specifying the random
        seed will always intialize it to the same value! Use an RNG and 
        provide it as third argument if random outcomes are wanted.
        
        """        
        # try to get acces to .model object:
        try:
            model = cls._tsobject.giveModel()
        except:
            raise Exception(('model description for data model is '
                             'required, but apparently not yet initialized'))  
    
        # try to get acces to .model.varDescr object:
        try:
            varDescr = model.giveVarDescr()
        except:
            raise Exception(('variable description for data model is '
                             'required, but apparently not yet initialized'))  

        # try to get acces to .model.factorization object:
        try:
            factorization = model.giveFactorization()
        except:
            raise Exception(('model factorization for data model is '
                             'required, but apparently not yet initialized'))                         
        
        
        np.random.seed(RNGseed)                            
                            
        if isinstance(numTrials , numbers.Integral):  
            numTrials = np.array([numTrials],dtype=int)   
        if isinstance(numTrials, list ):           
            numTrials = np.array(numTrials,dtype=int)                           
        if not isinstance(numTrials, np.ndarray): # if not true by now ...
            raise TypeError(('variable Trials may either be a an integer, '
                             ' a list or an ndarray'))                            
        numTrials.reshape([1,numTrials.size])
                            
        if isinstance( Ts, numbers.Integral ):               
            Ts = Ts * np.ones(numTrials.shape, # make sure Ts is a vector
                              dtype=int)       # of dim. Trials.size-by-1 
        if isinstance( Ts, list ):             # and that it is of type 
            Ts = np.array(Ts,dtype=int)        # 'ndarray'            
        if not isinstance(Ts, np.ndarray): # if not true by now ...
            raise TypeError(('variable Ts may either be an integer, a list '
                             'or an ndarray'))                            
        if Ts.size != numTrials.size:
            print('dimensionality of Ts:')
            print(Ts.shape)
            print('dimensionality of Trials:')
            print(Trials.shape)                            
            raise Exception(('vector of individual trial lengths does '
                             'not match the vector of number of trials'))
        Ts.reshape([1,Ts.size])

        cls._Ts        = np.hstack([cls._Ts.copy(), Ts])
        cls._numTrials = np.hstack([cls._numTrials.copy(), numTrials])
        cls._numExperiments = cls._numTrials.size         
        
        # if noise at time t is independent of the rest of the time series, 
        # we may initialize the series already with noise!       
                                                   
        xyuSubgroupTallies = varDescr.giveVarSubgroupTallies() 
        noiseDistrList     = factorization.giveNoiseDistrList()
        
        # create index offsets for quick access to X,Y,U within self._xyu
        cls._offsets[0] = 0
        cls._offsets[1] = cls._offsets[0] + varDescr.giveVarDims('x')
        cls._offsets[2] = cls._offsets[1] + varDescr.giveVarDims('y')
        cls._offsets[3] = cls._offsets[2] + varDescr.giveVarDims('u')
        
        t=0
        while t < Ts.size:
            
            distrType = [[],[],[]] # assemble full list of distributions
            j = 0
            while j < xyuSubgroupTallies[0]: # subgroups of X                    
                distrType[0].append(noiseDistrList[0][j].giveDistrType())
                j += 1
            j = 0
            while j < xyuSubgroupTallies[1]: # subgroups of Y                    
                distrType[1].append(noiseDistrList[1][j].giveDistrType())
                j += 1
            j = 0
            while j < xyuSubgroupTallies[2]: # subgroups of U                    
                distrType[2].append('none')
                j += 1
            distrType = np.array(distrType)
            
            cls._xyu.append(cls._seediidNoise(
                                       cls._offsets, 
                                       varDescr.giveVarSubgroupIndices(), 
                                       varDescr.giveVarSubgroupSizes(),
                                       numTrials[t], Ts[t],   
                                       distrType ) 
                           ) 
            cls._yObs.append([]) # make space for observed version of Y
            cls._obsScheme.append([])            
            t += 1
           
        # now fill the allocated space with actual data! 
        try:
            tsmodel = cls._tsobject.giveModel() 
        except: 
            raise Exception('trying to sample data from non-specified model')
            
        i=0
        while i < Ts.size:
            timeSeriesData._sampleModelGivenNoiseSeed(cls._xyu[i], tsmodel)
            i += 1
            
    def deleteExperiments(cls, experiment=0):
        """ .deleteExperiment(experiment)
        experiment: (list/array of) index of experiment(s) to manipulate
        Deletes specified experiments from this timeSeriesData object.
        
        """        
        if isinstance(experiment, np.ndarray):
            experiment = experiment.tolist()            
        cls._Ts        = cls._Ts.tolist()
        cls._numTrials = cls._numTrials.tolist()
        if isinstance(experiment, list):
            experiment.sort()
            experiment.reverse()
            if experiment[0] >= cls._numExperiments:
                print('maximum index of experiments:')
                print(experiment[0])
                print('number of currently stored experiments:')
                print(cls._numExperiments)
                raise Exception('cannot delete experiments that do no exist')
                     
            for i in range(len(experiment)):
                cls._xyu.pop(      experiment[i])
                cls._yObs.pop(     experiment[i])
                cls._obsScheme.pop(experiment[i])
                cls._Ts.pop(       experiment[i])
                cls._numTrials.pop(experiment[i])
        elif isinstance(experiment, numbers.Integral) and experiment >= 0:
            cls._xyu.pop(      experiment)
            cls._yObs.pop(     experiment)            
            cls._obsScheme.pop(experiment)
            cls._Ts.pop(       experiment)
            cls._numTrials.pop(experiment)
        else:
            raise Exception(('argument experiment has to be a list, ndarray '
                             'or non-negative int'))
        cls._Ts        = np.array(cls._Ts)
        cls._numTrials = np.array(cls._numTrials)
        cls._numExperiments = cls._numTrials.size         
                
    def setObservationScheme(cls, 
                             experiment,
                             subpops, 
                             obsTimes,
                             obsPops):
        """ .setObservationScheme(experiment,subpops*,obsTimes*,obsPops*)
        experiment: index of experiment to manipulate
        subpops:    list of indices defining the subpopulations
        obsTimes:   array of switch times between subpopulations
        obsPops:    array of populations observed after each switch

        """  
        if not isinstance(experiment, numbers.Integral) or experiment > 0:
            print('experiment:')
            print(experiment)
            raise Exception('argument experiment has to be a positive int.')
            
        try:
            cls._obsScheme[experiment]
        except:
            print('experiment:')
            print( experiment )
            print('number of stored experiments:')
            print( cls._numExperiments )
            raise Exception('experiment out of index of stored experiments')

        if not isinstance(subpops, list):            
            raise Exception('argument subpops has to be a list. It is not')
        if isinstance(obsTimes, list):
            obsTimes = toArray(obsTimes)
        elif not isinstance(obsTimes, np.ndarray):            
            raise Exception('argument obsTimes has to be a list or ndarray. '
                            'It is not')
        if len(subpops) < np.max(obsPops):
            print('number of subpopulations:')
            print(len(subpops))
            print('maximum index of subpopulations given to be observed:')
            print(np.max(obsPops))
            raise Exception(('gave observation time for non-defined '
                             'subpopulation.'))
        
        for i in range(len(subpops)):
            if not isinstance(subpops[i], (list, np.ndarray)):
                print('i:')
                print(i)
                print('subpops[i]')
                print( subpops[i] )
                raise Exception(('arguments subpops[i] have to be lists or '
                                 'ndarrays for all provided i'))                
            
        cls._obsScheme[experiment] = {'subpops':  subpops,
                                      'obsTimes': obsTimes,
                                      'obsPops':  obsPops}
            
    def simulateObservationScheme(cls, 
                                  experiment=0, 
                                  subpops=None,
                                  obsTimes=None,
                                  obsPops=None,
                                  mask=float('NaN')):
        """.simulateObservationScheme(experiment,subpops,obsTimes,obsPops,mask)
        experiment: index of experiment to manipulate
        subpops:    list of indices defining the subpopulations
        obsTimes:   array of switch times between subpopulations
        obsPops:    array of populations observed after each switch
        mask:       default value for unobserved activity data 

        """  
        if not isinstance(experiment, numbers.Integral) or experiment > 0:
            print('experiment:')
            print(experiment)
            raise Exception('argument experiment has to be a positive int.')
        if not (subpops is None or isinstance(subpops, list)):  
            print('subpops:')
            print( subpops  )
            raise Exception('argument subpops has to be a list. It is not')
        if isinstance(obsTimes, list):
            obsTimes = toArray(obsTimes)
        elif not (obsTimes is None or isinstance(obsTimes, np.ndarray)):            
            raise Exception('argument obsTimes has to be a list or ndarray. '
                            'It is not')
        if subpops is None:
            subpops  = cls._obsScheme[experiment]['subpops']
        if obsTimes is None:
            obsTimes = cls._obsScheme[experiment]['obsTimes']
        if obsPops is None:
            obsPops  = cls._obsScheme[experiment]['obsPops']
            
        if len(subpops) < np.max(obsPops):
            print('number of subpopulations:')
            print(len(subpops))
            print('maximum index of subpopulations given to be observed:')
            print(np.max(obsPops))
            raise Exception(('gave observation time for non-defined '
                             'subpopulation.'))
            
        y = cls.giveTracesY(experiment) # all dims, time points and trials!

        yDims = range(y.shape[0])
        for i in range(len(subpops)):
            if isinstance(subpops[i], list):
                jRange = range(len(subpops[i]))
            elif isinstance(subpops[i], np.ndarray):
                jRange = range(subpops[i].size)
            else:
                print('i:')
                print(i)
                print('subpops[i]')
                print( subpops[i] )
                raise Exception(('arguments subpops[i] have to be lists or '
                                 'ndarrays for all provided i'))                
            if np.any([not subpops[i][j] in yDims for j in jRange]):
                print('subpops[i]')
                print( subpops[i] )
                print('number of dimensions of Y:')
                print(y.shape[0])
                raise Exception(('arguments subpops[i] must not contain '
                                 'non-existent oberved variables Y for any '
                                 'i.'))
        unobs = np.array([j not in subpops[obsPops[0]] for j in yDims])
        y[np.ix_(unobs, range(obsTimes[0]))] = mask        
        for i in range(1,obsTimes.size):
            unobs = np.array([j not in subpops[obsPops[i]] for j in yDims])
            y[np.ix_(unobs, range(obsTimes[i-1], obsTimes[i]))] = mask
                                                     
        cls._yObs[experiment] = y
                
                
    @staticmethod                        
    def _seediidNoise(offsets   =np.zeros(4),         # list of integers
                      xyuGrIdx  =np.array([[[0]]]),   # list of index lists
                      xyuGrSize= np.array([[[1]]]),
                      Trial=1, T=42,  
                      noiseDistrs=np.array([['Gaussian']])):
        tr = np.zeros([offsets[3],T,Trial])                            
        i = 0
        while i < len(xyuGrIdx): # i = 0, 1, 2 for X, Y, U, resp.
            j = 0
            while j < len(xyuGrIdx[i]): # going through subgroups
                if noiseDistrs[i][j] != 'none':
                    distrType =  {      # all share the same 'size' argument
                    'Gaussian':    np.random.normal,  
                    'gaussian':    np.random.normal,  
                    'normal':      np.random.normal,  
                    'uniform':     np.random.uniform,   
                    # add more HERE if necessary
                                 }[noiseDistrs[i][j]]   
                    tr[np.ix_(xyuGrIdx[i][j] + offsets[i],
                              np.arange(T),
                              np.arange(Trial))] = \
                         distrType(size=[xyuGrSize[i][j],T,Trial])
                j += 1
            i += 1
            
        return tr
    
    @staticmethod
    def _sampleModelGivenNoiseSeed(xyu_t, tsmodel):
        
        factorization = tsmodel.giveFactorization()
        hierarchy     = factorization.giveSamplingHierarchy()
        linkFunctions = factorization.giveLinkFunctionList()
        noiseDistrs   = factorization.giveNoiseDistrList()
        conds         = factorization.giveConditionalLists()
        initDistrs    = factorization.giveInitialDistrList()
        varDescr      = tsmodel.giveVarDescr()
        
        sgIdx  = varDescr.giveVarSubgroupIndices()
        xyuDim = varDescr.giveVarDims()
        
        dts = varDescr.giveVarTimeScope()
        dtsInit       = factorization.giveInitVarTimeScope()
            
        idxSgInput  = []
        idxSgOutput = []
        for sg in hierarchy:    
            condsLs = conds[sg[0]][sg[1]] # dependencies of curr. subgroup
            idxSgInput.append(timeSeriesData._condsToIdx(condsLs,sgIdx,xyuDim))
            idxSgOutput.append(sum(xyuDim[0:sg[0]]) + sgIdx[sg[0]][sg[1]])
        idxSgInput  = np.array(idxSgInput)
        idxSgOutput = np.array(idxSgOutput)
        lenHierarchy = len(hierarchy)

        t = 0
        while t < len(initDistrs):
            sgCt = 0 # subgroup counter
            while sgCt < lenHierarchy: # go through all subgroups ...
                sg = hierarchy[sgCt] # sg = [varGroup,varSubgroup]
                initDis = initDistrs[t][sg[0]][sg[1]]
                if not initDis is None:
                    tr = 0
                    while tr < xyu_t.shape[2]: # tr = 0,...,Trials-1                    
                        # overwrite seeded noise:
                        xyu_t[idxSgOutput[sgCt],t,tr] = initDis.drawSample(1)
                        tr += 1 
                sgCt += 1
            t += 1            
        t = - min(dts) # e.g. = 2 if dt = [-2,-1,0]
        tMax = xyu_t.shape[1] - max([max(dts),0]) # = T for causal systems
        while t < tMax: # t = 0,...,T-1, less for acausal systems
            sgCt = 0 # subgroup counter
            while sgCt < lenHierarchy: # go through all subgroups ...
                sg = hierarchy[sgCt] # sg = [varGroup,varSubgroup]
                condsLs = conds[sg[0]][sg[1]] # condsLs is list of triplets
                noiseDis  = noiseDistrs[sg[0]][sg[1]]
                noiIntAct = noiseDis.giveNoiseInteraction()
                tr = 0
                while tr < xyu_t.shape[2]: # tr = 0,...,Trials-1
                    
                    # noise is already seeded, now just need to transform it:
                    xyu_t[idxSgOutput[sgCt],t,tr] = \
                       noiseDis._transfNoiseSeed(xyu_t[idxSgOutput[sgCt],t,tr])
                        
                    # now work on deterministic part of update equations: 
                    sgInput = [] # assemble input for deterministic part of  
                    i = 0        # the update for the current variable subgroup
                    while i < len(condsLs): # go through all inputs ...
                        sgInput.append(
                                  xyu_t[idxSgInput[sgCt][i],t+condsLs[i][0],tr]
                                      )
                        i += 1                        
                    xyu_t[idxSgOutput[sgCt],t,tr] = noiIntAct(
                            xyu_t[idxSgOutput[sgCt],t,tr],
                            linkFunctions[sg[0]][sg[1]].computeValue(sgInput)
                                                             )
                    tr += 1
                sgCt += 1
            t += 1
                    
    @staticmethod
    def _condsToIdx(condsList, subgroupIdx, xyuDim):
        i = 0
        xyuIdx = []
        while i < len(condsList):
            currEl = condsList[i]
            xyuIdx.append( (subgroupIdx[currEl[1]][currEl[2]] 
                            + sum(xyuDim[0:currEl[1]])) )
            i+=1
        return np.array(xyuIdx)

    def giveNumExperiments(cls):
        """OUT = .giveNumExperiments()
        OUT: integer number of experiments stored in this timeSeriesData object 
        
        """
        
        return cls._numExperiments
    
    def giveNumTrials(cls, experiment=None):
        """OUT = .giveNumTrials(experiments)
        experiments: ndarray of indices for stored experiments
        OUT: ndarray of number of trials in queried experiments 
        
        """
        if experiment is None:
            experiment = np.arange(cls._numExperiments)
        else:
            experiment = toArray(experiment)
        
        return cls._numTrials[experiment]
    
    def giveTrialLengths(cls, experiment=None):
        """OUT = .giveTrialLengths(experiments)
        experiments: ndarray of indices for stored experiments
        OUT: ndarray of length of trials in queried experiments 
        
        """
        if experiment is None:
            experiment = np.arange(cls._numExperiments)
        else:
            experiment = toArray(experiment)
        
        return cls._Ts[experiment]
            
    
    
    def giveTracesX(cls, experiment=0, trials=None, times=None, dims=None):
        """OUT = .giveTracesX(experiment,trials,times,dims)
        experiment: integer index for stored experiment
        trials:  ndarray of indices for stored trials in selected experiment
        times:   ndarray of indices for time steps in selected experiment
        dims:    ndarray of indices for components of X        
        OUT: ndarray of number of traces of selected components of X 
        Please note: This method returns a COPY of the desired data traces.
        
        """
        if cls._xyu == []:  # if no experiments added yet, _xyu is empty list
            return np.array([]) # and cannot be indexed as below.

        if not isinstance(experiment, numbers.Integral):
            raise Exception(('can only give input traces for individual '
                             'experiments indexed by a single integer'))
        if experiment >= cls._numExperiments:
            print('number of experiments stored: ')
            print(cls._numExperiments)
            raise Exception(('experiment index too high - asking for an '
                             'experiment that was not (yet) added'))        
        if trials is None:
            trials = np.arange(cls._xyu[experiment].shape[2])
            
        if times  is None:
            times  = np.arange(cls._xyu[experiment].shape[1])
            
        if dims   is None:
            dims   = np.arange(0, cls._offsets[1]-cls._offsets[0])
            
        dims   = toArray(dims)            
        trials = toArray(trials)
        times  = toArray(times)
        
        if np.amax(dims) > cls._offsets[1]-cls._offsets[0]:
            print('requested highest dimension of X:')
            print(np.amax(dims))
            print('stored highest dimension of X:')
            print(cls._offsets[1]-cls._offsets[0])
            raise Exception(('requesting dimensions of X beyond those '
                             'that are stored'))
        else: # transform to indices within cls._xyu
            dims += cls._offsets[0]
        try:
            return cls._xyu[experiment][np.ix_(dims,times,trials)].copy()
        except:
            print('Data for X has size = (times,trials) =')
            print(cls._xyu[experiment].shape[1:3])
            raise Exception('Queried data outside of these bounds')


    def giveTracesY(cls, experiment=0, trials=None, times=None, dims=None):
        """OUT = .giveTracesY(experiment,trials,times,dims)
        experiment: integer index for stored experiment
        trials:  ndarray of indices for stored trials in selected experiment
        times:   ndarray of indices for time steps in selected experiment
        dims:    ndarray of indices for components of Y        
        OUT: ndarray of number of traces of selected components of Y 
        Please note: This method returns a COPY of the desired data traces.
        
        """
        if cls._xyu == []:  # if no experiments added yet, _xyu is empty list
            return np.array([]) # and cannot be indexed as below.

        if not isinstance(experiment, numbers.Integral):
            raise Exception(('can only give input traces for individual '
                             'experiments indexed by a single integer'))
        if experiment >= cls._numExperiments:
            print('number of experiments stored: ')
            print(cls._numExperiments)
            raise Exception(('experiment index too high - asking for an '
                             'experiment that was not (yet) added'))        
        if trials is None:
            trials = np.arange(cls._xyu[experiment].shape[2])
            
        if times  is None:
            times  = np.arange(cls._xyu[experiment].shape[1])
            
        if dims   is None:
            dims   = np.arange(0, cls._offsets[2]-cls._offsets[1])
            
        dims   = toArray(dims)            
        trials = toArray(trials)
        times  = toArray(times)
        
        if np.amax(dims) > cls._offsets[2]-cls._offsets[1]:
            print('requested highest dimension of Y:')
            print(np.amax(dims))
            print('stored highest dimension of Y:')
            print(cls._offsets[2]-cls._offsets[1])
            raise Exception(('requesting dimensions of Y beyond those '
                             'that are stored'))
        else: # transform to indices within cls._xyu
            dims += cls._offsets[1]
        try:
            return cls._xyu[experiment][np.ix_(dims,times,trials)].copy()
        except:
            print('Data for Y has size = (times,trials) =')
            print(cls._xyu[experiment].shape[1:3])
            raise Exception('Queried data outside of these bounds')

            
    def giveObservationScheme(cls, experiment=0):
        """ .giveSimulationScheme(experiment)
        experiment: index of experiment to manipulate. Use experiment=None
                    to return a list of all stored observation schemes.
        OUT: dict with keys 'subpops', 'obsTimes', 'obsPops'
        Please note: This method returns a COPY of the desired information.
        
        """  
        if experiment is None:
            return cls._obsScheme
        elif not isinstance(experiment, numbers.Integral) or experiment > 0:
            print('experiment:')
            print(experiment)
            raise Exception('argument experiment has to be a positive int.')            
        try:
            cls._obsScheme[experiment]
        except:
            print('experiment:')
            print( experiment )
            print('number of stored experiments:')
            print( cls._numExperiments )
            raise Exception('experiment out of index of stored experiments')

        return cls._obsScheme[experiment].copy()   
            
            
    def giveTracesYobs(cls, experiment=0,trials=None,times=None,dims=None):
        """OUT = .giveTracesYobs(experiment,trials,times,dims)
        experiment: integer index for stored experiment
        trials:  ndarray of indices for stored trials in selected experiment
        times:   ndarray of indices for time steps in selected experiment
        dims:    ndarray of indices for components of Y        
        OUT: ndarray of number of observed traces of selected components of Y 
        Please note: This method returns a COPY of the desired data traces.
        
        """
        if not isinstance(experiment, numbers.Integral):
            raise Exception(('can only give input traces for individual '
                             'experiments indexed by a single integer'))
        if experiment >= cls._numExperiments:
            print('number of experiments stored: ')
            print(cls._numExperiments)
            raise Exception(('experiment index too high - asking for an '
                             'experiment that was not (yet) added'))     
        if cls._yObs[experiment] == []: # if not added yet, is empty list
            return np.array([])         # and cannot be indexed as below.
            
        if trials is None:
            trials = np.arange(cls._yObs[experiment].shape[2])
            
        if times  is None:
            times  = np.arange(cls._yObs[experiment].shape[1])
            
        if dims   is None:
            dims   = np.arange(cls._yObs[experiment].shape[0])
            
        dims   = toArray(dims)            
        trials = toArray(trials)
        times  = toArray(times)
        
        if np.amax(dims) > cls._yObs[experiment].shape[0]:
            print('requested highest dimension of Y:')
            print(np.amax(dims))
            print('stored highest dimension of Y:')
            print(cls._yObs[experiment].shape[0])
            raise Exception(('requesting dimensions of Y beyond those '
                             'that are stored'))
        try:
            return cls._yObs[experiment][np.ix_(dims,times,trials)].copy()
        except:
            print('Data for Y has size = (times,trials) =')
            print(cls._yObs[experiment].shape[1:3])
            raise Exception('Queried data outside of these bounds')

    
    def giveTracesU(cls, experiment=0, trials=None, times=None, dims=None):
        """OUT = .giveTracesU(experiment,trials,times,dims)
        experiment: integer index for stored experiment
        trials:  ndarray of indices for stored trials in selected experiment
        times:   ndarray of indices for time steps in selected experiment
        dims:    ndarray of indices for components of U        
        OUT: ndarray of number of traces of selected components of U
        Please note: This method returns a COPY of the desired data traces.
        
        """
        if cls._offsets[3] - cls._offsets[2] == 0: # if uDim = 0
            return np.array([]) # and cannot be indexed as below.
        
        if cls._xyu == []:  # if no experiments added yet, _xyu is empty list
            return cls._xyu # and cannot be indexed as below. Return and quit.

        if not isinstance(experiment, numbers.Integral):
            raise Exception(('can only give input traces for individual '
                             'experiments indexed by a single integer'))
        if experiment >= cls._numExperiments:
            print('number of experiments stored: ')
            print(cls._numExperiments)
            raise Exception(('experiment index too high - asking for an '
                             'experiment that was not (yet) added'))        
        if trials is None:
            trials = np.arange(cls._xyu[experiment].shape[2])
            
        if times  is None:
            times  = np.arange(cls._xyu[experiment].shape[1])
            
        if dims   is None:
            dims   = np.arange(0, cls._offsets[3]-cls._offsets[2])
            
        dims   = toArray(dims)            
        trials = toArray(trials)
        times  = toArray(times)
        
        if np.amax(dims) > cls._offsets[3]-cls._offsets[2]:
            print('requested highest dimension of U:')
            print(np.amax(dims))
            print('stored highest dimension of U:')
            print(cls._offsets[3]-cls._offsets[2])
            raise Exception(('requesting dimensions of U beyond those '
                             'that are stored'))
        else: # transform to indices within cls._xyu
            dims += cls._offsets[2]
        
        try:
            return cls._xyu[experiment][np.ix_(dims,times,trials)].copy()
        except:
            print('Data for U has size = (times,trials) =')
            print(cls._xyu[experiment].shape[1:3])
            raise Exception('Queried data outside of these bounds')
            

    def giveTraces(cls, experiment=0, 
                   xyu='xyu', trials=None, times=None, dims=None):
        """OUT = .giveTracesU(experiment,xyu,trials,times,dims)
        experiment: integer index for stored experiment
        xyu:     string that defines for which of the variable groups X,Y,U
                 data traces are to be returned. Possible values are e.g.  
                 'xyu','xu','y' and other ordered substrings of 'xyu'
        trials:  ndarray of indices for stored trials in selected experiment
        times:   ndarray of indices for time steps in selected experiment
        dims:    ndarray of indices for components of X, Y, U. Overrules xyu        
        OUT: ndarray of number of traces of selected components of data 
        Please note: This method returns a COPY of the desired data traces.
        
        """
        if cls._xyu == []:  # if no experiments added yet, _xyu is empty list
            return np.array([]) # and cannot be indexed as below.

        if not isinstance(experiment, numbers.Integral):
            raise Exception(('can only give input traces for individual '
                             'experiments indexed by a single integer'))
        if experiment >= cls._numExperiments:
            print('number of experiments stored: ')
            print(cls._numExperiments)
            raise Exception(('experiment index too high - asking for an '
                             'experiment that was not (yet) added'))        
                        
        if trials is None:
            trials = np.arange(cls._xyu[experiment].shape[2])
        else:
            trials = toArray(trials)
            
        if times  is None:
            times  = np.arange(cls._xyu[experiment].shape[1])
        else:
            times  = toArray(times)            
            
        if dims   is None:            
            dims   = { 'xyu' : np.arange(cls._offsets[0], cls._offsets[3]),
                       'xy'  : np.arange(cls._offsets[0], cls._offsets[2]),
                       'xu'  : np.concatenate(
                                [np.arange(cls._offsets[0], cls._offsets[1]),
                                 np.arange(cls._offsets[2], cls._offsets[3])]
                                             ),
                       'yu'  : np.arange(cls._offsets[1], cls._offsets[3]),
                        'x'  : np.arange(cls._offsets[0], cls._offsets[1]),
                        'y'  : np.arange(cls._offsets[1], cls._offsets[2]),
                        'u'  : np.arange(cls._offsets[2], cls._offsets[3])
                      }[xyu]
        else:
            dims   = toArray(dims)            
            if not dims.size == len(xyu):
                print(('number of variable groups for which traces '
                       'are requested (xyu):'))
                print(len(xyu))
                print(('number of variable groups for which '
                       'dimensionalities are specified (dims): '))
                print(dims.size)
                raise Exception(('when specifying both arguemnts xyu and '
                                 'dims, they need to agree in length on the '
                                 'described number of variable groups'))
        
        if np.amax(dims) > cls._offsets[3]-cls._offsets[0]:
            print('requested highest dimension:')
            print(np.amax(dims))
            print('stored highest dimension:')
            print(cls._offsets[3]-cls._offsets[0])
            raise Exception(('requesting dimensions of larger than those '
                             'stored of X,Y,U combined'))
        try:
            return cls._xyu[experiment][np.ix_(dims,times,trials)].copy()
        except:
            print('Data has size = (times,trials) =')
            print(cls._xyu[experiment].shape[1:3])
            raise Exception('Queried data outside of these bounds')            
            
    
    def giveMotherObject(cls):
        """OUT = .returnMotherObject()
        OUT: object that called the constructor for this object (possibly None)
        
        """
        return cls._tsobject
                            
#----this -------is ------the -------79 -----char ----compa rison---- ------bar
                                                        
class timeSeriesModel:
    """model definition object for use in the context of state-space models
    
    Assumes the time series to consist of the complete-data series {X, Y} and 
    additionally observed inputs U. 
    Since data models e.g. used for generating toy data and analysis models
    applied to such data may in general differ, all that is hard-imposed onto 
    objects of this class are
    the distinction between observed input u, output y and latent state x, and 
    the disinction between temporal and spatial (i.e. u, y, x) dimensions.

    A state-space model is defined by the structure and functional form of
    the dependencies between U, Y and X. 
    The observed output data Y is usually deemed a noisy function of U and X, 
    but potentially also of past instantiations of Y.
    The latents X are usually functions of past X, and potentially also of U.
    Input U is typically assumed fixed, but may be integrated out via a prior. 
    Each of the variable groups has its own noise added/multiplied to it.

    To allow complex models with e.g. latent state hierachy (X ~ Poiss(exp(Z)),
    the variable groups Y, X and U can be further divided into subgroups.

    INPUTS to constructor:
    tsobject:     pointer to object that called the constructor. 
                  Possibly is 'None', but usually a timeSeriesObject()
    modelClass:   string specifying the model class, e.g.'stateSpace'
    modelDescr:   string specifying the model type, e.g. 'LDS'
    xGr:          np.ndarray of size xDim that specifies the assigned 
                  subgroup of each latent variable. Implicitly defines 
                  xDim with xGr.size!
    yGr:          np.ndarray of size yDim that specifies the assigned 
                  subgroup of each observed variable. Defines yDim! 
    uGr:          np.ndarray of size uDim that specifies the assigned 
                  subgroup of each input variable. Defines uDim!
    dts:          np.ndarray that gives the relative time steps that 
                  are relevant for evaluating the variables of the 
                  current step. E.g. for a classic LDS, dt = -1 
                  (dependency of x on x) and dt = 0 (dep. of y on x)
                  are relevant and dts = np.array([-1,0]) in this case
    deps:         list of np.ndarrays. Length of list has to match size
                  of dts. Each entry of the list is a 2D binary array
                  specifying which subgroups of X, Y depend on which
                  other subgroups of X,Y and U. 
    linkFunctions:     list of lists of strings specifying the types and
                       parameters of the deterministic parts of the condi-
                       tional distr.s linking subgroups of X, Y and U
    noiseDistrs:       list of lists of strings specifying the types and
                       parameters of the random parts of the conditional
                       distributionss linking subgroups of X, Y and U
    noiseInteractions: list of lists of strings specifying how the random
                       and deterministic parts of the conditional distri-
                       butions interact. E.g. '+', 'multiplicative', '^'
    initDistrs:        list of lists of lists of strings specifying the
                       probability distributions of the intial time steps.
                       Length of outermost list has to match dts.size 
    isHomogeneous: boolean that specifies whether the model is 
                   homogeneous over time                      
    
    
    VARIABLES and METHODS:
    .objectDescr            - always equals 'state_space_model_model'   
    .supportedModelClasses  - list of strings of supported model classes
    .defaultModel()         - sets (a mostly blank) default model    
    parsToUpdateParsList()  - interfaces between standard parameter
                              collections as for LDS and the parameter
                              convention of the timeSeries framework
    .updatePars()           - updates stored parameters of the model
    .givePars()             - returns pointer to parameters of the model
    .giveVarDescr()         - returns pointer to model parameters description     
    .giveFactorization()    - returns pointer to model parameters factorization
    .giveModelDescription() - returns string modelDescr (see above)
    .giveIfHomogeneous()    - returns boolean whether model is homogeneous 
    .giveIfCausal()         - returns boolean whether model is causal
    .giveMotherObject()     - returns object one step above in object hierarchy 
    .fit()                  - fits the model, depending on model type
    _fitLDS()               - fits model if LDS
    _LDSlogLikelihood()     - log-likelihood of LDS model
    _LDS_E_step()           - E-step of EM algorithm for LDS model
    _KalmanFilter()         - part of E-step for LDS model
    _KalmanSmoother()       - part of E-step for LDS model
    _KalmanParsToMoments()  - translates posterior parameters to moments
    _LDS_M_step             - M-step for EM algorithm for LDS model


    """
    objDescr = 'state_space_model_model'
    
    supportedModelClasses = ['stateSpace', 'default', 'unknown', 'none']
    
    def __init__(self,
                 tsobject = None,
                 modelClass='default',
                 modelDescr='custom',
                 xGr=0, yGr=0, uGr=0, 
                 dts=None,deps=None,
                 linkFunctions=None,
                 noiseDistrs=None,
                 noiseInteractions=None,
                 initDistrs=None,                 
                 isHomogeneous=True):
                            
        self._tsobject         = tsobject        
        if modelClass in self.supportedModelClasses:
            self._modelClass        = modelClass
        else:
            print('selected model class:')
            print(modelClass)
            print('supported model classes:')
            print(supportedModelClasses)
            raise Exception('selected model class not supported')
        
        if modelClass=='empirical':
            if not modelDescr == 'none':
                print('modelDescr:')
                print( modelDescr  )                
                print(('WARNING: only allowed model description for model '
                       'class "empirical" is modelDescr="none". Will use '
                       'this instead.'))
                self._modelDescr    = 'none'
                self_.isCausal      = True  # some bold guesses regarding 
                self._isHomogeneous = False # the state of the world
                
        elif modelClass=='stateSpace':
                        
            self._modelDescr    = ('stateSpace_' + modelDescr) 
            if isinstance(isHomogeneous, bool):
                self._isHomogeneous = isHomogeneous # 'true' or 'false'
            else:
                raise Exception('isHomogeneous has to be boolean')

            self._varDescr = stateSpaceModelVariables(xGr,yGr,uGr,dts,deps) 
            if self._varDescr.giveVarTimeScope()[-1]>0:
                self._isCausal      = False
                print(('WARNING: using acausal state-space model! Ancestral '
                       'sampling and other algorithms might not work'))
            else:
                self._isCausal      = True

            self._factorization = stateSpaceModelFactorization(
                                                    self,
                                                    linkFunctions,
                                                    noiseDistrs,
                                                    noiseInteractions,
                                                    initDistrs
                                                              )
                            
                            
            # .... 
        elif modelClass in ['default']:
            self._modelDescr = modelDescr
        elif modelClass in ['empirical','unknown','experiment']:
            self._modelDescr = 'unknown'
            print(('no data model. Assuming data is given by real experiment '
                   'or other external source'))
        else:
            print('modelClass = ')
            print(modelClass)
            raise Exception('unknown model class')
                   
    def fit(cls, 
            maxIter=1000, 
            epsilon=np.log(1.05), # stop if likelihood change < 5%
            initPars=None,
            ifPlotProgress=False,
            experiment=0, 
            trials=None, 
            times=None):
        """ OUT = .fit(maxIter,epsilon,initPars, 
                 ifPlotProgress,experiment,trials,times)
        maxIter:  maximum allowed iterations for iterative fitting (e.g. EM)
        epsilon:  convergence criterion, e.g. difference of log-likelihoods
        initPars: set of parameters to start fitting. If == None, the 
                  parameters currently stored in the model will be used 
        ifPlotProgress: boolean, specifying if fitting progress is visualized
        experiment: integer index for stored experiment
        trials:  ndarray of indices for stored trials in selected experiment
        times:   ndarray of indices for time steps in selected experiment
        OUT:     List of performance measures (e.g. log-likelihoods) produced 
                 during the model fitting procedure
        Fits the stored model to data stored in the 'empirical' object.
        Will call other functions depending on the model type. 
        
        """        
        
        try:
            data = cls._tsobject.giveMotherObject().giveEmpirical().giveData()
        except:
            raise Exception(('error trying to fit model to data. Could not '
                             'find empirical data object!'))
        y = data.giveTracesY(experiment, trials, times) # get all the
        u = data.giveTracesU(experiment, trials, times) # observed data

        if cls._modelDescr == 'stateSpace_LDS':
            if initPars is None:                # use current pars as init
                [A, Q, mu0, V0, C, R] = cls.givePars().copy() 
            elif isinstance(initPars, list) and len(initPars)==6: 
                [A, Q, mu0, V0, C, R] = initPars
            elif isinstance(initPars, dict):
                [A, Q, mu0, V0, C, R] = cls.givePars().copy()
                if 'A' in initPars.keys():             
                    A   = initPars['A']                
                if 'Q' in initPars.keys():             # use current ...
                    Q   = initPars['Q']                # ... but overwrite
                if 'mu0' in initPars.keys():           # whatever is given
                    mu0 = initPars['mu0']
                if 'V0' in initPars.keys():
                    V0  = initPars['V0']
                if 'C' in initPars.keys():                    
                    C = initPars['C']
                if 'R' in initPars.keys():                    
                    R = initPars['R']
                    
            xDim = cls._varDescr.giveVarDims('x')
            [As, Qs, mu0s, V0s, Cs, Rs, LLs] = \
                           timeSeriesModel._fitLDS(y, 
                                                   [A, Q, mu0, V0, C, R], 
                                                   maxIter, 
                                                   epsilon, 
                                                   ifPlotProgress,
                                                   xDim)
            [linkPars, noisePars, initPars] = \
                timeSeriesModel.parsToUpdateParsList([As[-1],Qs[-1],
                                                      mu0s[-1],V0s[-1],
                                                      Cs[-1],Rs[-1]], 
                                                      cls._modelDescr)
            cls.updatePars(linkPars, noisePars, initPars)

            print('data log-likelihood for parameter initializations:')
            print(LLs[0]) # guaranteed to have length >= 1
            print('data log-likelihood for fitted parameters:')
            print(LLs[-1])
            
            return LLs  
            
        
        else:
            print('current model type:')
            print(cls._modelDescr)
            raise Exception(('fitting procedures for this model type are '
                             'not (yet) implemented!'))
     
    @staticmethod # i.e. it cannot touch cls = timeSeriesModel() to obtain any
                  # additional data such as parameters or variable  
                  # dimensionalities- everything has to be handed over !
    def _fitLDS(y, 
                initPars=None, 
                maxIter=1000, 
                epsilon=np.log(1.05), # stop if likelihood change < 5%
                ifPlotProgress=False, 
                xDim=None):
        """ OUT = _fitLDS(y*,initPars, maxIter, epsilon, 
                    ifPlotProgress,xDim)
        initPars: set of parameters to start fitting. If == None, the 
                  parameters currently stored in the model will be used,
                  otherwise needs initPars = [A,Q,mu0,V0,C,R]
        maxIter:  maximum allowed iterations for iterative fitting (e.g. EM)
        epsilon:  convergence criterion, e.g. difference of log-likelihoods
        ifPlotProgress: boolean, specifying if fitting progress is visualized
        xDim:     dimensionality of (sole subgroup of) latent state X
        Fits an LDS model to data stored in the 'empirical' object.
        
        """
        
        yDim = y.shape[0] 
                
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
            raise Exception('argument epsilon has to be a boolean')
            
        # we will work with continuously changing working copies of
        # the parameter initializations (no memory track along updates):
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
            C   =       np.random.normal(size=[yDim, xDim])            
        elif np.all(initPars[4].shape==(yDim,xDim)): 
            C   = initPars[4].copy()
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
            R   =       np.identity(yDim)            
        elif np.all(initPars[5].shape==(yDim,yDim)): 
            R   = initPars[5].copy()
        else:
            print('yDim:')
            print(yDim)
            print('R.shape:')
            print(initPars[5].shape)
            raise Exception(('Bad initialization for LDS parameter R.'
                             'Shape not matching dimensionality of y'))

        E_step = timeSeriesModel._LDS_E_step # greatly improves
        M_step = timeSeriesModel._LDS_M_step # readability ...
        LL     = timeSeriesModel._LDSlogLikelihood 
        
        # evaluate initial state       
        [Ext, Extxt, Extxtm1, LLtr] = E_step(A, Q, mu0, V0, C, R, y)
        #LL_new                = LL(A,Q,mu0,V0,C,R,Ext,Extxt,Extxtm1,y)      
        LL_new = np.sum(LLtr)
        LL_old = -float('Inf')
        dLL = []              # performance trace for status plotting
        log10 = np.log(10)    # for convencience, see below
        LLs = [LL_new.copy()] # performance trace to be returned
        stepCount = 0        
        ifBreakLoop = False
        # start EM iterations, run until convergence 
        Exts    = [Ext]
        Extxts  = [Extxt]
        Extxtm1s= [Extxtm1]
        As   = [A]
        Qs   = [Q]
        mu0s = [mu0]
        V0s  = [V0]
        Cs   = [C]
        Rs   = [R]
        
        while LL_new - LL_old > epsilon and stepCount < maxIter:

            LL_old = LL_new            
            [A,Q,mu0,V0,C,R]      = M_step(Exts[stepCount], 
                                           Extxts[stepCount], 
                                           Extxtm1s[stepCount],
                                           y)
            As.append(A.copy())
            Qs.append(Q.copy())
            mu0s.append(mu0.copy())
            V0s.append(V0.copy())
            Cs.append(C.copy())
            Rs.append(R.copy())    
            stepCount += 1            
            
            [Ext, Extxt, Extxtm1, LLtr] = E_step(As[stepCount], 
                                                 Qs[stepCount], 
                                                 mu0s[stepCount], 
                                                 V0s[stepCount], 
                                                 Cs[stepCount], 
                                                 Rs[stepCount],
                                                 y)
            Exts.append(Ext.copy())
            Extxts.append(Extxt.copy())
            Extxtm1s.append(Extxtm1.copy())
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
                inp = input("Enter (y)es or (n)o: ")
                if inp == "no" or inp.lower() == "n":
                    return None # break EM loop because st. is wrong
            dLL.append(np.log(LL_new - LL_old)/log10)
                
        LLs = toArray(LLs)        
        return [As, Qs, mu0s, V0s, Cs, Rs, LLs]

    @staticmethod
    def _LDSlogLikelihood(A,Q,mu0,V0,C,R,Ext,Extxt,Extxtm1,y):
        """ OUT=_LDSlogLikelihood(A*,Q*,mu0*,V0*,C*,R*,Ext*,Extxt*,Extxtm1*,y*)
        see Bishop, 'Pattern Recognition and Machine Learning', ch. 13
        for formulas and input/output naming conventions   
        CURRENTLY LACKS TERMS FOR ENTROPY OF POSTERIOR q AND PRODUCES 
        WRONG RESULTS. DO NOT USE!
        (E-step can directly LL from Kalman-filter forward pass)

        """
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
            sMu0mMu1V0Inv += np.inner(mu0 - Ext[:,0,tr], 
                                      np.dot(V0inv, mu0-Ext[:,0,tr])) 
            tr += 1
            
        # pre-compute statistics important for  terms related to x_t | x_{t-1}
        sExtxtm1 = np.sum(Extxtm1[:,:,1:T,:], (2,3)) # sum over E[x_t x_{t-1}']        
        sExtxt2toN   = np.sum(             Extxt[:,:,1:T,:], (2,3) )  # sums 
        sExtxt1toN   = sExtxt2toN + np.sum(Extxt[:,:,  0,:],2)  # over 
        sExtxt1toNm1 = sExtxt1toN - np.sum(Extxt[:,:,T-1,:],2)  # E[x_t x_t']

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
                    + Trial * xDim * np.log(2*sp.pi)  # + constant in mu0, V0 
                    ) 
        # add E_q[log p(x_t, x_{t-1} | Y, \theta)] for t = 2, ..., T
        Qinv   = np.linalg.inv(Q)    
        QinvA  = np.dot(Qinv, A)
        AQinvA = np.dot(A.transpose(), QinvA)    
        LL -= 1/2 * ( Trial * (T-1) * np.log(np.linalg.det(Q))
                    +     np.sum(  Qinv  * sExtxt2toN ) 
                    - 2 * np.sum(  QinvA * sExtxtm1 )
                    +     np.sum( AQinvA * sExtxt1toNm1 )
                    + Trial*(T-1)*xDim*np.log(2*sp.pi)  # + constant in A, Q 
                    ) 
        # add E_q[log p(y_t, x_t     |  \theta)  ] for t = 1, ..., T
        CRinvC = np.dot(np.dot( C.transpose(), Rinv), C)    
        LL -= 1/2 * ( Trial * T * np.log(np.linalg.det(R)) 
                    +     syRinvy 
                    - 2 * syRinvCExt 
                    +     np.sum( CRinvC * sExtxt1toN ) 
                    + Trial*T*yDim*np.log(2*sp.pi)      # + constant in C, R
                    )
        # add - E_q[log q( x )], i.e. the entropy of q(x) 
        #LL += 1/2 * ( Trial * np.log(np.linalg.det(V0)) # from H[x_1]
        #            + Trial *(T-1)*np.log(np.linalg.det(Q)) # H[x_n|x_{n-1}] 
        #            + Trial *T*xDim*np.log(2*sp.pi) # + constant in A, Q 
        #            )

        return LL

    @staticmethod
    def _LDS_E_step(A,Q,mu0,V0,C,R,y): 
        """ OUT = _LDS_E_step(A*,Q*,mu0*,V0*,C*,R*,y*)
        see Bishop, 'Pattern Recognition and Machine Learning', ch. 13
        for formulas and input/output naming conventions   
        """
        [mu,V,P,K,c] = timeSeriesModel._KalmanFilter(A,Q,mu0,V0,C,R,y)
        [mu_h,V_h,J] = timeSeriesModel._KalmanSmoother(A, mu, V, P)

        [Ext,Extxt,Extxtm1]=timeSeriesModel._KalmanParsToMoments(mu_h,V_h,J)

        LL = np.sum(np.log(c),axis=0) # sum over times, get Trial-dim. vector
        
        return [Ext, Extxt, Extxtm1, LL]

    @staticmethod
    def _KalmanFilter(A,Q,mu0,V0,C,R,y):        
        """ OUT = _KalmanFilter(A*,Q*,mu0*,V0*,C*,R*,y*)
        see Bishop, 'Pattern Recognition and Machine Learning', ch. 13
        for formulas and input/output naming conventions   
        """
        xDim  = A.shape[0]
        yDim  = y.shape[0]
        T     = y.shape[1]
        Trial = y.shape[2]
        mu = np.zeros([xDim,     T,Trial])
        V  = np.zeros([xDim,xDim,T,Trial])
        P  = np.zeros([xDim,xDim,T,Trial])
        K  = np.zeros([xDim,yDim,T,Trial])
        c  = np.zeros([          T,Trial])
        mvnpdf = sp.stats.multivariate_normal.pdf
        Id = np.identity(xDim)
        tr = 0

        Atr = A.transpose()
        Ctr = C.transpose()
        while tr < Trial:
            # first time step: [mu0,V0] -> [mu1,V1]
            Cmu0    = np.dot(C,mu0)
            P0      = V0 # = np.dot(np.dot(A, V0), Atr) + Q
            CPCR    = np.dot(np.dot(C,P0), Ctr) + R
            CPCRinv = np.linalg.inv(CPCR)
            K[:,:,0,tr] = np.dot(np.dot(P0,Ctr),CPCRinv)
            mu[ :,0,tr] = mu0 + np.dot(K[:,:,0,tr],y[:,0,tr]-Cmu0)
            V[:,:,0,tr] = np.dot(Id - np.dot(K[:,:,0,tr],C),P0)
            P[:,:,0,tr] = np.dot(np.dot(A,V[:,:,0,tr]), Atr) + Q
            c[    0,tr] = mvnpdf(y[:,0,tr], mean=Cmu0, cov=CPCR)
            t = 1 # now start with second time step ...
            while t < T:
                Amu  = np.dot(A,mu[:,t-1,tr])
                CAmu = np.dot(C,Amu)
                PCtr = np.dot(P[:,:,t-1,tr], Ctr)
                CPCR    = np.dot(C, PCtr) + R
                CPCRinv = np.linalg.inv(CPCR)
                K[:,:,t,tr] = np.dot(PCtr,CPCRinv)
                mu[ :,t,tr] = Amu + np.dot(K[:,:,t,tr],y[:,t,tr]-CAmu)
                V[:,:,t,tr] = np.dot(Id - np.dot(K[:,:,t,tr],C),P[:,:,t-1,tr])
                P[:,:,t,tr] = np.dot(np.dot(A,V[:,:,t,tr]), Atr) + Q
                c[    t,tr] = mvnpdf(y[:,t,tr], mean=CAmu, cov=CPCR)
                t += 1
            tr += 1
        return [mu,V,P,K,c]
    
    @staticmethod
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

    @staticmethod
    def _KalmanParsToMoments(mu_h, V_h, J):
        """ OUT = _KalmanParsToMoments(mu)h*,V_h*,J*)
        see Bishop, 'Pattern Recognition and Machine Learning', ch. 13
        for formulas and input/output naming conventions   
        """                
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

    @staticmethod
    def _LDS_M_step(Ext, Extxt, Extxtm1, y):   
        """ OUT = _LDS_M_step(Ext*,Extxt*,Extxtm1*,y*)
        see Bishop, 'Pattern Recognition and Machine Learning', ch. 13
        for formulas and input/output naming conventions   
        """                        
        xDim  = Ext.shape[0]
        T     = Ext.shape[1]
        Trial = Ext.shape[2]    
        yDim  = y.shape[0]

        sExtxtm1 = np.sum(Extxtm1[:,:,1:T,:], (2,3)) # sum over E[x_t x_{t-1}']        
        sExtxt2toN   = np.sum(             Extxt[:,:,1:T,:], (2,3) )  # sums 
        sExtxt1toN   = sExtxt2toN + np.sum(Extxt[:,:,  0,:],2)  # over 
        sExtxt1toNm1 = sExtxt1toN - np.sum(Extxt[:,:,T-1,:],2)  # E[x_t x_t']  
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

    def updatePars(cls, linkPars=None, noisePars=None, initPars=None):  
        """ updatePars(linkPars, noisePars, initPars)
        linkFunctions:     list of lists of strings specifying the types and
                           parameters of the deterministic parts of the condi-
                           tional distr.s linking subgroups of X, Y and U
        noiseDistrs:       list of lists of strings specifying the types and
                           parameters of the random parts of the conditional
                           distributionss linking subgroups of X, Y and U
        noiseInteractions: list of lists of strings specifying how the random
                           and deterministic parts of the conditional distri-
                           butions interact. E.g. '+', 'multiplicative', '^'
        initDistrs:        list of lists of lists of strings specifying the
                           probability distributions of the intial time steps.
                           Length of outermost list has to match dts.size 
        Updates the parameters of a model 

        """
        if cls._modelClass == 'empirical':
            print('"empirical" model has no parameters to update')
            return None
        
            # assemble pointers to containers for paramters
        linkFuns   = cls.giveFactorization().giveLinkFunctionList()
        noiseDists = cls.giveFactorization().giveNoiseDistrList()
        initDists  = cls.giveFactorization().giveInitialDistrList()
            
        # assemble information needed to traverse above containers
        varDescr = cls.giveVarDescr()
        xyuSubgroupTallies = varDescr.giveVarSubgroupTallies()
        dts                = varDescr.giveVarTimeScope()
            
        # apply parameter changes ...
        # ... to the deterministic parts of the conditional distributions
        if not linkPars is None:            
            if not isinstance(linkPars, list):
                print('linkPars')
                print( linkPars )
                raise Exception(('argument linkPars has to be list. '
                                 'It is not.'))
            elif len(linkPars)!=2:
                print('linkPars')
                print( linkPars )
                raise Exception(('argument linkPars has to contain two '
                                 'lists - one each for the parameters of'
                                 ' subgroups of X and Y, respectively.'))
            elif len(linkPars[0])!=xyuSubgroupTallies[0]:
                print('linkPars[0]')
                print( linkPars[0] )
                raise Exception(('linkPars[0] has to contain as many '
                                 'entries as there are subgroups of X'))
            elif len(linkPars[1])!=xyuSubgroupTallies[1]:
                print('linkPars[1]')
                print( linkPars[1] )
                raise Exception(('linkPars[1] has to contain as many '
                                 'entries as there are subgroups of Y'))
            # if we got up to here, everything seems fine: 
            for i in range(xyuSubgroupTallies[0]):
                if not linkPars[0][i] is None:
                    linkFuns[0][i].updatePars(linkPars[0][i])
            for i in range(xyuSubgroupTallies[1]):
                if not linkPars[1][i] is None:
                    linkFuns[1][i].updatePars(linkPars[1][i])

        # ... to the random parts of the conditional distributions
        if not noisePars is None:            
            if not isinstance(noisePars, list):
                print('noisePars')
                print( noisePars )
                raise Exception(('argument noisePars has to be list. '
                                 'It is not.'))
            elif len(noisePars)!=2:
                print('noisePars')
                print( noisePars )
                raise Exception(('argument noisePars has to contain two '
                                 'lists - one each for the parameters of'
                                 ' subgroups of X and Y, respectively.'))
            elif len(noisePars[0])!=xyuSubgroupTallies[0]:
                print('noisePars[0]')
                print( noisePars[0] )
                raise Exception(('noisePars[0] has to contain as many '
                                 'entries as there are subgroups of X'))
            elif len(noisePars[1])!=xyuSubgroupTallies[1]:
                print('noisePars[1]')
                print( noisePars[1] )
                raise Exception(('noisePars[1] has to contain as many '
                                 'entries as there are subgroups of Y'))
            # if we got up to here, everything seems fine: 
            for i in range(xyuSubgroupTallies[0]):
                if not noisePars[0][i] is None:
                    noiseDists[0][i].updatePars(noisePars[0][i])
            for i in range(xyuSubgroupTallies[1]):
                if not noisePars[1][i] is None:
                    noiseDists[1][i].updatePars(noisePars[1][i])
                
        # ... to the initial distributions                    
        if not initPars is None:
            if not isinstance(initPars, list):
                print('initPars')
                print( initPars )
                raise Exception(('argument noisePars has to be list. '
                                 'It is not.'))
            elif len(initPars)!=len(np.where(dts<0)):
                print('initPars')
                print( initPars )
                raise Exception(('argument initPars has to contain as '
                                 'many entries as there are relevant '
                                 'time offsets dt stored in dts.'))
            for dti in np.where(dts<0):
                if len(initPars[dti])!=2:
                    print('dti:')
                    print( dti  )
                    print('initPars[dti]')
                    print( initPars[dti] )
                    raise Exception(('argument initPars[dti] has to '
                                     'contain two lists - one each for '
                                     'the parameters of subgroups of X '
                                     'and Y, respectively.'))
                elif len(initPars[dti][0])!=xyuSubgroupTallies[0]:
                    print('dti:')
                    print( dti  )
                    print('initPars[dti][0]')
                    print( initPars[dti][0] )
                    raise Exception(('initPars[dti][0] has to contain '
                                     'as many entries as there are '
                                     'subgroups of X'))
                elif len(initPars[dti][1])!=xyuSubgroupTallies[1]:
                    print('dti:')
                    print( dti  )
                    print('initPars[dti][1]')
                    print( initPars[dti][1] )
                    raise Exception(('initPars[dti][1] has to contain '
                                     'as many entries as there are '
                                     'subgroups of Y'))
            # if we got up to here, everything seems fine: 
            for dti in np.where(dts<0):
                for i in range(xyuSubgroupTallies[0]):
                    if not initPars[dti][0][i] is None:
                        tmp = initPars[dti][0][i]
                        initDists[dti][0][i].updatePars(tmp)
                for i in range(xyuSubgroupTallies[1]):
                    if not initPars[dti][1][i] is None:
                        tmp = initPars[dti][1][i]
                        initDists[dti][1][i].updatePars(tmp)
                    
    @staticmethod
    def parsToUpdateParsList(pars, modelDescr):
        """ OUT = parsToUpdateParsList(pars*,modelDescr*)
        pars:       list or dict of parameters as ordered and/or called in 
                    standard descriptions of statistical models.
        modelDescr: string specifying the model used, e.g. 'stateSpace_LDS'
        OUT:        list of [linkPars, noisePars, initPars] as used by the
                    timeSeries framework
        Translates standard variable collections from model descriptions, 
        such as e.g. [A,Q,mu0,V0,C,R] for the LDS, into the convention used
        by the timeSeries framework as encoded in linkPars, noisePars, and
        initPars. Can only be used for known and supported models. 
        Using this method for new models will require updating the method!    

        """
        if not isinstance(pars, (list, dict)):
            print('pars:')
            print(pars)
            raise Exception(('argument pars has to be a LIST  or DICT '
                             'containing the new parameters selected '
                             'for an update'))
            
        linkPars  =  [[],[]]
        noisePars =  [[],[]]
        initPars  = [[[],[]]] # outermost list for time steps dt !        
        
        if modelDescr == 'stateSpace_LDS':
            if (isinstance(pars, list) and  len(pars) != 6):
                print('len(pars):')
                print(len(pars))
                raise Exception(('for LDS, argument pars has to have '
                                 'a length of exactly 6! It does not.'))
                
            elif isinstance(pars, list):                         
                A   = pars[0]
                Q   = pars[1]
                mu0 = pars[2]
                V0  = pars[3]
                C   = pars[4]
                R   = pars[5]
                # make sure the ordering of parameters follows a sorting
                # by [dt,varGroup,varSubgroup], i.e. first have those 
                # parameters for the smallest dt (dt=-1 before dt=0 etc.),  
                # then in a nested loop sorted by varGroup (X before Y) 
                #  and then by subgroup index.
                linkPars[0] = [ A ]       
                linkPars[1] = [ C ]      
                noisePars[0] = [ [None, Q] ]
                noisePars[1] = [ [None, R] ]                 
                initPars[0][0] = [ [mu0, V0] ]
                initPars[0][1] = [   None    ]  
            elif isinstance(pars, dict):
                linkPars[0] = [ None ]       
                linkPars[1] = [ None ]      
                noisePars[0] = [ [None, None] ]
                noisePars[1] = [ [None, None] ]                 
                initPars[0][0] = [ [None, None] ]
                initPars[0][1] = [   None    ]  
                if 'A' in pars.keys():
                    linkPars[0][0] = pars['A']
                if 'C' in pars.keys():
                    linkPars[1][0] = pars['C']
                if 'Q' in pars.keys():
                    noisePars[0][1] = pars['Q']
                if 'R' in pars.keys():
                    noisePars[1][1] = pars['R']
                if 'mu0' in pars.keys():
                    initPars[0][0][0] = pars['mu0']
                if 'V0' in pars.keys():
                    initPars[0][0][1] = pars['V0']                              
            
        elif modelDescr == 'stateSpace_inputARLDS':
            if isinstance(pars, list) and  len(pars) != 11:
                print('len(pars):')
                print(len(pars))
                raise Exception(('for inputARLDS, argument pars has to '
                                 'have length of exactly 11! It does not.'))
            elif isinstance(pars, list):                                         
                A    = pars[0]
                B    = pars[1]
                Q    = pars[2]
                mu0  = pars[3]
                V0   = pars[4]
                C    = pars[5]
                D    = pars[6]
                E    = pars[7]
                R    = pars[8]
                nu0  = pars[9]
                W0   = pars[10]                        
                linkPars[0]= [  A,B  ]
                linkPars[1]= [ E,C,D ]
                noisePars[0] = [ [None, Q] ]
                noisePars[1] = [ [None, R] ]         
                initPars[0][0] = [ [mu0, V0] ] # initPars[dt][X] = list(...)
                initPars[0][1] = [ [nu0, W0] ] # initPars[dt][Y] = list(...)
            elif isinstance(pars, dict):         
                linkPars[0]= [  None,None  ]     
                linkPars[1]= [ None,None,None ]     
                noisePars[0] = [ [None, None] ]
                noisePars[1] = [ [None, None] ]         
                initPars[0][0] = [ [None, None] ] 
                initPars[0][1] = [ [None, None] ]   
                if 'A' in pars.keys():
                    linkPars[0][0] = pars['A']
                if 'B' in pars.keys():
                    linkPars[0][1] = pars['B']
                if 'E' in pars.keys():
                    linkPars[1][0] = pars['E']                                
                if 'C' in pars.keys():
                    linkPars[1][1] = pars['C']
                if 'D' in pars.keys():
                    linkPars[1][2] = pars['D']                                
                if 'Q' in pars.keys():
                    noisePars[0][1] = pars['Q']
                if 'R' in pars.keys():
                    noisePars[1][1] = pars['R']
                if 'mu0' in pars.keys():
                    initPars[0][0][0] = pars['mu0']
                if 'V0' in pars.keys():
                    initPars[0][0][1] = pars['V0']                                   
                if 'nu0' in pars.keys():
                    initPars[0][1][0] = pars['nu0']
                if 'W0' in pars.keys():
                    initPars[0][1][1] = pars['W0']               
        else: 
            print('modelDescr:')
            print(modelDescr)
            raise Exception(('method updatePars not yet implemented for the '
                             'selected model description! Will need to '
                             'update the parameters of link functions, noise '
                             'and initial distributions one at a time through '
                             'directly calling and individually updating.'))  
            
        return [linkPars, noisePars, initPars]

    def givePars(cls):        
        """ OUT = .givePars()
        OUT:        list or dict of parameters as ordered and/or called in 
                    standard descriptions of statistical models.
        Translates the convention for parameter storage used as by the  
        timeSeries framework and as encoded in linkPars, noisePars, initPars 
        into a more widely used description of variables, such as e.g. 
        [A,Q,mu0,V0,C,R] for the LDS. Can only be used for supported models. 
        Using this method for new models will require updating the method!        

        """        
        if cls._modelClass == 'empirical':
            print('"empirical" model has no parameters to update')
            return None
        
        if cls._modelDescr == 'stateSpace_LDS':
            linkFuns   = cls.giveFactorization().giveLinkFunctionList()
            noiseDists = cls.giveFactorization().giveNoiseDistrList()
            initDists  = cls.giveFactorization().giveInitialDistrList()
            
            A =   linkFuns[0][0].givePars()[0].copy()     
            Q =   noiseDists[0][0].givePars()[1].copy()
            mu0 = initDists[0][0][0].givePars()[0].copy()
            V0  = initDists[0][0][0].givePars()[1].copy()
            C   = linkFuns[1][0].givePars()[0].copy()
            R =   noiseDists[1][0].givePars()[1].copy()
            
            return [A,Q,mu0,V0,C,R]
        
        else: 
            print('modelDescr:')
            print(modelDescr)
            raise Exception(('method _updatePars not yet implemented for the '
                             'selected model description! Will need to '
                             'update the parameters of link functions, noise '
                             'and initial distributions one at a time through '
                             'directly calling and individually updating.'))            
            
    def giveVarDescr(cls):   
        """OUT = .giveVarDescr()
        OUT: pointer to timeSeriesVariables() object describing subgroups of
             variables X,Y,U and their high-level dependencies
             
        """ 
        if cls._modelClass == 'empirical':
            print('"empirical" model has no variable description')
            return None
        
        if cls._modelClass in cls.supportedModelClasses:
            try:
                return cls._varDescr
            except:
                raise Exception(('variable description object "varDescr" '
                                 ' apparently not (yet?) initialized'))                
        else:
            raise Expection(('currently used model class does not support '
                             ' variable description objects varDescr'))
            
    def giveFactorization(cls):
        """OUT = .giveFactorization()
        OUT: pointer to timeSeriesFactorization() object holding a 
             description of the interrelation of subgroups of X,Y and U
             
        """
        if cls._modelClass == 'empirical':
            print('"empirical" model has no variable factorization')
            return None
        
        if cls._modelClass in cls.supportedModelClasses:
            try:
                return cls._factorization
            except:
                raise Exception(('model factorization object "factorization" '
                                 ' apparently not (yet?) initialized'))
        else:
            raise Expection(('currently used model class does not support '
                             ' factorizaiton description objects'))

    def giveModelDescription(cls):
        """OUT = .giveModelDescription()
        OUT: pointer to string shortly describing the currently used model 
        
        """
        if cls._modelClass in cls.supportedModelClasses:
            return cls._modelDescr
        else:
            raise Expection(('currently used model class does not support '
                             ' standard model description'))
            
    def giveIfHomogeneous(cls):
        """OUT = .giveModelDescription()
        OUT: boolean giving whether the model is homogeneous over time 
        
        """        
        if cls._modelClass in cls.supportedModelClasses:
            return cls._isHomogeneous
        else:
            raise Expection(('currently used model class does not support '
                             ' description in terms of temporal homogeneity'))

    def giveIfCausal(cls):
        """OUT = .giveModelDescription()
        OUT: boolean giving whether the model is causal 
        """
        
        if cls._modelClass in cls.supportedModelClasses:
            return cls._isCausal
        else:
            raise Expection(('currently used model class does not support '
                             ' description in terms of temporal causality'))
            
    def giveMotherObject(cls):
        """OUT = .returnMotherObject()
        OUT: object that called the constructor for this object (possibly None)
        """
        
        return cls._tsobject

    
    @classmethod                            
    def defaultModel(cls, modelDescr='default'):
        """OUT = .defaultModel(modelDescr)
        modelDescr: string specifying the model used, e.g. 'stateSpace_LDS'
        OUT:        pointer to timeSeriesModel() initialized in default state
        
        """        
        xGr=0
        yGr=0
        uGr=0
        modelClass='unknown'
        linkFunctions=None
        noiseDistrs=None
        noiseInteractions=None
        initDistrs=None
        ifStatesSpace=False
        model = cls(modelClass,
                    modelDescr,
                    xGr, yGr, uGr,
                    linkFunctions,
                    noiseDistrs,
                    noiseInteractions,
                    initDistrs,
                    ifStatesSpace)         
        return model
        
                            
#----this -------is ------the -------79 -----char ----compa rison---- ------bar

class stateSpaceModelVariables:
    """subgroup definition for variables X, Y, Z of a state-space model
    
    State-space models over {X,Y} with potentially additional input variables U
    are defined by (conditional) probability distribution of some subgroup of
    variables U, X, Y given another. The full probability of X,Y|U factors into
    these conditional distributions. stateSpaceModelFactor objects each 
    describe one of these dependencies, such as p(Y | X, U) or p(X_t | X_{t-1}).
    
    For more complex models with e.g. latent state hierachy (X ~ Poiss(exp(Z))
    the distinction between Y, X and U has to be refined. 
    
    Objects of this class serve to encode and store such subgroup structure 
    among X, Y, U. 
    
    INPUTS to the constructor:
    xGr:   np.ndarray of size xDim that specifies the assigned 
           subgroup of each latent variable. Implicitly defines 
           xDim with xGr.size!
    yGr:   np.ndarray of size yDim that specifies the assigned 
           subgroup of each observed variable. Defines yDim! 
    uGr:   np.ndarray of size uDim that specifies the assigned 
           subgroup of each input variable. Defines uDim!
    dts:   np.ndarray that gives the relative time steps that 
           are relevant for evaluating the variables of the 
           current step. E.g. for a classic LDS, dt = -1 
           (dependency of x on x) and dt = 0 (dep. of y on x)
           are relevant and dts = np.array([-1,0]) in this case
    deps:  list of np.ndarrays. Length of list has to match size
           of dts. Each entry of the list is a 2D binary array
           specifying which subgroups of X, Y depend on which
           other subgroups of X,Y and U.     
    dtsX:  optional, gives relevant time offsets for X     
    dtsY:  optional, gives relevant time offsets for Y   
    dtsU:  optional, gives relevant time offsets for U   
    
    
    VARIABLES and METHODS:
    .objectDescr              - always equals 'state_space_model_varDescr'   
    .giveVarDims()            - returns dimensionalities of variables X,Y,U
    .giveVarSubgroupTallies() - returns the numbers of subgroups of X,Y,U
    .giveVarSubgroupSizes()   - returns the sizes of subgroups of X,Y,U
    .giveVarSubgroupLabels()  - returns the subgroup labels of each individual
                                compoment of X,Y,U, respectively
    .giveVarSubgroupIndices() - returns the components of X,Y,U that go with
                                all variable subgroups, respectively
    .giveVarTimeScope()       - return array of relevant time offsets for 
                                evaluating the current state. 
    .giveVarDependencies()    - return a list of arrays describing how the 
                                subgroups of X,Y depend on subgroupf of X,Y,U
    .checkVarDependencies()   - checks the consistency of variable dependencies
    
    """
    objDescr = 'state_space_model_varDescr'
            
    def __init__(self, 
                 xGr=0, 
                 yGr=0, 
                 uGr=0, 
                 dts=[0],
                 deps=None,
                 dtsU=None,
                 dtsX=None,
                 dtsY=None):
        
        if isinstance(dts,(numbers.Integral,float,np.ndarray)):
            self._dts = np.unique(toArray(dts))
        elif isinstance(dts, list) and len(dts) > 0:
            self._dts = np.unique(toArray(dts))
        elif isinstance(dts, np.ndarray) and dts.size == 0:
            print('temporal offset summary variable dts: ')
            print(dts)
            raise Exception(('temporal offset summary variable dts has to '
                             'be nonempty'))
        else:
            print('temporal offset summary variable dts: ')
            print(dts)
            raise Exception(('temporal offset summary variable dts has to '
                             'be nonempty int, float, list or ndarray'))
            
        if isinstance( xGr, (numbers.Integral,list) ):
            xGr = toArray(xGr)            
        if not isinstance(xGr, np.ndarray): # if not true by now ...
            raise TypeError(('variable xGr may either be an integer, a list '
                             'or a ndarray'))                            

        if isinstance( yGr, (numbers.Integral,list) ):
            yGr = toArray(yGr)            
        if not isinstance(yGr, np.ndarray): # if not true by now ...
            raise TypeError(('variable yGr may either be an integer, a list '
                             'or a ndarray'))                            
            
        if isinstance( uGr, (numbers.Integral,list) ):
            uGr = toArray(uGr)            
        if not isinstance(uGr, np.ndarray): # if not true by now ...
            raise TypeError(('variable uGr may either be an integer, a list '
                             'or a ndarray'))                            
    
        if (xGr.size > 0 and
            np.any(np.unique(xGr) != np.array(range(int(np.amax(xGr))+1)))):
            print('sugroup IDs for X: ')
            print( np.unique(xGr) )
            raise Exception(('subgrouping of latent variables X does not '
                             'meet the requirements. Please only use sub'
                             'group IDs 0, 1, ..., #groups'))
        if (yGr.size > 0 and
            np.any(np.unique(yGr) != np.array(range(int(np.amax(yGr))+1)))):
            print('sugroup IDs for Y: ')
            print( np.unique(yGr) )
            raise Exception(('subgrouping of output variables Y does not '
                             'meet the requirements. Please only use sub'
                             'group IDs 0, 1, ..., #groups'))
        if (uGr.size > 0 and
            np.any(np.unique(uGr) != np.array(range(int(np.amax(uGr))+1)))):
            print('sugroup IDs for U: ')
            print( np.unique(uGr) )
            raise Exception(('subgrouping of input variables U does not '
                             'meet the requirements. Please only use sub'
                             'group IDs 0, 1, ..., #groups'))
                            
        self._xDim = xGr.size                
        self._yDim = yGr.size                
        self._uDim = uGr.size                
                
        self._xGr = xGr # subgrouping  
        self._yGr = yGr # indices for
        self._uGr = uGr # X, Y and U 
                            
        if self._xDim > 0:
            self._xNumGr = int(np.amax(xGr) + 1)
        else:
            self._xNumGr = 1
        if self._yDim > 0:            
            self._yNumGr = int(np.amax(yGr) + 1)
        else:
            self._yNumGr = 1
        if self._uDim > 0:
            self._uNumGr = int(np.amax(uGr) + 1)
        else:
            self._uNumGr = 1
                            
        self._xGrSize = np.zeros([self._xNumGr], dtype=int)
        self._yGrSize = np.zeros([self._yNumGr], dtype=int)
        self._uGrSize = np.zeros([self._uNumGr], dtype=int)
                                        
        self._xGrIdx = np.zeros(self._xNumGr,dtype=np.ndarray)     
        i = 0
        while i < self._xNumGr: # number of subgroups for X
            if xGr.size > 0:
                tmp = np.where(xGr==i)[0] # .where returns i-th row, j-th column,           
                self._xGrIdx[i] = tmp    # we only care for rows in .where().[0]
                self._xGrSize[i] = np.size(tmp)
            else:
                self._xGrIdx[i] = np.array([],dtype=int)
                self._xGrSize[i] = 0
            i += 1

        self._yGrIdx = np.zeros(self._yNumGr,dtype=np.ndarray)     
        i = 0
        while i < self._yNumGr: # number of subgroups for Y
            if yGr.size > 0:
                tmp = np.where(yGr==i)[0].tolist() # .where returns i-th row, j-th column,           
                self._yGrIdx[i] = tmp    # we only care for rows in .where().[0]
                self._yGrSize[i] = np.size(tmp)
            else:
                self._yGrIdx[i] = np.array([],dtype=int)
                self._yGrSize[i] = 0
            i += 1

        self._uGrIdx = np.zeros(self._uNumGr,dtype=np.ndarray)     
        i = 0
        while i < self._uNumGr: # number of subgroups for U
            if uGr.size > 0:
                tmp = np.where(uGr==i)[0] # .where returns i-th row, j-th column,           
                self._uGrIdx[i] = tmp    # we only care for rows in .where().[0]
                self._uGrSize[i] = np.size(tmp)
            else:
                self._uGrIdx[i] = np.array([],dtype=int)
                self._uGrSize[i] = 0
            i += 1
            
        if self._xGrSize.sum() != self._xDim:   
            raise Exception(('Error calling timeSeriesModelVariables. '
                             'The number of elements in all subgroups '
                             'of X does not cover all components of X.'))
                            
        if self._yGrSize.sum() != self._yDim:   
            raise Exception(('Error calling timeSeriesModelVariables. '
                             'The number of elements in all subgroups '
                             'of Y does not cover all components of Y.'))

        if self._uGrSize.sum() != self._uDim:   
            raise Exception(('Error calling timeSeriesModelVariables. '
                             'The number of elements in all subgroups '
                             'of U does not cover all components of U.'))
                
        # start checking dependency table on level of U, X Y as variable groups
                            
        if deps is None: # initialize some default values
            if dtsX is None:
                self._dtsX = [0]
            elif isinstance(dtsX, (numbers.Integral,list,np.ndarray)):
                self._dtsX = np.unique(toArray(dtsX))            
            else:
                raise Exception(('provided temporal dependency scope of X does'
                                 ' not meet the requirements. Has to be int, '
                                 'list or ndarray'))                
            if dtsY is None:
                self._dtsY = [0]
            elif isinstance(dtsY, (numbers.Integral,list,np.ndarray)):
                self._dtsY = np.unique(toArray(dtsY))            
            else:                
                raise Exception(('provided temporal dependency scope of Y does'
                                 ' not meet the requirements. Has to be int, '
                                 'list or ndarray'))                
            if dtsU is None:
                self._dtsU = [0]
            elif isinstance(dtsU, (numbers.Integral,list,np.ndarray)):
                self._dtsU = np.unique(toArray(dtsU))            
            else:
                raise Exception(('provided temporal dependency scope of U does'
                                 ' not meet the requirements. Has to be int, '
                                 'list or ndarray'))                
                
            # prune the set of all involved relative time offsets that X,Y,U 
            # depend on to those that are actually relevant/appearing :
            self._dts = np.arange(min(min(self._dtsX), 
                                      min(self._dtsY), 
                                      min(self._dtsU)),
                                  max(max(self._dtsX), 
                                      max(self._dtsY), 
                                      max(self._dtsU))+1)            
            deps = []
            i = 0
            while i < self._dts.size:
                deps.append( np.array(
                   [np.zeros([self._xNumGr,self._xNumGr],dtype=bool),  # X|X
                    np.zeros([self._yNumGr,self._xNumGr],dtype=bool),  # X|Y
                    np.zeros([self._uNumGr,self._xNumGr],dtype=bool),  # X|U
                    np.zeros([self._xNumGr,self._yNumGr],dtype=bool),  # Y|X
                    np.zeros([self._yNumGr,self._yNumGr],dtype=bool),  # Y|Y
                    np.zeros([self._uNumGr,self._yNumGr],dtype=bool)], # Y|U
                    ).reshape([2,3]).transpose([1,0]))
                i += 1
            
        self.checkVarDependencies(deps) # check if meets all requirements 
                    
        self._deps = deps                            

        # (over-)write temporal variable dependency scopes:
        
        self._dtsX = []
        self._dtsY = []
        self._dtsU = []
        
        i = 0
        while i < len(deps): # = self._dts.size
            if np.any(deps[i][0,0]) or np.any(deps[i][0,1]):
                self._dtsX.append(self._dts[i])
            if np.any(deps[i][1,0]) or np.any(deps[i][1,1]):
                self._dtsY.append(self._dts[i])
            if np.any(deps[i][2,0]) or np.any(deps[i][2,1]):
                self._dtsU.append(self._dts[i])
            i += 1
            
        self._dtsX = toArray(self._dtsX)
        self._dtsY = toArray(self._dtsU)
        self._dtsU = toArray(self._dtsU)
                                
                  

    def giveVarDims(cls, xyu='xyu'):
        """ OUT = .giveVarDims(xyu):
        xyu: ordered subset of x, y and u, e.g. "yu", "x", "xyu"
        OUT: if len(xyu)==1, then returns dimensionality of X,Y or U.
             If len(xyu) >1, returns an array with selected dims. 
        Returns the dimensionalities of variables X,Y,U. Can be called 
        to return this information only for a specified subset of X,Y,U.  

        """
        try:
            tmp = {'xyu' : np.array([cls._xDim, cls._yDim, cls._uDim]),
                   'xy' :  np.array([cls._xDim, cls._yDim           ]),
                   'xu' :  np.array([cls._xDim,            cls._uDim]),
                   'yu' :  np.array([           cls._yDim, cls._uDim]),
                   'x' :             cls._xDim                        ,
                   'y' :                        cls._yDim             ,
                   'u' :                                   cls._uDim
                  }[xyu]
        except:
            raise Exception(('argument "xyu" has to be an ordered subset '
                           'of x, y and u, e.g. "yu", "x", "xyu"'))        
        return tmp

    def giveVarSubgroupTallies(cls, xyu='xyu'):
        """ OUT = .giveVarSubgroupTallies(xyu):
        xyu: ordered subset of x, y and u, e.g. "yu", "x", "xyu"
        OUT: if len(xyu)==1, then returns number of subgroups of X,Y or U.
             If len(xyu) >1, returns an array with numbers of subgroups of  
                             selected subset of X,Y,U. 
        Returns the numbers of subgroups of variables X,Y,U. Can be called 
        to return this information only for a specified subset of X,Y,U. 

        """
        
        try:
            tmp = {'xyu' : np.array([cls._xNumGr, cls._yNumGr, cls._uNumGr]),
                   'xy' :  np.array([cls._xNumGr, cls._yNumGr             ]),
                   'xu' :  np.array([cls._xNumGr,              cls._uNumGr]),
                   'yu' :  np.array([             cls._yNumGr, cls._uNumGr]),
                   'x' :             cls._xNumGr                            ,
                   'y' :                          cls._yNumGr               ,
                   'u' :                                       cls._uNumGr
                  }[xyu]
        except:
            raise Exception(('argument "xyu" has to be an ordered subset '
                           'of x, y and u, e.g. "yu", "x", "xyu"'))        
        return tmp        

    def giveVarSubgroupSizes(cls, xyu='xyu'):
        """ OUT = .giveVarSubgroupSizes(xyu):
        xyu: ordered subset of x, y and u, e.g. "yu", "x", "xyu"
        OUT: if len(xyu)==1, then returns subgroup sizes of X,Y or U.
             If len(xyu) >1, returns an array with subgroups sizes of a  
                             selected subset of X,Y,U. 
        Returns the sizes of subgroups of variables X,Y,U. Can be called 
        to return this information only for a specified subset of X,Y,U. 

        """        
        try:
            tmp = {
                'xyu' : np.array([cls._xGrSize, cls._yGrSize, cls._uGrSize]),
                'xy' :  np.array([cls._xGrSize, cls._yGrSize              ]),
                'xu' :  np.array([cls._xGrSize,               cls._uGrSize]),
                'yu' :  np.array([              cls._yGrSize, cls._uGrSize]),
                'x' :             cls._xGrSize                              ,
                'y' :                           cls._yGrSize                ,
                'u' :                                         cls._uGrSize
                  }[xyu]
        except:
            raise Exception(('argument "xyu" has to be an ordered subset '
                           'of x, y and u, e.g. "yu", "x", "xyu"'))        
        return tmp       
        
    def giveVarSubgroupLabels(cls, xyu='xyu'):
        """ OUT = .giveVarSubgroupLabels(xyu):
        xyu: ordered subset of x, y and u, e.g. "yu", "x", "xyu"
        OUT: if len(xyu)==1, then returns subgroup labels of X,Y or U.
             If len(xyu) >1, returns an array with subgroups lables of a  
                             selected subset of X,Y,U. 
        Returns the subgroups labels for all individual components of the 
        variables X,Y,U. Can be called to return this information only for 
        a specified subset of X,Y,U. 

        """        
        try:
            tmp = {'xyu' : np.array([cls._xGr, cls._yGr, cls._uGr]),
                   'xy' :  np.array([cls._xGr, cls._yGr          ]),
                   'xu' :  np.array([cls._xGr,           cls._uGr]),
                   'yu' :  np.array([          cls._yGr, cls._uGr]),
                   'x' :             cls._xGr                      ,
                   'y' :                       cls._yGr            ,
                   'u' :                                 cls._uGr
                  }[xyu]
        except:
            raise Exception(('argument "xyu" has to be an ordered subset '
                           'of x, y and u, e.g. "yu", "x", "xyu"'))        
        return tmp.copy()        
       
    def giveVarSubgroupIndices(cls, xyu='xyu'):
        """ OUT = .giveVarSubgroupIndices(xyu):
        xyu: ordered subset of x, y and u, e.g. "yu", "x", "xyu"
        OUT: if len(xyu)==1, then returns subgroup indices of X,Y or U.
             If len(xyu) >1, returns an array with subgroups indices of a  
                             selected subset of X,Y,U. 
        Returns the indices of subgroups of variables X,Y,U. Can be called 
        to return this information only for a specified subset of X,Y,U. 

        """        
        try:
            tmp = {'xyu' : np.array([cls._xGrIdx, cls._yGrIdx, cls._uGrIdx]),
                   'xy' :  np.array([cls._xGrIdx, cls._yGrIdx             ]),
                   'xu' :  np.array([cls._xGrIdx,              cls._uGrIdx]),
                   'yu' :  np.array([             cls._yGrIdx, cls._uGrIdx]),
                   'x' :             cls._xGrIdx                            ,
                   'y' :                          cls._yGrIdx               ,
                   'u' :                                       cls._uGrIdx
                  }[xyu]
        except:
            raise Exception(('argument "xyu" has to be an ordered subset '
                           'of x, y and u, e.g. "yu", "x", "xyu"'))        
        return tmp.copy()        
    
    def giveVarTimeScope(cls, xyu = None):
        """ OUT = .giveVarTimeScopes(xyu):
        xyu: ordered subset of x, y and u, e.g. "yu", "x", "xyu"
        OUT: if xyu is None, returns all relevant time offsets of the model
             if len(xyu)==1, then returns relevant time offsets of X,Y or U.
             If len(xyu) >1, returns an array with relevant time offsets of a  
                             selected subset of X,Y,U. 
        Returns the relevant time offsets of variables X,Y,U. Can be called 
        to return this information only for a specified subset of X,Y,U, or
        to return all the time offsets relevant to the model (usually the
        union). 

        """        
        if xyu is None:
            tmp = cls._dts
        else:
            try:
                tmp = {'xyu' : np.array([cls._dtsX, cls._dtsY, cls._dtsU]),
                       'xy' :  np.array([cls._dtsX, cls._dtsY           ]),
                       'xu' :  np.array([cls._dtsX,            cls._dtsU]),
                       'yu' :  np.array([           cls._dtsY, cls._dtsU]),
                       'x' :             cls._dtsX                        ,
                       'y' :                        cls._dtsY             ,
                       'u' :                                   cls._dtsU
                      }[xyu]
            except:
                raise Exception(('argument "xyu" has to be an ordered '
                            'subset of x, y and u, e.g. "yu", "x", "xyu"'))        
        
        return tmp.copy()
    
    def giveVarDependencies(cls, dts = None):
        """ OUT = .giveVarDependencies(dts):
        dts: list of time offsets for which the dependencies are desired
        OUT: list of arrays. The list contains 2D binary arrays specifying
             the dependency structure of subgroups of X,Y on subgroups of 
             X,Y and U at different time offsets dt. 
        Returns the variable dependencies between (subgroups of) X,Y,U at 
        selected time offsets. 

        """
        if dts is None:
            dts = []
            i = 0
            while i < cls._dts.size:
                dts.append(cls._dts[i])
                i += 1
        elif isinstance(dts,(numbers.Integral,float)) and dts in cls._dts:
            dts = [int(dts)]
        elif isinstance(dts,(numbers.Integral,float)) and not dts in cls._dts:
            raise Exception('selected time offset dt out of stored bounds')
        elif (isinstance(dts,(list,np.ndarray))
              and not np.all([idx in cls._dts for idx in dts])):
            print('desired time offsets dt: ')
            print(dts)
            print('represented time offsets dt of the model: ')
            print(cls._dts)
            raise Exception('queried for dt out of stored bounds')
        elif not isinstance(dts,(list,np.ndarray)):   
            raise Exception(('time offset index set "dts" has to be either '
                             'int, float or a list/ndarray of indexes. '
                             'It is neither'))
        
        return [cls._deps[np.where(cls._dts == i)[0]] for i in dts].copy()
    
    def checkVarDependencies(cls, deps=None):
        """ .checkVarDependencies(deps)
        deps: list of arrays. The list contains 2D binary arrays specifying
              the dependency structure of subgroups of X,Y on subgroups of 
              X,Y and U at different time offsets dt. 
        Checks the consistency of dependencies among subgroups of X,Y,U.
        Checks e.g. whether the model describes an acyclic graph or not (the
        latter preventing basic ancestral sampling).
        Can be called without arguments (will check the 'deps' variable of the
        current timeSeriesVariables() object) or with a specific deps argument

        """
        
        if deps is None:
            deps = cls._deps
        
        if not isinstance(deps, list):
            raise Exception(('variable dependency description "deps" '
                                 'has to be a list of arrays (one for each'
                                 ' relative time offset dt)'))    
            
        if len(deps) != cls._dts.size:    
            print(' len(deps): ')
            print( len(deps) )
            print(' temporal offsets dts: ')
            print( cls._dts )
            raise Exception(('ariable dependency description "deps" does '
                             'not meet the provded relative time offsets '
                             'dt. Cannot register which relative '
                             'time-steps X and Y actually depend on'))
            
        # now start checking dependency tables for individual subgroups       
        
        i = 1
        while i < cls._dts.size:
            
            if deps[i].shape == (2,3): # now deps[i] is ndarray
                deps = deps.transpose((1,0,2))
                print(('conditional dependency variables "deps" must have '
                       'shape [3,2]. Provided deps[dt] has shape [2,3], '
                       'taking transpose'))
            elif not deps[i].shape == (3,2):            
                print('deps[dt].shape is: ')      
                print(deps[i].shape)      
                raise Exception(('conditional dependency variables "deps" ' 
                            'must have shape = [3,2].'))
                
            if   not (deps[i][0,0].shape ==                    # case X|X
                                (cls._xNumGr,          # subgroups of X
                                 cls._xNumGr)):        # subgroups of X
                print('for the i-th relative time step dT, with dT =')
                print(cls._dts[i])
                print('deps[dt][0,0] (giving X|X ) is:')
                print(deps[i][0,0])
                print('of shape')
                print(deps[i][0,0].shape)
                raise Exception(('conditional dependency variables "deps" ill-' 
                            'specifies X|X - does not match # of subgroups.'))
                
            elif not (deps[i][1,0].shape ==                    # case X|Y
                                (cls._yNumGr,          # subgroups of Y
                                 cls._xNumGr)):        # subgroups of X
                print('for the i-th relative time step dT, with dT =')
                print(cls._dts[i])
                print('deps[dt][1,0] (giving X|Y ) is:')
                print(deps[i][1,0])
                print('of shape')
                print(deps[i][1,0].shape)
                raise Exception(('conditional dependency variables "deps" ill-' 
                            'specifies X|Y - does not match # of subgroups.'))

            elif not (deps[i][2,0].shape ==                    # case X|U
                                (cls._uNumGr,          # subgroups of U
                                 cls._xNumGr)):        # subgroups of X
                print('for the i-th relative time step dT, with dT =')
                print(cls._dts[i])
                print('deps[dt][2,0] (giving X|U ) is:')
                print(deps[i][2,0])
                print('of shape')
                print(deps[i][2,0].shape)
                raise Exception(('conditional dependency variables "deps" ill-' 
                            'specifies X|U - does not match # of subgroups.'))
                
            elif not (deps[i][0,1].shape ==                    # case Y|X
                                (cls._xNumGr,          # subgroups of X
                                 cls._yNumGr)):        # subgroups of Y
                print('for the i-th relative time step dT, with dT =')
                print(cls._dts[i])
                print('deps[dt][0,1] (giving Y|X ) is:')
                print(deps[i][0,1])
                print('of shape')
                print(deps[i][0,1].shape)
                raise Exception(('conditional dependency variables "deps" ill-' 
                            'specifies Y|X - does not match # of subgroups.'))

            elif not (deps[i][1,1].shape ==                    # case Y|Y
                                (cls._yNumGr,          # subgroups of Y
                                 cls._yNumGr)):        # subgroups of Y
                print('for the i-th relative time step dT, with dT =')
                print(cls._dts[i])
                print('deps[dt][1,1] (giving Y|Y ) is:')
                print(deps[i][1,1])
                print('of shape')
                print(deps[i][1,1].shape)
                raise Exception(('conditional dependency variables "deps" ill-' 
                            'specifies Y|Y - does not match # of subgroups.'))
                
            elif not (deps[i][2,1].shape ==                    # case Y|U
                                (cls._uNumGr,          # subgroups of U
                                 cls._yNumGr)):        # subgroups of Y
                print('for the i-th relative time step dT, with dT =')
                print(cls._dts[i])
                print('deps[dt]2,1] (giving Y|U ) is:')
                print(deps[i][2,1])
                print('of shape')
                print(deps[i][2,1].shape)
                raise Exception(('conditional dependency variables "deps" ill-' 
                            'specifies Y|U - does not match # of subgroups.'))

            elif (not np.all(    # check if all low-level entries are boolean
                             [deps[i][0,0].dtype.type==np.bool_,   
                              deps[i][1,0].dtype.type==np.bool_,  
                              deps[i][2,0].dtype.type==np.bool_, 
                              deps[i][0,1].dtype.type==np.bool_, 
                              deps[i][1,1].dtype.type==np.bool_,
                              deps[i][2,1].dtype.type==np.bool_]
                            )         ):
                raise Exception(('all entries of conditional dependency '
                                 'variables "deps" have to be boolean. '
                                 'This is not the case.'))
            i += 1
        
    
#----this -------is ------the -------79 -----char ----compa rison---- ------bar

class stateSpaceModelFactorization:
    """description of dependency structure between all the model variables
    
    State-space models over {X,Y} with potentially additional input variables U
    are defined by (conditional) probability distribution of some subgroup of
    variables U, X, Y given another. The full probability of X,Y|U factors into
    these conditional distributions. stateSpaceModelFactorization objects 
    describe these dependencies, analaguous to graphical model representations.
     
    On top of acting as a state-space model description, they keep track of all
    the model components and help ensure consistency of computations. This 
    comprises ordering the factors in a way that supports ancestral sampling. 
    
    Note that in general state-space models with the Markov property, the 
    number of factors grows at least with the data size T * Trials. 
    stateSpaceModelFactor objects are not meant to represent each individual 
    factor in the factorization of p(X,Y,U), but only the unique templates.
    If the distribution of each individual variable (a given component of X,Y
    or U at any given time) is desired to be described with a factor, then 
    pointers should be used. This allows to save memory (for homogeneous
    LDSs or HMMs, all Y|X over all time points 1,...,T could point to the 
    same stateSpaceModelFactor object), while retaining flexibility (each 
    pointer for each time point could in fact point to its own 
    stateSpaceModelFactor instantiation to model changes of Y|X over time). 
    
    INPUTS to constructor:
    model:             pointer to object that called the constructor. 
                       Possibly is 'None', but usually a timeSeriesModel()
    linkFunctions:     list of lists of strings specifying the types and
                       parameters of the deterministic parts of the condi-
                       tional distr.s linking subgroups of X, Y and U
    noiseDistrs:       list of lists of strings specifying the types and
                       parameters of the random parts of the conditional
                       distributionss linking subgroups of X, Y and U
    noiseInteractions: list of lists of strings specifying how the random
                       and deterministic parts of the conditional distri-
                       butions interact. E.g. '+', 'multiplicative', '^'
    initDistrs:        list of lists of lists of strings specifying the
                       probability distributions of the intial time steps.
                       Length of outermost list has to match dts.size    
                       
    VARIABLES and METHODS:
    .objDescr                   - always is 'state_space_model_factorization'
    .setFactorDetermPart()      - sets the link function of a cond. distr.
    .setFactorRandomPart()      - sets the noise distr. of a cond. distr.
    .setFactorInitialVar()      - sets the noise distr. of an initial distr.
    .updateFactorDetermPart()   - update parameters of a link function
    .updateFactorRandomPart()   - update parameters of a noise distribution
    .updateFactorInitDistr()    - update parameters of an initial distribution
    .giveLinkFunctionList()     - gives list (of lists) of all link functions 
    .giveNoiseDistrList()       - gives list (of lists) of all noise distr.s
    .giveInitialDistrList()     - gives list (of lists) of all initial distr.s
    .giveConditionalLists()     - gives list (of lists) of all variable 
                                  subgroups that the cond. distr.s depend on
    .giveInitVarTimeScope()     - gives time offsets for initial distr.s
    .giveSamplingHierarchy()    - gives suitable subgroup ordering for sampling
    .giveInitialVarList()       - gives list of variable subgroups that need
                                  their intial distribution specified
    _getCondsFromDeps()         - gets lists of cond. distr. conditionals 
                                  from variable dependencies stored in 'deps'
    _getInDims()                - get dimensionalities of link function inputs 
    _getDefaultLinkFunction()   - initializes link functions to default 
    _getVarSubgroupGenerations()- establishes variable subgroup hierarchy
    _getHierarchyFromGens()     - establishes ordering from subgroup hierarchy
    _getInitialVarSubgroups()   - get list of variable subgroups that need
                                  their intial distribution specified
    ._checkFactorizationHomogeneous() - checks consistency of factorization
    _checkIfOwnFather()         - checks if a given graph is acyiclic
    .giveMotherObject()         - returns timeSeriesModel() object
    
    """
    objDescr = 'state_space_model_factorization'

    def __init__(self, 
                 model,          # 'mother' object, holds model.varDescr 
                 linkFunctions,  # deterministic part of cond. distributions
                 noiseDistrs,    # stochastic part of cond. distributions
                 noiseInteractions=None,
                 initDistrs=None):   
                 
        self._tsmodel = model # allows acces to variable description object
        
        varDescr = self._tsmodel.giveVarDescr()             
        deps     = varDescr.giveVarDependencies()
        
        if model.giveIfHomogeneous(): # simple case: can work with a single
                                      # template that is repeated over time
        
            # compute list of conditional variables for quicker referencing
            self._conds = [[], # list for conditional vars for p(X_t | U, X, Y)
                           []] # list for conditional vars for p(Y_t | U, X, Y)                
            dts                = varDescr.giveVarTimeScope()
            xyuSubgroupTallies = varDescr.giveVarSubgroupTallies()
        
            i=0                    # first iterate over latent state subgroups
            while i < xyuSubgroupTallies[0]: 
                self._conds[0].append(self._getCondsFromDeps(
                                               deps, # returns a list of   
                                               0,    # triplets of format:  
                                               i,    #  [dt, xyu,
                                               dts,  #  subGroupLabel]
                                               xyuSubgroupTallies)
                                     ) 
                i += 1                
            i=0                  # then iterate over observed output subgroups
            while i < xyuSubgroupTallies[1]: 
                self._conds[1].append(self._getCondsFromDeps(
                                               deps, # returns a list of   
                                               1,    # triplets of format:  
                                               i,    #  [dt, xyu,
                                               dts,  #  subGroupLabel]
                                               xyuSubgroupTallies)
                                     ) 
                i += 1

        
            # check argument linkFunctions
            xyuSubgroupSizes = varDescr.giveVarSubgroupSizes()
            if linkFunctions is None:
                linkFunctions = [[],[]]
                i = 0
                while i < xyuSubgroupTallies[0]:
                    inDims = self._getInDims(0,i,self._conds,xyuSubgroupSizes)
                    linkFunctions[0].append(
                                     self._getDefaultLinkFunction(inDims)
                                           )
                    i += 1
                i = 0
                while i < xyuSubgroupTallies[1]:
                    inDims = self._getInDims(1,i,self._conds,xyuSubgroupSizes)
                    linkFunctions[1].append(
                                     self._getDefaultLinkFunction(inDims)
                                           )
                    i += 1
                                            

            if not isinstance(linkFunctions, list):
                raise Exception('argument linkFunctions has to be a list')
            elif len(linkFunctions) != 2:
                raise Exception(('at the top level, argument linkFunctions '
                                 'has to have length 2: One entry each for '
                                 'link functions of variable groups X and '
                                 'Y'))
            elif (not isinstance(linkFunctions[0],list) or 
                  not isinstance(linkFunctions[1],list)):
                raise Exception(('argument linkFunction has to be a list '
                                 'containing two lists. Each of these two '
                                 'lists has to contain the link functions '
                                 'of variables X or Y, respectively'))
                
            if len(linkFunctions[0]) == 1: 
                i = 0
                while i < xyuSubgroupTallies[0]:
                    linkFunctions[0].append(linkFunctions[0][0])
                    i += 1            
            elif len(linkFunctions[0]) != xyuSubgroupTallies[0]:
                raise Exception(('number of provided link functions for '
                                 'variables X does not meet number of '
                                 'subgroups of X. Need one link function '
                                 'per subgroup. If only one is given, it '
                                 'will be applied to all subgroups'))
            if len(linkFunctions[1]) == 1: 
                i = 0
                while i < xyuSubgroupTallies[1]:
                    linkFunctions[1].append(linkFunctions[1][0])
                    i += 1            
            elif len(linkFunctions[1]) != xyuSubgroupTallies[1]:
                raise Exception(('number of provided link functions for '
                                 'variables Y does not meet number of '
                                 'subgroups of Y. Need one link function '
                                 'per subgroup. If only one is given, it '
                                 'will be applied to all subgroups'))
            
            # linkFunctions seems okay, now apply it: 
            self._determ = [[],[]]
            i = 0
            while i < xyuSubgroupTallies[0]:
                inDims = self._getInDims(0,i,self._conds,xyuSubgroupSizes)
                self._determ[0].append(linkFunction(
                                                linkFunctions[0][i], 
                                                inDims, 
                                                xyuSubgroupSizes[0][i]
                                                   ))
                i += 1
            i = 0
            while i < xyuSubgroupTallies[1]:
                inDims = self._getInDims(1,i,self._conds,xyuSubgroupSizes)
                self._determ[1].append(linkFunction(
                                                linkFunctions[1][i], 
                                                inDims, 
                                                xyuSubgroupSizes[1][i]
                                                   ))
                i += 1
                
            # now check noiseDistrs
            
            if noiseDistrs is None:
                noiseDistrs = [[],[]]
                i = 0
                while i < xyuSubgroupTallies[0]:
                    noiseDistrs[0].append('Gaussian')
                    i += 1
                i = 0
                while i < xyuSubgroupTallies[1]:
                    noiseDistrs[1].append('Gaussian')
                    i += 1

                
            if not isinstance(noiseDistrs, list):
                raise Exception('argument noiseDistrs has to be a list')
            elif len(noiseDistrs) != 2:
                raise Exception(('at the top level, argument noiseDistrs '
                                 'has to have length 2: One entry each for '
                                 'noise distributions of variable groups '
                                 'X and Y'))
            elif (not isinstance(noiseDistrs[0],list) or 
                  not isinstance(noiseDistrs[1],list)):
                raise Exception(('argument noiseDistrs has to be a list '
                                 'containing two lists. Each of these two '
                                 'lists has to contain the noise distribu'
                                 'tions of variables X or Y, respectively'))
                
            if len(noiseDistrs[0]) == 1: 
                i = 0
                while i < xyuSubgroupTallies[0]:
                    noiseDistrs[0].append(noiseDistrs[0][0])
                    i += 1            
            elif len(noiseDistrs[0]) != xyuSubgroupTallies[0]:
                print(noiseDistrs[0])
                print(varDescr.giveVarSubgroupTallies())
                raise Exception(('number of provided noise distrib. for '
                                 'variables X does not meet number of '
                                 'subgroups of X. Need one noise distr. '
                                 'per subgroup. If only one is given, it '
                                 'will be applied to all subgroups'))
            if len(noiseDistrs[1]) == 1: 
                i = 0
                while i < xyuSubgroupTallies[1]:
                    noiseDistrs[1].append(noiseDistrs[1][0])
                    i += 1            
            elif len(noiseDistrs[1]) != xyuSubgroupTallies[1]:
                print(noiseDistrs[1])
                print(varDescr.giveVarSubgroupTallies())
                raise Exception(('number of provided noise distrib. for '
                                 'variables Y does not meet number of '
                                 'subgroups of Y. Need one noise distr. '
                                 'per subgroup. If only one is given, it '
                                 'will be applied to all subgroups'))            
            
            
            if noiseInteractions is None:
                noiseInteractions = [[],[]]
                i = 0
                while i < xyuSubgroupTallies[0]:
                    noiseInteractions[0].append('+')
                    i += 1
                i = 0
                while i < xyuSubgroupTallies[1]:
                    noiseInteractions[1].append('+')
                    i += 1
            
            if not isinstance(noiseInteractions, list):
                raise Exception('noiseInteractions has to be a list')
            elif len(noiseInteractions) != 2:
                raise Exception(('at the top level, noiseInteractions '
                                 'has to have length 2: One entry each for '
                                 'noise distributions of variable groups '
                                 'X and Y'))
            elif (not isinstance(noiseInteractions[0],list) or 
                  not isinstance(noiseInteractions[1],list)):
                raise Exception(('noiseInteractions has to be a list '
                                 'containing two lists. Each of these two '
                                 'lists has to define the noise interaction'
                                 'with variables X or Y, respectively'))
                
            if len(noiseInteractions[0]) == 1: 
                i = 0
                while i < xyuSubgroupTallies[0]:
                    noiseInteractions[0].append(noiseInteractions[0][0])
                    i += 1            
            elif len(noiseInteractions[0]) != xyuSubgroupTallies[0]:
                print(noiseInteractions[0])
                print(varDescr.giveVarSubgroupTallies())
                raise Exception(('number of provided noise interact. for '
                                 'variables X does not meet number of '
                                 'subgroups of X. Need one noise intera. '
                                 'per subgroup. If only one is given, it '
                                 'will be applied to all subgroups'))
            if len(noiseInteractions[1]) == 1: 
                i = 0
                while i < xyuSubgroupTallies[1]:
                    noiseInteractions[1].append(noiseInteractions[1])
                    i += 1            
            elif len(noiseInteractions[1]) != xyuSubgroupTallies[1]:
                print(noiseInteractions[1])
                print(varDescr.giveVarSubgroupTallies())
                raise Exception(('number of provided noise interact. for '
                                 'variables Y does not meet number of '
                                 'subgroups of Y. Need one noise intera. '
                                 'per subgroup. If only one is given, it '
                                 'will be applied to all subgroups'))   
                
            # noiseDistrs and noiseInteractions seem okay, now apply them: 
            self._random = [[],[]]
            i = 0
            while i < xyuSubgroupTallies[0]:
                self._random[0].append(noiseDistr(noiseDistrs[0][i], 
                                                  noiseInteractions[0][i],
                                                  xyuSubgroupSizes[0][i]))
                i += 1
            i = 0
            while i < xyuSubgroupTallies[1]:
                self._random[1].append(noiseDistr(noiseDistrs[1][i],  
                                                  noiseInteractions[1][i],
                                                  xyuSubgroupSizes[1][i]))
                i += 1                
                
                
            self._checkFactorizationHomogeneous(self._conds)        
            
            # compute ordering of var subgroups that allows ancestral sampling
            dts = varDescr.giveVarTimeScope()
            dt0 = np.where(dts==0)[0][0]
            if not 0 in dts: # no dependency of any X,Y on current time
                self._generation = [[],[]]
                i = 0
                while i < xyuSubgroupTallies[0]:
                    gens[0].append(0) # put all subgroups in first generation
                    i += 1
                i = 0
                while i < xyuSubgroupTallies[1]:
                    gens[1].append(0) # put all subgroups in first generation
                    i += 1                 
            else:
                self._generation = self._getVarSubgroupGenerations(
                                            deps[dt0], # check only for dt=0
                                            xyuSubgroupTallies
                                                                  )
                
            self._hierarchy = self._getHierarchyFromGens(self._generation,
                                                         xyuSubgroupTallies)
            
            
            # now check Markov chain initial segment distributions
            
            tmp = stateSpaceModelFactorization._getInitialVarSubgroups(
                                                  deps,
                                                  dts,
                                                  xyuSubgroupTallies
                                                                       )
            self._initVars = tmp[0]
            self._dtsInit  = tmp[1]
            del tmp
            
            if initDistrs is None:
                initDistrs = []
                for dti in range(len(self._dtsInit)):
                    initDistrs.append( [ [],[] ] )
                    i = 0
                    while i < xyuSubgroupTallies[0]:
                        if i in self._initVars[dti][0]:
                            initDistrs[dti][0].append('Gaussian')
                        else :
                            initDistrs[dti][0].append('none')
                        i += 1
                    i = 0
                    while i < xyuSubgroupTallies[1]:
                        if i in self._initVars[dti][1]:
                            initDistrs[dti][1].append('Gaussian')
                        else :
                            initDistrs[dti][1].append('none')
                        i += 1

            for dti in range(len(self._dtsInit)):
                if not isinstance(initDistrs[dti], list):
                    raise Exception('argument initDistrs has to be a list')
                elif len(initDistrs[dti]) != 2:
                    print('i =')
                    print(i)
                    raise Exception(('initial distributions both for '
                                     'subgroups of X and Y are  needed. '
                                     'Thus argument initDistrs[i] has to '
                                     'have length 2 for each i: One entry '
                                     'each for initial distributions of '
                                     'variable groups X and Y'))
                elif (not isinstance(initDistrs[dti][0],list) or 
                      not isinstance(initDistrs[dti][1],list)):
                    print('i =')
                    print(i)
                    raise Exception(('argument initDistrs[i] has to be a '
                                     'list containing two lists. Each of '
                                     'these two lists has to contain the '
                                     'initial distributions of variables '
                                     'X or Y, respectively'))
                
                if (len(initDistrs[dti][0])==1 and 
                    self._initVars[dti][0].size>0): 
                    i = 0
                    while i < xyuSubgroupTallies[0]:
                        if i in self._initVars[dti][0]:
                            initDistrs[dti][0].append(initDistrs[dti][0][0])
                        else: 
                            initDistrs[dti][0].append(None)
                        i += 1            
                elif len(initDistrs[dti][0]) != xyuSubgroupTallies[0]:
                    print('i =')
                    print(i)
                    print(initDistrs[dti][0])
                    print(varDescr.giveVarSubgroupTallies())
                    raise Exception(('initial distrib. for variables X does '
                                     'not meet the formatting conventions. '
                                     'initDistrs[i][0] has to be a list with '
                                     'one element per subgroup of X. Each '
                                     'element has to specify a valid '
                                     'probability distribution or be None '
                                     '(in case this subgroup does not need '
                                     'an initial distribution)'))
                elif (sum([(not i is None) for i in initDistrs[dti][0]]) 
                       != self._initVars[dti][0].size):
                    print('i =')
                    print(i)
                    print(initDistrs[dti][0])
                    print(self._initVars[dti][0])
                    raise Exception(('number of provided initial distrib. for '
                                     'variables X does not meet number of '
                                     'subgroups of X that require an initial. '
                                     'distribution  If only one is given, it '
                                     'will be applied to all subgroups that'
                                     'require initialization'))

                if (len(initDistrs[dti][1]) == 1 and 
                    self._initVars[dti][1].size > 0): 
                    i = 0
                    while i < xyuSubgroupTallies[1]:
                        if i in self._initVars[dti][0]:
                            initDistrs[dti][1].append(initDistrs[dti][1][0])
                        else: 
                            initDistrs[dti][1].append(None)
                        i += 1            
                elif len(initDistrs[dti][1]) != xyuSubgroupTallies[1]:
                    print('i =')
                    print(i)
                    print(initDistrs[dti][1])
                    print(varDescr.giveVarSubgroupTallies())
                    raise Exception(('initial distrib. for variables Y does '
                                     'not meet the formatting conventions. '
                                     'initDistrs[i][0] has to be a list with '
                                     'one element per subgroup of Y. Each '
                                     'element has to specify a valid '
                                     'probability distribution or be None '
                                     '(in case this subgroup does not need '
                                     'an initial distribution)'))
                elif (sum([(not i is None) for i in initDistrs[dti][1]]) 
                       != self._initVars[dti][1].size):
                    print('i =')
                    print(i)
                    print(initDistrs[dti][1])
                    print(self._initVars[dti][1])
                    raise Exception(('number of provided initial distrib. for '
                                     'variables Y does not meet number of '
                                     'subgroups of Y that require an initial. '
                                     'distribution  If only one is given, it '
                                     'will be applied to all subgroups that'
                                     'require initialization'))
            
            self._init = []
            for dti in range(self._dtsInit.size):
                self._init.append([[],[]])
                i = 0
                while i < xyuSubgroupTallies[0]:
                    if i in self._initVars[dti][0]:
                        self._init[dti][0].append(initDistr(
                                                   initDistrs[dti][0][i], 
                                                   xyuSubgroupSizes[0][i]
                                                           ) )
                    else:
                        self._init[dti][0].append(None)
                    i += 1            
                i = 0
                while i < xyuSubgroupTallies[1]:
                    if i in self._initVars[dti][1]:
                        self._init[dti][1].append(initDistr(
                                                   initDistrs[dti][1][i], 
                                                   xyuSubgroupSizes[1][i]
                                                           ) )
                    else:
                        self._init[dti][1].append(None)
                    i += 1

                
        else: # i.e. if state-space model is inhomogeneous
            raise Exception(('Trying to construct model factorization for '
                             'inhomogeneous state-space models. Support for '
                             'inhomogeneous models not yet implemented.'))
            self._factors = [ [ [],    # will have to be a list of factors each
                                [] ] ] # for X resp. Y, wrapped in another list
                                       # that captures evolving time ...
                
    def setFactorDetermPart(cls, 
                            varGroup, varSubgroup, 
                            linkFun):
        """ .setFactorDetermPart(varGroup*, varSubgroup*,linkFun*)
        varGroup:    int giving the variable group, i.e. '0' for X, '1' for Y
        varSubgroup: int giving the index of the variable subgroup
        linkFun:     string specifying the link function type, e.g. 'linear'
        Sets the deterministic part of a conditional distribution (also termed
        link function) to a specific pre-stored function type. 

        """
        if isinstance(linkFun,linkFunction):
            cls._determ[varGroup, varSubgroup] = linkFun
        elif isinstance(linkFun, str):
            varDescr = cls._tsobject.giveVarDescr()
            xyuSubgroupsizes = varDescr.giveVarSubgroupSizes()
            inDims = _getInDims(varGroup,
                                varSubgroup,
                                cls._conds,
                                xyuSubGroupSizes)
            cls._determ[varGroup, varSubgroup] = linkFunction(
                             linkFun, 
                             inDims, 
                             xyuSubgroupSizes[varGroup][varSubgroup]
                                                             )
        else:
            print('selected linkFunction to be set:')
            print('varGroup:')
            print(varGroup)
            print('varSubgroup:')
            print(varSubgroup)
            print('selected link function:')
            print(linkFun)
            raise Exception(('setting the deterministic part of a factor of a '
                             'state-space model requires either a linkFunction'
                             ' object or a string that denotes the desired '
                             'type of linkFunction object to be created'))
        
    def setFactorRandomPart(cls,                             
                            varGroup, varSubgroup, 
                            noiseDis, noiseInt=None):
        """ .setFactorRandomPart(varGroup*, varSubgroup*, noiseDis*, noiseInt)
        varGroup:    int giving the variable group, i.e. '0' for X, '1' for Y
        varSubgroup: int giving the index of the variable subgroup
        noiseDis:    string specifying the noise distr. type, e.g. 'Gaussian'
        noiseInt:    string specifying the noise interaction type, e.g. '+'
        Sets the random part of a conditional distribution (also termed noise
        distribution) to a specific pre-stored distribution type. 

        """
        if noiseInt is None:
            noiseInt = '+' # default: additive noise 
                
        if isinstance(noiseDis,noiseDistr):
            cls._random[varGroup, varSubgroup] = noiseDis
        elif isinstance(noiseDis, str):
            varDescr = cls._tsobject.giveVarDescr()        
            dims = varDescr.giveSubgroupSizes()[varGroup][varSubgroup]
            cls._random[varGroup, varSubgroup] = noiseDistr(noiseDis, 
                                                            noiseInt, 
                                                            dims)
        else:
            print('selected noiseDistr to be set:')
            print('varGroup:')
            print(varGroup)
            print('varSubgroup:')
            print(varSubgroup)
            print('selected noise distribution:')
            print(noiseDis)
            raise Exception(('setting the random part of a factor of a state-'
                             'space model requires either a noiseDistr object '
                             'or a string that denotes the desired type of '
                             'noiseDistr object to be created'))

    def setFactorInitialVar(cls,                             
                            varGroup, varSubgroup, 
                            initDis):
        """ .setFactorInitialVar(varGroup*, varSubgroup*, initDis*)
        varGroup:    int giving the variable group, i.e. '0' for X, '1' for Y
        varSubgroup: int giving the index of the variable subgroup
        initDis:     string specifying the noise distr. type, e.g. 'Gaussian'
        Sets an initial distribution to a specific pre-stored distribution type. 

        """                
        if isinstance(initDis,initDisr):
            cls._init[varGroup, varSubgroup] = initDis
        elif isinstance(initDis, str):
            varDescr = cls._tsobject.giveVarDescr()        
            dims = varDescr.giveSubgroupSizes()[varGroup][varSubgroup]
            cls._init[varGroup, varSubgroup] = initDistr(initDis, 
                                                         dims)
        else:
            print('selected initDistr to be set:')
            print('varGroup:')
            print(varGroup)
            print('varSubgroup:')
            print(varSubgroup)
            print('selected initial distribution:')
            print(initDis)
            raise Exception(('setting an initial distribution for a state-'
                             'space model requires either a initDistr object '
                             'or a string that denotes the desired type of '
                             'initDistr object to be created'))
                        
    def updateFactorDetermPart(cls, varGroup, varSubgroup, pars):
        """ .updateFactorDetermPart(varGroup*, varSubgroup*,pars*)
        varGroup:    int giving the variable group, i.e. '0' for X, '1' for Y
        varSubgroup: int giving the index of the variable subgroup
        pars:        list of ndarrays giving the parameters of the link function
        Sets the parameters of the deterministic part of a conditional distr. 
        (also termed link function). 

        """        
        cls._determ[varGroup, varSubgroup].updatePars(pars)

    def updateFactorRandomPart(cls, varGroup, varSubgroup, pars):
        """ .updateFactorRandomPart(varGroup*, varSubgroup*,pars*)
        varGroup:    int giving the variable group, i.e. '0' for X, '1' for Y
        varSubgroup: int giving the index of the variable subgroup
        pars:        list of ndarrays giving the parameters of the noise distr.
        Sets the parameters of the random part of a conditional distr. 
        (also termed noise distribution). 

        """        
        cls._random[varGroup, varSubgroup].updatePars(pars)

    def updateFactorInitDistr( cls, varGroup, varSubgroup, pars):
        """ .updateFactorInitDistr(varGroup*, varSubgroup*,pars*)
        varGroup:    int giving the variable group, i.e. '0' for X, '1' for Y
        varSubgroup: int giving the index of the variable subgroup
        pars:        list of ndarrays giving the parameters of the init. distr.
        Sets the parameters of an initial distribution. 

        """                
        cls._init[  varGroup, varSubgroup].updatePars(pars)
        
    def giveLinkFunctionList(cls,xyu='xy'):
        """ OUT = .giveLinkFunctionList(xy):
        xy:  ordered subset of x, y, e.g. "y", "xy"
        OUT: if len(xy)==1, returns list of link functions
             If len(xy) >1, returns list of lists of link functions  
        Returns the link functions for cond. distr.s of X,Y. Can be called 
        to return this information only for a specified subset of X,Y. 

        """        
        tmp = { 'xy' : cls._determ,
                'x'  : cls._determ[0],
                'y'  : cls._determ[1]
               }[xyu]
        return tmp

    def giveNoiseDistrList(cls,xyu='xy'):
        """ OUT = .giveNoiseDistrList(xy):
        xy:  ordered subset of x, y, e.g. "y", "xy"
        OUT: if len(xy)==1, returns list of noise distributions
             If len(xy) >1, returns list of lists of noise distributions  
        Returns the noise distributions for cond. distr.s of X,Y. Can be  
        called to return this information only for a specified subset of X,Y. 

        """        
        tmp = { 'xy' : cls._random,
                'x'  : cls._random[0],
                'y'  : cls._random[1]
               }[xyu]
        return tmp

    def giveInitialDistrList(cls,xyu='xy'):
        """ OUT = .giveInitialDistrList(xy):
        xy:  ordered subset of x, y, e.g. "y", "xy"
        OUT: if len(xy)==1, returns list of lists of initial distributions
             If len(xy) >1, returns list of lists of lists of initial distr.s  
        Returns the initial distributions for cond. distr.s of X,Y. Can be  
        called to return this information only for a specified subset of X,Y.
        Note that there is an extra outer list for different relevant time 
        offsets.

        """                
        tmp = { 'xy' : cls._init,
                'x'  : [cls._init[dti][0] for dti in range(cls._dtsInit.size)],
                'y'  : [cls._init[dti][1] for dti in range(cls._dtsInit.size)]
               }[xyu]
        return tmp
    
    def giveConditionalLists(cls, xyu='xy'):
        """ OUT = .giveConditionalLists(xy):
        xy:  ordered subset of x, y, e.g. "y", "xy"
        OUT: if len(xy)==1, returns list of conditioned-on variable subgroups
             If len(xy) >1, returns list of lists conditioned-on subgroups  
        Returns the variable subgroups that the link functions of X,Y depend  
        on. Can be called to return this information only for a specified 
        subset of X,Y. 

        """                
        tmp = { 'xy' : cls._conds,
                'x'  : cls._conds[0],
                'y'  : cls._conds[1]
               }[xyu]
        return tmp.copy()    
    
    def giveInitVarTimeScope(cls):
        """ OUT = .giveInitVarTimeScope()
        OUT: array of relevant time steps that require initial distributions
        Returns array of the initial time steps for which the current model
        requires initial distributions.

        """
        return cls._dtsInit.copy()
    
    def giveSamplingHierarchy(cls):
        """ OUT = .giveSamplingHierarchy()
        OUT: list of variable group/subgroup pairs 
        Returns a list of length-2 lists (= [variable group, variable subgroup]).
        Their ordering constitutes a valid ancestral sampling ordering.

        """            
        return cls._hierarchy.copy()
    
    def giveInitialVarList(cls, xyu='xy'):
        """ OUT = .giveInitialVarList(xy):
        xy:  ordered subset of x, y, e.g. "y", "xy"
        OUT: if len(xy)==1, returns list of lists of variable subgroups that 
                            require an initial distribution
             If len(xy) >1, returns list of lists of lists of subgroups that
                            require an initial distribution
        Returns the variable subgroups of X,Y that require an initial dist.
        to be specified. Can be called to return this information only for a 
        specified subset of X,Y. 
        Note that there is an extra outer list for different relevant time 
        offsets.
    
        """                        
        tmp = { 'xy' : cls._initVars,
                'x'  : [cls._initVars[dti][0] for dti in range(cls._dtsInit.size)],
                'y'  : [cls._initVars[dti][1] for dti in range(cls._dtsInit.size)]
               }[xyu]
        return tmp.copy()
    
    def _checkFactorizationHomogeneous(cls, conds):        
        print('checking factorization of model ...')
        varDescr = cls._tsmodel.giveVarDescr()
        if not cls._tsmodel.giveIfCausal():
            print(('WARNING: Provided model is acausal. Could potentially '
                   'represent a cyclic directed graph'))
        i = 0
        while i < varDescr.giveVarSubgroupTallies('x'): # for each X subgroup 
            if cls._checkIfOwnFather(cls._conds, cls._conds[0][i], [0,0,i]):
                raise Exception(('checking if directed acyclic graph... ' 
                                 'Factorization describes a cyclic graph!'))                      
            i += 1
        i = 0
        while i < varDescr.giveVarSubgroupTallies('y'): # for each Y subgroup
            if cls._checkIfOwnFather(cls._conds, cls._conds[1][i], [0,1,i]):
                raise Exception(('checking if directed acyclic graph... ' 
                                 'Factorization describes a cyclic graph!'))                      
            i += 1
        print('... factorization (locally) describes directed acyclic graph')
            
    @staticmethod
    def _checkIfOwnFather(conds, parents, template):
        if parents == []:
            return False
        for x in parents: # for each subgroup of X ...
            if (x[0] == 0 # temporal offset dt = 0, i.e. same time step 
                and x == template):
                return True
            elif (x[0] == 0   # temporal offset dt = 0, i.e. same time step 
                  and x[1] in [0,1] # is variable X or Y (U never has parents) 
                  and stateSpaceModelFactorization._checkIfOwnFather(
                                                         conds, 
                                                         conds[x[1]][x[2]], 
                                                         template                 
                                                                    ) ):
                return True
            # else: we do not care for dt < 0 or dt > 0: no cycle possible
        return False # if did not return yet
                              
    @staticmethod
    def _getCondsFromDeps(deps, 
                          varGroup,
                          varSubgroup,
                          dts,
                          xyuSubgroupTallies):
        conds = []
        t = 0
        while t < dts.size: # loop over relative time offsets dt
            for xyu in [0,1,2]: # loop over X, Y and U variable groups
                otherSubgroup = 0  # loop over X/Y/U subgroups 
                while otherSubgroup < xyuSubgroupTallies[xyu]:                
                    if deps[t][xyu,varGroup][otherSubgroup,varSubgroup]: 
                        conds.append([dts[t], xyu, otherSubgroup])
                    # else: simply do not add to list of variables depended on
                    otherSubgroup += 1 
            t += 1
        return conds
    
        
    @staticmethod
    def _getInDims(varGroup,
                   varSubgroup,
                   conds,
                   xyuSubGroupSizes):
        ls = conds[varGroup][varSubgroup] # list of inputs
        inDims = []
        i = 0
        while i < len(ls):
            inDims.append(xyuSubGroupSizes[ ls[i][1] ][ ls[i][2] ])
            i += 1                        # varGroup   varSubgroup
        return inDims
            
    @staticmethod
    def _getDefaultLinkFunction(numInputs):
        linkFunction = { 0 : 'zeroFun',
                         1 : 'linear',
                         2 : 'linearTwoInputs',
                         3 : 'linearThreeInputs',
                         4 : 'linearFourInputs',
                         5 : 'linearFiveInputs'
                       }[numInputs]
        return linkFunction
                                          
    @staticmethod
    def _getVarSubgroupGenerations(deps,xyuSubgroupTallies,
                                   gens=None,currGen=0,toDo=None):
                
        if gens is None: # start out setting all generations to highest one
            maxGen = sum(xyuSubgroupTallies[0:2]) # total number var subgroups
            gens = [[],[]]
            i = 0
            while i < xyuSubgroupTallies[0]:
                gens[0].append(maxGen)
                i += 1
            i = 0
            while i < xyuSubgroupTallies[1]:
                gens[1].append(maxGen)
                i += 1 
                
        if toDo is None: # call with toDo = None for auto-initialization 
            toDo =  [list(range(xyuSubgroupTallies[0])),
                     list(range(xyuSubgroupTallies[1]))]
            
        uncond = [[], # subgroups of X,Y that do notcondition on any other
                  []] # subgroup that is still on the to-do list 
        for i in toDo[0]:
            if (not np.any([deps[0,0][j,i] for j in toDo[0]])                                
                           and 
                not np.any([deps[1,0][j,i] for j in toDo[1]])):                
                uncond[0].append(i)
                
        for i in toDo[1]:
            if (not np.any([deps[0,1][j,i] for j in toDo[0]])                
                           and 
                not np.any([deps[1,1][j,i] for j in toDo[1]])):                
                uncond[1].append(i)
                
        toDo[0] = list(set(toDo[0]) - set(uncond[0])) # remove from
        toDo[1] = list(set(toDo[1]) - set(uncond[1])) # to-do list
        
        i = 0                       # for those removed from to-do list in
        while i < len(uncond[0]):   # this iteration, add current gen
            gens[0][uncond[0][i]] = currGen
            i += 1
        i = 0        
        while i < len(uncond[1]):
            gens[1][uncond[1][i]] = currGen
            i += 1
        
        if toDo != [[],[]]: # if anything left in to-do list ...
            return stateSpaceModelFactorization._getVarSubgroupGenerations(
                                              deps, 
                                              xyuSubgroupTallies, 
                                              gens, 
                                              currGen+1, 
                                              toDo)
        else:
            return gens
        
    @staticmethod
    def _getHierarchyFromGens(gens,xyuSubgroupTallies):
        hierarchy = []
        i = 0
        currGen = 0
        while i < sum(xyuSubgroupTallies[0:2]):
            j = 0
            while j < xyuSubgroupTallies[0]:
                if gens[0][j] == currGen:
                    hierarchy.append(np.array([0,j]))
                    i += 1
                j += 1
            j = 0
            while j < xyuSubgroupTallies[1]:
                if gens[1][j] == currGen:
                    hierarchy.append(np.array([1,j]))
                    i += 1
                j += 1            
            currGen += 1
            
        return hierarchy
    
    @staticmethod
    def _getInitialVarSubgroups(deps,dts,xyuSubgroupTallies):
        initVars = []
        dtsInit  = []
        for dti in range(dts.size): # for each varSubgroup that depends on an            
            if dts[dti] < 0:        # earlier time step ...
                initVars.append([[],  # lists of tuples [varGroup,varSubgroup], 
                                 []]) # initVars[0] for X, initVars[1] for Y    
                dtsInit.append(dts[dti])
                i = 0
                while i < xyuSubgroupTallies[0]: # does this X subgroup ...
                    if (np.any(deps[dti][0,0][:,i]) or # depend on earlier X?
                        np.any(deps[dti][1,0][:,i]) or # depend on earlier Y?
                        np.any(deps[dti][2,0][:,i]) or # depend on earlier U?
                        np.any([i in initVars[j][0] for j in range(dti)])):
                        initVars[dti][0].append(i) # ... add to the list 
                    i += 1
                i = 0
                while i < xyuSubgroupTallies[1]: # does this Y subgroup ...
                    if (np.any(deps[dti][0,1][:,i]) or # depend on earlier X?
                        np.any(deps[dti][1,1][:,i]) or # depend on earlier Y?
                        np.any(deps[dti][2,1][:,i]) or # depend on earlier U?
                        np.any([i in initVars[j][1] for j in range(dti)])):
                        initVars[dti][1].append(i) # ... add to the list
                    i += 1
                initVars[dti][0] = np.unique(initVars[dti][0]) # could have  
                initVars[dti][1] = np.unique(initVars[dti][1]) # repetitions
        dtsInit = toArray(dtsInit)
        
        return [initVars, # list of length len(dts), each entry contains list 
                dtsInit]  # of length two which each contains an array
                                                     
    def giveMotherObject(cls):
        """OUT = .returnMotherObject()
        OUT: object that called the constructor for this object (possibly None)
        """
        return cls._tsmodel
                                                  
        
#----this -------is ------the -------79 -----char ----compa rison---- ------bar
        
class noiseDistr:
    """noise description for variables X, Y, Z of a state-space model
    
    State-space models over {X,Y} with potentially additional input variables U
    are defined by (conditional) probability distribution of some subgroup of
    variables U, X, Y given another. The full probability of X,Y|U factors into
    these conditional distributions. stateSpaceModelFactor objects each 
    describe one of these dependencies, such as p(Y | X, U) or p(X_t | X_{t-1}).
        
    Objects of this class serve to encode and store the noise that gives the 
    uncertainty involved in those distributions, e.g. specifying that 
    p(Y | X) = C*X + chol(R) * N(0, I), i.e. with Gaussian noise N(0, I)
    
    INPUTS to constructor:
    distrType:        string specifying selected noise distribution type
    noiseInteraction: string specifying selected noise interaction type
    dims:             dimensionality of random variables to be represented
    pars:             parameters for this noise distr.
    
    VARIABLES and METHODS:
    .objDescr                  - always is 'state_space_model_noiseDistr'
    .supportedNoiseTypes       - list of strings giving supported noise types
    .supportedInteractionTypes - list of strings giving supported noise 
                                 interaction types
    .setNoiseInteraction()     - sets the noise interaction type
    .updatePars()              - updates parameters for this noise distr.
    .givePars()                - returns parameters for this noise distr.
    .giveDistrType()           - returns string specifying noise distribution
    .giveNoiseInteraction()    - returns string specifying noise interaction
    ._transfNoiseSeed()        - transforms pre-seeded noise to match the
                                 distribution of this noise distr.
    
    """
    objDescr = 'state_space_model_noiseDistr'
     
    supportedNoiseTypes       = ['none',
                                 'gaussian', 'Gaussian', 'normal']
                                 

    supportedInteractionTypes = ['+', 'additive', 
                                 '*', 'multiplicative', 
                                 '^', 'power']
        
    def __init__(self, 
                 distrType='none',
                 noiseInteraction='+',
                 dims=1,
                 pars=None):
        
        if distrType in self.supportedNoiseTypes:
            self._distrType        = distrType
        else:
            print('selected noise distribution type:')
            print(distrType)
            print('supported noise distribution types:')
            print(self.supportedNoiseTypes)
            raise Exception('selected noise distribution not supported')
        
        self.setNoiseInteraction(noiseInteraction)
        
        if isinstance(dims,numbers.Integral) and dims > 0:
            self._dims = dims
        elif isinstance(dims,float) and dims > 0:
            self._dims = int(dims)
        else: 
            raise Exception(('dimensionality argument for noiseDistr object '
                             'has to be a positive integer'))
            
        # need to work on a databank of supported distributions...    
        if self._distrType == 'none':
            self._pars = np.array([])
        elif self._distrType in ['gaussian', 'Gaussian', 'normal']:
            self._pars = np.zeros(3,dtype=np.ndarray)
            self._pars[0] = np.zeros(dims)
            self._pars[1] = np.identity(dims)
            # for computational quickness, we will also store the  
            # cholesky decomposition of the covariance:
            self._pars[2] = np.linalg.cholesky(self._pars[1])
        
        # update right away if possible:
        if not (pars is None): # if called with parameters
            if not isinstance(pars, np.ndarray): # if only one parameter,
                pars = toArray(pars)             # required the list 
            self.updatePars(pars)                # brackets may be omitted
            
            
    def giveDistrType(cls):
        """ OUT = .giveDistrType()
        OUT: string specifying the current noise distribution type

        """        
        return cls._distrType
        
    def giveNoiseInteraction(cls):
        """ OUT = .giveNoiseInteraction()
        OUT: string specifying the current noise interaction type

        """        
        return cls._noiseInteraction

    def givePars(cls, whichPars=None):
        """ OUT = .givePars(whichPars)
        whichPars: list of indices of parameters to return
        OUT:       ndarray of parameters of this noise distr.
        Returns pointer to the parameters of this noise distribution. Allows
        choosing specific parameters by handing over a list of indices. 
        Ordering of parameters is fixed and best seen from the code body of
        the method. whichPars = None  returns all parameters
        Example for 'Gaussian' noise distr:
        It is pars[0] = mu, pars[1] = Sigma, pars[2] = chol(Sigma)
        whichPars = [0,2] returns mu and chol(Sigma)

        """         
        if whichPars is None:
            whichPars = np.arange(len(cls._pars))
        elif isinstance(whichPars, (numbers.Integral,float)):
            whichPars = [ int(whichPars) ] # allow calling with just integer
        
        try:
            return [cls._pars[i] for i in whichPars].copy()
        except:
            print('selected parameters to return:')
            print(whichPars)
            print('actual parameters of given noiseDistr:')
            print(cls._pars)
            raise Exception(('error returning selected pars. Most likely '
                             'asked for list of parameters that do not '
                             'all exist.'))
                      
    def setNoiseInteraction(cls, noiseInteraction):
        """ .setNoiseInteraction(noiseInteraction*)
        noiseInteraction: string specifying selected noise interaction type
        Sets the noise interaction type to the given value. Note that
        only certain noise interaction types are allowed.
        """        
        if noiseInteraction in cls.supportedInteractionTypes:
            cls._noiseInteraction = {
                        '+' :              np.add,
                        'additive' :       np.add,
                        '*' :              np.multiply,
                        'multiplicative' : np.multiply,
                        '^' :              np.power,
                        'power' :          np.power 
                                    }[noiseInteraction]
        else:
            print('selected noise interaction type:')
            print(noiseInteraction)
            print('supported interaction types:')
            print(cls.supportedInteractionTypes)
            raise Exception('selected noise interaction not supported')
        
        
    def updatePars(cls, pars, idxParsIn = None):
        """ .updatePars(pars*, idxParsIn)
        pars:      list of parameters for this noise distr. ndarrays will
                   be interpreted as a list of length one, i.e. as the 
                   first parameter of the distribution
        idxParsIn: index set specifying which parameters are handed over
        Sets the parameters of this noise distribution. The parameters 
        for any distribution type are ordered, and the parameters have 
        to be given in the right ordering. The argument idxParsIn allows
        to hand over only specific parameters by telling which parameters
        they are supposed to update. pars and idxParsIn need to have
        same length.
        Example for 'Gaussian' noise distr:
        It is pars[0] = mu, pars[1] = Sigma, pars[2] = chol(Sigma)        
        Handing over only newPars = [np.random.uniform(size=[dims,dims])
        will result in an error, as pars[0] is a vector, not a matrix.
        Handing over newPars and idxParsIn=[1] will overwrite Sigma, 
        idxParsIn=[2] will overwrite chol(Sigma). 
        The method only checks for correct dimensionalities of the new
        parameters, not for contenxt, e.g. if Sigma truly is pos.def. !
        
        """                         
        if isinstance(pars,list):
            numParsIn = len(pars)
        elif isinstance(pars,np.ndarray):
            pars = [pars] # interpreting as
            numParsIn = 1 # single parameter
        else:
            raise Exception('argument pars has to be a list of parameters.') 

        if idxParsIn is None:
            idxParsIn = list(range(numParsIn))
        elif isinstance(idxParsIn,numbers.Integral):
            idxParsIn = [idxParsIn]
        elif isinstance(idxParsIn,np.ndarray):
            idxParsIn = idxParsIn.tolist() # lists simpler to work with
        elif not isinstance(idxParsIn,list):
            print('idxParsIn:')
            print(idxParsIn)
            raise Exception(('argument idxParsIn can be None, an integer, '
                             'a list or an array. It is none of these.'))
            
        if cls._distrType == 'none':
            raise Exception(('void noise distribution does not have '
                             'parameters to update'))        
        elif cls._distrType in ['normal', 'gaussian', 'Gaussian']: 
            numPars = 3
            if numParsIn>numPars:
                raise Exception(('this noise distrib object models Gaussian '
                                 'noise. Need to provide a list of arguemnts '
                                 'of length three: mean, (co-)variance and '
                                 'cholesky decompostition of covariance.'))
            if 1 in idxParsIn and not 2 in idxParsIn: # when giving new Sigma, 
                                                      # but no new chol(Sigma)  
                pars.append(np.linalg.cholesky(pars[1]))
                idxParsIn.append(2)                                        
                numParsIn += 1
                            
        if not np.all([i in range(numPars) for i in idxParsIn]):
            print('idxParsIn:')
            print(idxParsIn)
            print('range of IDs of stored parameters:')
            print(range(numPars))
            raise Exception(('when handing over argument idxParsIn, '
                             'make sure that the indexed parameters '
                             'you wish to update also exist.'))
        elif len(idxParsIn)!=numParsIn:
            print('idxParsIn:')
            print(idxParsIn)
            print('number of parameters handed over:')
            print(numParsIn)
            raise Exception(('when handing over argument idxParsIn, '
                             'make sure that a matching number of '
                             'parameters is handed over'))                
                
        i = 0
        while i < numParsIn:
            if pars[idxParsIn[i]] is None:  
                i += 1          # convention: do not update and continue
            elif not isinstance(pars[idxParsIn[i]],np.ndarray):
                print('trying to update pars[i] for i=')
                print(idxParsIn[i])
                print('selected new pars[i] is')
                print(pars[idxParsIn[i]])
                raise Exception('convention: all parameters are ndarrays.')
            elif pars[idxParsIn[i]].shape==cls._pars[idxParsIn[i]].shape:
                cls._pars[idxParsIn[i]] = pars[idxParsIn[i]].copy()
                i += 1
            else:
                print('trying to update pars[i] for i=')
                print(idxParsIn[i])
                print('selected new pars[i] has shape')
                print(pars[idxParsIn[i]].shape)
                print('stored pars[i] has shape:')
                print(cls._pars[idxParsIn[i]].shape)
                raise Exception(('shapes do not match. Parameters that shall '
                                 'not be updated can be handed over as None'))

                
    def _transfNoiseSeed(cls, xyuData):
        """ OUT = ._transfNoiseSeed(xyuData)
        xyuData: ndarray of data with seeded noise
        OUT:     same array as xyuData with transformed noise
        Transforms pre-seeded standard noise to match the distribution of 
        this noisDistr() object. For example shifts and re-forms Gaussian 
        white noise to have correct mean and covariance structure
        
        """
        if len(xyuData.shape) > 2:
            print('data for noise transformation has shape:')
            print(xyuData.shape)
            raise Exception(('this method can only work with 2D data arrays. '
                             'If you wish to have 3+ dimensional data, split '
                             'the data e.g. along trials and hand these '
                             'arrays over individually.'))
                          
        if xyuData.shape[0] != cls._dims:
            print('data for noise transformation has shape:')
            print(xyuData.shape)
            print('i.e. data dimensionality should be')
            print(xyuData.shape[0])
            print('data dimensionality for this noise distribution is:')
            print(cls._dims)
            raise Exception('dimensionality mismatch!')
                             
        if cls._distrType == 'none':                     
            return xyuData
        elif cls._distrType in ['normal', 'gaussian', 'Gaussian']:
            return np.dot(cls._pars[2],xyuData) + cls._pars[0] 
            
        
#----this -------is ------the -------79 -----char ----compa rison---- ------bar
        
class initDistr:
    """ description of initial distribution for variables f a state-space model
    
    State-space models over {X,Y} with potentially additional input variables U
    are defined by (conditional) probability distribution of some subgroup of
    variables U, X, Y given another. The full probability of X,Y|U factors into
    these conditional distributions, plus the distributions that describe the
    state of the variables at the very beginning of the time series. 
    initDistr objects each describe one of these initial distributions, such
    as p(X_t) for t = 1.
        
    Objects of this class serve to encode and store the noise that gives the 
    uncertainty involved in those distributions, e.g. specifying that 
    p(X_1) = mu_0 + chol(V_0) * N(0, I), i.e. with Gaussian noise N(0, I)

    INPUTS to constructor:
    distrType:        string specifying selected noise distribution type
    dims:             dimensionality of random variables to be represented
    pars:             parameters for this initial distr.
    
    VARIABLES and METHODS:
    .objDescr                  - always is 'state_space_model_initDistr'
    .supportedNoiseTypes       - list of strings giving supported noise types
    .drawSample()              - generates samples from this initial distr.
    .updatePars()              - updates parameters for this initial distr.
    .givePars()                - returns parameters for this initial distr.
    .giveDistrType()           - returns string specifying initial distribution
    """
    objDescr = 'state_space_model_initDistr'
     
    supportedNoiseTypes       = ['none',
                                 'gaussian', 'Gaussian', 'normal']
        
    def __init__(self, 
                 distrType='none',
                 dims=1,
                 pars=None):
        
        if distrType in self.supportedNoiseTypes:
            self._distrType        = distrType
        else:
            print('selected initial distribution type:')
            print(distrType)
            print('supported initial distribution types:')
            print(self.supportedNoiseTypes)
            raise Exception('selected initial distribution not supported')
        
        if isinstance(dims,numbers.Integral) and dims > 0:
            self._dims = dims
        elif isinstance(dims,float) and dims > 0:
            self._dims = int(dims)
        else: 
            raise Exception(('dimensionality argument for initDistr object '
                             'has to be a positive integer'))
            
        # need to work on a databank of supported distributions...    
        if self._distrType == 'none':
            self._pars = np.array([])
        elif self._distrType in ['gaussian', 'Gaussian', 'normal']:
            self._pars = np.zeros(3,dtype=np.ndarray)
            self._pars[0] = np.zeros(dims)
            self._pars[1] = np.identity(dims)
            # for computational quickness, we will also store the  
            # cholesky decomposition of the covariance:
            self._pars[2] = np.linalg.cholesky(self._pars[1])
        
        # update right away if possible:
        if not (pars is None): # if called with parameters
            if not isinstance(pars, np.ndarray): # if only one parameter,
                pars = toArray(pars)             # required the list 
            self.updatePars(pars)                # brackets may be omitted
            
    def drawSample(cls, numSamples=1):
        """ OUT = drawSample(numSamples)
        numSamples: int specifying the desired number of samples
        OUT:        array of drawn samples
        Generates samples from an intial distribution. 
        
        """        
        if not isinstance(numSamples,numbers.Integral):
            print('numSamples:')
            print(numSamples)
            raise Exception('argument numSamples has to be an integer')
        elif numSamples <= 0:
            print('numSamples:')
            print(numSamples)
            raise Exception('argument numSamples has to be positive')

        if cls._distrType == 'none':
            return np.zeros(cls._dims)
        elif cls._distrType in ['normal', 'gaussian', 'Gaussian']: 
            return cls._pars[0] + np.dot(cls._pars[2],
                                         np.random.normal(size=[cls._dims]))
            
    def giveDistrType(cls):
        """ OUT = .giveDistrType()
        OUT: string specifying the current noise distribution type

        """                
        return cls._distrType
        
    def givePars(cls, whichPars=None):
        """ OUT = .givePars(whichPars)
        whichPars: list of indices of parameters to return
        OUT:       ndarray of parameters of this noise distr.
        Returns pointer to the parameters of this noise distribution. Allows
        choosing specific parameters by handing over a list of indices. 
        Ordering of parameters is fixed and best seen from the code body of
        the method. whichPars = None  returns all parameters
        Example for 'Gaussian' noise distr:
        It is pars[0] = mu, pars[1] = Sigma, pars[2] = chol(Sigma)
        whichPars = [0,2] returns mu and chol(Sigma)

        """                 
        if whichPars is None:
            whichPars = np.arange(len(cls._pars))
        elif isinstance(whichPars, (numbers.Integral,float)):
            whichPars = [ int(whichPars) ] # allow calling with just integer
        
        try:
            return [cls._pars[i] for i in whichPars].copy()
        except:
            print('selected parameters to return:')
            print(whichPars)
            print('actual parameters of given noiseDistr:')
            print(cls._pars)
            raise Exception(('error returning selected pars. Most likely '
                             'asked for list of parameters that do not '
                             'all exist.'))        
        
    def updatePars(cls, pars, idxParsIn = None):
        """ .updatePars(pars*, idxParsIn)
        pars:      list of parameters for this noise distr. ndarrays will
                   be interpreted as a list of length one, i.e. as the 
                   first parameter of the distribution
        idxParsIn: index set specifying which parameters are handed over
        Sets the parameters of this noise distribution. The parameters 
        for any distribution type are ordered, and the parameters have 
        to be given in the right ordering. The argument idxParsIn allows
        to hand over only specific parameters by telling which parameters
        they are supposed to update. pars and idxParsIn need to have
        same length.
        Example for 'Gaussian' noise distr:
        It is pars[0] = mu, pars[1] = Sigma, pars[2] = chol(Sigma)        
        Handing over only newPars = [np.random.uniform(size=[dims,dims])]
        will result in an error, as pars[0] is a vector, not a matrix.
        Handing over newPars and idxParsIn=[1] will overwrite Sigma, 
        idxParsIn=[2] will overwrite chol(Sigma). 
        The method only checks for correct dimensionalities of the new
        parameters, not for content, e.g. if Sigma truly is pos.def. !
        
        """                         
        if isinstance(pars,list):
            numParsIn = len(pars)
        elif isinstance(pars,np.ndarray):
            pars = [pars] # interpreting as
            numParsIn = 1 # single parameter
        else:
            raise Exception('argument pars has to be a list of parameters.') 

        if idxParsIn is None:
            idxParsIn = list(range(numParsIn))
        elif isinstance(idxParsIn,numbers.Integral):
            idxParsIn = [idxParsIn]
        elif isinstance(idxParsIn,np.ndarray):
            idxParsIn = idxParsIn.tolist() # lists simpler to work with
        elif not isinstance(idxParsIn,list):
            print('idxParsIn:')
            print(idxParsIn)
            raise Exception(('argument idxParsIn can be None, an integer, '
                             'a list or an array. It is none of these.'))
            
        if cls._distrType == 'none':
            raise Exception(('void noise distribution does not have '
                             'parameters to update'))        
        elif cls._distrType in ['normal', 'gaussian', 'Gaussian']: 
            numPars = 3
            if numParsIn>numPars:
                raise Exception(('this noise distrib object models Gaussian '
                                 'noise. Need to provide a list of arguemnts '
                                 'of length three: mean, (co-)variance and '
                                 'cholesky decompostition of covariance.'))
            if 1 in idxParsIn and not 2 in idxParsIn: # when giving new Sigma, 
                                                      # but no new chol(Sigma)  
                pars.append(np.linalg.cholesky(pars[1]))
                idxParsIn.append(2)                                        
                numParsIn += 1
                            
        if not np.all([i in range(numPars) for i in idxParsIn]):
            print('idxParsIn:')
            print(idxParsIn)
            print('range of IDs of stored parameters:')
            print(range(numPars))
            raise Exception(('when handing over argument idxParsIn, '
                             'make sure that the indexed parameters '
                             'you wish to update also exist.'))
        elif len(idxParsIn)!=numParsIn:
            print('idxParsIn:')
            print(idxParsIn)
            print('number of parameters handed over:')
            print(numParsIn)
            raise Exception(('when handing over argument idxParsIn, '
                             'make sure that a matching number of '
                             'parameters is handed over'))                
                
        i = 0
        while i < numParsIn:
            if pars[idxParsIn[i]] is None:  
                i += 1          # convention: do not update and continue
            elif not isinstance(pars[idxParsIn[i]],np.ndarray):
                print('trying to update pars[i] for i=')
                print(idxParsIn[i])
                print('selected new pars[i] is')
                print(pars[idxParsIn[i]])
                raise Exception('convention: all parameters are ndarrays.')
            elif pars[idxParsIn[i]].shape==cls._pars[idxParsIn[i]].shape:
                cls._pars[idxParsIn[i]] = pars[idxParsIn[i]].copy()
                i += 1
            else:
                print('trying to update pars[i] for i=')
                print(idxParsIn[i])
                print('selected new pars[i] has shape')
                print(pars[idxParsIn[i]].shape)
                print('stored pars[i] has shape:')
                print(cls._pars[idxParsIn[i]].shape)
                raise Exception(('shapes do not match. Parameters that shall '
                                 'not be updated can be handed over as None'))
        
        
#----this -------is ------the -------79 -----char ----compa rison---- ------bar
    
class linkFunction:
    """link function description for variables X, Y, Z of a state-space model
    
    State-space models over {X,Y} with potentially additional input variables U
    are defined by (conditional) probability distribution of some subgroup of
    variables U, X, Y given another. The full probability of X,Y|U factors into
    these conditional distributions. stateSpaceModelFactor objects each 
    describe one of these dependencies, such as p(Y | X, U) or p(X_t | X_{t-1}).
        
    Objects of this class serve to encode and store the deterministic part that 
    the conditional probabilities can often be described with, e.g. in the
    Gaussian case where p(Y|X) = N(Y | C*X, R), it is equivalently
    Y ~ f(X) + e, where f(X) = C*X and p(e) = N(e | 0, R). 
    In simple cases such as the above example, the link funciton f is linear,  
    but more complicated cases such as artificial neural networks are 
    also possible.
    
    INPUTS to constructor:
    functionType: string specifying the link function type, e.g. 'linear'
    inDims:       list of dimensionality(s) of input(s) to this link function
    outDims:      dimensionality of output of this link function
    pars:         parameters for this initial distr.
    
    VARIABLES and METHODS:
    .objDescr               - always is 'state_space_model_linkFunction'
    .supportedFunctionTypes - list of string giving supported function types
    .computeValue()         - evalues the link function given inputs
    .giveFunctionType()     - returns string specifying the link function type
    .givePars()             - returns parameters of this link function
    .updatePars()           - updates parameters of this link function
    
    """
    objDescr = 'state_space_model_linkFunction'
    
    supportedFunctionTypes = ['zeroFun', 'identity', 
                              'linear', 'linearTwoInputs',
                              'linearThreeInputs', 'linearFourInputs',
                              'linearFiveInputs',
                              'affineLinear', 'affineLinearTwoInputs']
    
    def __init__(self, 
                 functionType='identity',
                 inDims=[1],
                 outDims=1, 
                 pars=None):
        
        if functionType in self.supportedFunctionTypes:
            self._functionType = functionType
        else:
            print('selected link function type:')
            print(functionType)
            print('supported link function types:')
            print(self.supportedFunctionTypes)
            raise Exception('selected function type not supported')
        
        if isinstance(inDims,(numbers.Integral)) and inDims > 0:
            self._inDims = toArray(inDims)                                 
        elif isinstance(inDims, list): # superficial input checking ...
            self._inDims = toArray(inDims)
        else: 
            print('inDims:')
            print(inDims)
            raise Exception(('input dimensionality argument for linkFunction '
                             'object has to be an ARRAY of positive integers. '
                             'For single-input link functions, input '
                             'dimensionality may be given as an integer'))
            
        if (not np.all(
                 [isinstance(i,(numbers.Integral,float)) for i in inDims]) 
             or np.any([   i <= 0                        for i in inDims]) ):
            print('inDims:')
            print(inDims)
            raise Exception(('input dimensionality argument for linkFunction '
                             'object has to be an array of POSITIVE INTEGERS'))
            
        if isinstance(outDims,numbers.Integral) and outDims > 0:
            self._outDims = outDims
        elif isinstance(outDims,float) and outDims > 0:
            self._outDims = int(outDims)
        elif (isinstance(outDims,np.ndarray) 
              and outDims.size==1 
              and outDims[0]>0):
            self._outDims = int(outDims[0])
        else: 
            raise Exception(('output dimensionality argument for linkFunction '
                             'object has to be a positive integer'))            
            
                                                                  
        # need to work on a databank of supported  distributions...    
        ssm = ssm_placeholder() # this might get exported in the future                                                    
                                 
        if self._functionType == 'identity':
            self._fun     = ssm.identity
            self._pars    = []            
                                 
        elif self._functionType == 'linear':                # f(x) = A*x
            numPars = 1
            if self._inDims.size != numPars :
                raise Exception(('linear link function only takes one input. '
                                 'linkFunction object was called with more '
                                 'than one specified input dimensionality'))
            self._fun     = ssm.linear
                                
        elif self._functionType == 'linearTwoInputs':       # f(x,u)=A*x+B*u
            numPars = 2
            if self._inDims.size != numPars :
                raise Exception(('linear link function with two inputs '
                                 'requires two arguments and hence two '
                                 'input dims'))
            self._fun     = ssm.linearTwoInputs            
                                
        elif self._functionType == 'linearThreeInputs':       # f(x,u)=A*x+B*u+F*z
            numPars = 3
            if self._inDims.size != numPars :
                raise Exception(('linear link function with three inputs ' 
                                 'requires three arguments and hence '
                                 'three input dims'))                                 
            self._fun     = ssm.linearThreeInputs
                                
        elif self._functionType == 'linearFourInputs':       # f(x,u)=A*x+B*u+F*z+G*s
            numPars = 4
            if self._inDims.size != numPars :
                raise Exception(('linear link function with four inputs ' 
                                 'requires four arguments and hence '
                                 'four input dims'))                                 
            self._fun     = ssm.linearFourInputs
                                
        elif self._functionType == 'linearFivenputs':       # f(x,u)=A*x+B*u+F*z+G*s+H*r
            numPars = 5
            if self._inDims.size != numPars :
                raise Exception(('linear link function with five inputs ' 
                                 'requires five arguments and hence '
                                 'five input dims'))                                 
            self._fun     = ssm.linearFiveInputs
                                
        elif self._functionType == 'affineLinear':          # f(x) = A*x+b
            numPars = 1
            if self._inDims.size != numPars :
                raise Exception(('affine linear link function only takes '
                                 'one input. linkFunction object was called '
                                 'with more than one specified input '
                                 'dimensionality'))
            self._fun     = ssm.affineLinear
                                
        elif self._functionType == 'affineLinearTwoInputs': # f(x,u)=A*x+B*u+b
            numPars = 2
            if self._inDims.size != numPars :
                raise Exception(('linear link function with input requires '
                                 'two arguments and hence two input dims'))                                 
            self._fun     = ssm.affineLinearTwoInputs
        
        # initialize parameters:
        self._pars    = np.zeros(numPars, dtype=np.ndarray)
        i = 0
        while i < numPars:
            if self._outDims == self._inDims[i]: # set some default: 
                self._pars[i] = np.identity(self._outDims)
            else:                                # set some default:
                self._pars[i] = np.zeros([self._outDims,self._inDims[i]])
            i += 1
        
        # update right away if possible:
        if not (pars is None): # if called with parameters
            if not isinstance(pars, np.ndarray): # if only one parameter,
                pars = toArray(pars)             # required the list 
            self.updatePars(pars)                # brackets may be omitted
                                   
    def computeValue(cls, x):
        """ OUT = .computeValue(x*)
        x:   (list of) inputs to link function 
        OUT: value of evaluated link function
        
        """
        return cls._fun(x, cls._pars)                                 
                                 
    def giveFunctionType(cls):
        """ OUT = .giveFunctionType()
        OUT: string specifying the current link function type

        """                        
        return cls._functionType
    
    def givePars(cls, whichPars=None):
        """ OUT = .givePars(whichPars)
        whichPars: list of indices of parameters to return
        OUT:       ndarray of parameters of this link function
        Returns pointer to the parameters of this link function. Allows
        choosing specific parameters by handing over a list of indices. 
        Ordering of parameters is fixed and best seen from the code body of
        the method. whichPars = None  returns all parameters
        Example for 'affineLinear' link function:
        It is pars[0] = A, pars[1] = b, y = Ax + b
        whichPars = [1] only returns parameter 'b'

        """                 
        if whichPars is None:
            whichPars = range(cls._pars.size)
        elif isinstance(whichPars, (numbers.Integral,float)):
            whichPars = [ int(whichPars) ] # allow calling with just integer
        
        try:
            return [cls._pars[i] for i in whichPars].copy()
        except:
            print('selected parameters to return:')
            print(whichPars)
            print('actual range of parameter IDs of given linkFunction:')
            print(range(cls._pars.size))
            raise Exception(('error returning selected pars. Most likely '
                             'asked for list of parameters that do not '
                             'all exist.'))
    
                                 
    def updatePars(cls, pars, idxParsIn=None):
        """ .updatePars(pars*, idxParsIn)
        pars:      list of parameters for this noise distr. ndarrays will
                   be interpreted as a list of length one, i.e. as the 
                   first parameter of the distribution
        idxParsIn: index set specifying which parameters are handed over
        Sets the parameters of this noise distribution. The parameters 
        for any distribution type are ordered, and the parameters have 
        to be given in the right ordering. The argument idxParsIn allows
        to hand over only specific parameters by telling which parameters
        they are supposed to update. pars and idxParsIn need to have
        same length.
        Example for 'affineLinear' link function:
        It is pars[0] = A, pars[1] = b, y = Ax + b
        Handing over only newPars = [np.ones([inDims[0]])]
        will result in an error, as pars[0] is a matrix, not a vector.
        Handing over newPars and idxParsIn=[1] will overwrite 'b' 
        The method only checks for correct dimensionalities of the new
        parameters, not for content, e.g. if 'b' contains any NaNs !
        
        """                         
        if isinstance(pars,list):
            numParsIn = len(pars)
        elif isinstance(pars,np.ndarray):
            pars = [pars] # interpreting as
            numParsIn = 1 # single parameter
        else:
            raise Exception('argument pars has to be a list of parameters.') 
            
        if cls._functionType == 'zeroFun':
            raise Exception('zero function has no parameters') 

        elif cls._functionType == 'identity':
            raise Exception('identity function has no parameters') 

        elif cls._functionType == 'linear':                # f(x) = A*x
            numPars = 1            
            if numParsIn > numPars:
                raise Exception(('this link function object models a linear '
                                 'link. Need to provide a list of arguments '
                                 'of length one: [A], with matrix A'))

        elif cls._functionType == 'linearTwoInputs':      # f(x) = A*x+B*u
            numPars = 2            
            if numParsIn > numPars:
                raise Exception(('this link function object models a linear '
                                 'link with two inputs. Need to provide a list'
                                 ' of arguments of length two: [A,B], '
                                 'with matrices A and B'))
            
        elif cls._functionType == 'linearThreeInputs':    # f(x) = A*x+B*u+F*z
            numPars = 3            
            if numParsIn > numPars:
                raise Exception(('this link function object models a linear '
                                 'link with three inputs. Need to provide a list'
                                 ' of arguments of length three: [A,B,F], '
                                 'with matrices A, B and F'))

        elif cls._functionType == 'linearFourInputs':    # f(x) = A*x+B*u+F*z+G*s
            numPars = 4            
            if numParsIn > numPars:
                raise Exception(('this link function object models a linear '
                                 'link with three inputs. Need to provide a list'
                                 ' of arguments of length four: [A,B,F,G], '
                                 'with matrices A, B, F and G'))
            
        elif cls._functionType == 'linearFiveInputs': # f(x) = A*x+B*u+F*z+G*s+H*r
            numPars = 5            
            if numParsIn > numPars:
                raise Exception(('this link function object models a linear '
                                 'link with three inputs. Need to provide a list'
                                 ' of arguments of length five: [A,B,F,G,H], '
                                 'with matrices A, B, F, G and H'))
            
        elif cls._functionType == 'affineLinear':          # f(x) = A*x+b
            numPars = 2            
            if numParsIn > numPars:
                raise Exception(('this link function object models an affine '
                                 'linear link. Need to provide a list of '
                                 'arguments of length two: [A,b], with matrix '
                                 'A and vector b'))
            
        elif cls._functionType == 'affineLinearTwoInputs': # f(x) = A*x+B*u+b
            numPars = 3            
            if numParsIn > numPars:
                raise Exception(('this link function object models an affine '
                                 'linear link with two inputs. Need to provide'
                                 ' a list of arguments of length three: '
                                 '[A,B,b], with matrices A, B and vector b'))
                
        if idxParsIn is None:
            idxParsIn = range(numParsIn)
        elif not np.all([i in range(numPars) for i in idxParsIn]):
            print('idxParsIn:')
            print(idxParsIn)
            print('range of IDs of stored parameters:')
            print(range(numPars))
            raise Exception(('when handing over argument idxParsIn, '
                             'make sure that the indexed parameters '
                             'you wish to update also exist.'))
        elif ((isinstance(idxParsIn,list) and len(idxParsIn)!=numParsIn) or 
             (isinstance(idxParsIn,np.ndarray) and idxParsIn.size!=numParsIn)): 
            print('idxParsIn:')
            print(idxParsIn)
            print('number of parameters handed over:')
            print(numParsIn)
            raise Exception(('when handing over argument idxParsIn, '
                             'make sure that a matching number of '
                             'parameters is handed over'))                
                
        i = 0
        while i < numParsIn:
            if pars[idxParsIn[i]] is None:  
                i += 1          # convention: do not update and continue
            elif (isinstance(pars[idxParsIn[i]],np.ndarray) and 
                  pars[idxParsIn[i]].shape==cls._pars[idxParsIn[i]].shape):
                cls._pars[idxParsIn[i]] = pars[idxParsIn[i]].copy()
                i += 1
            else:
                print('trying to update pars[i] for i=')
                print(idxParsIn[i])
                print('selected new pars[i] has shape')
                print(pars[idxParsIn[i]].shape)
                print('stored pars[i] has shape:')
                print(cls._pars[idxParsIn[i]].shape)
                raise Exception(('Shapes do not match. Parameters that shall '
                                 'not be updated can be handed over as None'))
            
#----this -------is ------the -------79 -----char ----compa rison---- ------bar

# to be imported from its own module: import ssmLinkFuns as ssm 
# (to give nice syntax -> ssm.identity(x) etc. ) 
class ssm_placeholder:                                 
    
    #def __init__(self):   
                  
    @staticmethod
    def identity(x):
        return x

    @staticmethod    
    def zeroFun(args=None):
        return 0
    
    @staticmethod
    def linear(args, pars):
        return np.dot(pars[0],args[0]) 
                                 
    @staticmethod
    def linearTwoInputs(args,pars):
        return np.dot(pars[0],args[0]) + np.dot(pars[1],args[1])

    @staticmethod
    def linearThreeInputs(args,pars):
        return (np.dot(pars[0],args[0]) + np.dot(pars[1],args[1])
                                        + np.dot(pars[2],args[2]))       

    @staticmethod
    def linearFourInputs(args,pars):
        return (np.dot(pars[0],args[0]) + np.dot(pars[1],args[1])
              + np.dot(pars[2],args[2]) + np.dot(pars[3],args[3]))   

    @staticmethod
    def linearFiveInputs(args,pars):
        return (np.dot(pars[0],args[0]) + np.dot(pars[1],args[1])
              + np.dot(pars[2],args[2]) + np.dot(pars[3],args[3])
              + np.dot(pars[4],args[4]))
                       
    @staticmethod
    def affineLinear(args, pars):
        return np.dot(pars[0],args[0]) + pars[1]
                                 
    @staticmethod
    def affineLinearWithInput(args,pars):
        return np.dot(pars[0],args[0]) + np.dot(pars[1],args[1]) + pars[2]


 

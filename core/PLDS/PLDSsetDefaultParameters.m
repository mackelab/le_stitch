function params = PLDSsetDefaultParameters(params,xDim,yDim)
%
% params = PLDSsetDefaultParameters(params,xDim,yDim)
%
%
% Lars Buesing, Jakob H Macke
%


%%%%%%%%%%% set standard parameters %%%%%%%%%%%%%%%%%%%%

params = touchField(params,'model');
params.model = touchField(params.model,'A',0.9*eye(xDim));                                  % these standard settings make sense for data binned at 10ms with average rates of roughly 10Hz
params.model = touchField(params.model,'Q',(1-0.9.^2)*eye(xDim));
params.model = touchField(params.model,'Q0',eye(xDim));
params.model = touchField(params.model,'x0',zeros(xDim,1));
params.model = touchField(params.model,'C',randn(yDim,xDim)./sqrt(xDim));
params.model = touchField(params.model,'d',zeros(yDim,1)-2.0);


%%%%%%%%%%% set standard observation model handles for variational inference %%%%%%%%%%%%%%%%%%%%

params.model = touchField(params.model,'likeHandle',       @ExpPoissonHandle);
params.model = touchField(params.model,'dualHandle',       @ExpPoissonDualHandle);
params.model = touchField(params.model,'domainHandle',     @ExpPoissonDomain);
params.model = touchField(params.model,'baseMeasureHandle',@PoissonBaseMeasure);
params.model = touchField(params.model,'inferenceHandle',  @PLDSVariationalInference);
params.model = touchField(params.model,'MStepHandle',      @PLDSMStep);


%%%%%%%%%%% set standard algorithmic parameters %%%%%%%%%%%%%%%%%%%%

params = touchField(params,'opts');
params.opts = touchField(params.opts,'algorithmic');


%%%% set parameters for Variational Inference %%%%

params.opts.algorithmic = touchField(params.opts.algorithmic,'VarInfX');
params.opts.algorithmic.VarInfX = touchField(params.opts.algorithmic.VarInfX,'minFuncOptions');

params.opts.algorithmic.VarInfX.minFuncOptions = touchField(params.opts.algorithmic.VarInfX.minFuncOptions,'display',	'none');
params.opts.algorithmic.VarInfX.minFuncOptions = touchField(params.opts.algorithmic.VarInfX.minFuncOptions,'maxFunEvals',	50000);
params.opts.algorithmic.VarInfX.minFuncOptions = touchField(params.opts.algorithmic.VarInfX.minFuncOptions,'MaxIter',	5000);
params.opts.algorithmic.VarInfX.minFuncOptions = touchField(params.opts.algorithmic.VarInfX.minFuncOptions,'progTol',	1e-9);
params.opts.algorithmic.VarInfX.minFuncOptions = touchField(params.opts.algorithmic.VarInfX.minFuncOptions,'optTol',	1e-5);
params.opts.algorithmic.VarInfX.minFuncOptions = touchField(params.opts.algorithmic.VarInfX.minFuncOptions,'Method',	'lbfgs');


%%%% set parameters for MStep of observation model %%%%%%%%

params.opts.algorithmic = touchField(params.opts.algorithmic,'MStepObservation');
params.opts.algorithmic.MStepObservation = touchField(params.opts.algorithmic.MStepObservation,'minFuncOptions');

params.opts.algorithmic.MStepObservation.minFuncOptions = touchField(params.opts.algorithmic.MStepObservation.minFuncOptions,'maxFunEvals', 5000);
params.opts.algorithmic.MStepObservation.minFuncOptions = touchField(params.opts.algorithmic.MStepObservation.minFuncOptions,'MaxIter',	  500);
params.opts.algorithmic.MStepObservation.minFuncOptions = touchField(params.opts.algorithmic.MStepObservation.minFuncOptions,'Method',	  'lbfgs');
params.opts.algorithmic.MStepObservation.minFuncOptions = touchField(params.opts.algorithmic.MStepObservation.minFuncOptions,'progTol',     1e-9);
params.opts.algorithmic.MStepObservation.minFuncOptions = touchField(params.opts.algorithmic.MStepObservation.minFuncOptions,'optTol',      1e-5);
params.opts.algorithmic.MStepObservation.minFuncOptions = touchField(params.opts.algorithmic.MStepObservation.minFuncOptions,'display',	  'none');


%%%% set parameters for EM iterations %%%%%%%%       

params.opts.algorithmic = touchField(params.opts.algorithmic,'TransformType','0');                                        % transform LDS parameters after each MStep to canonical form?
params.opts.algorithmic = touchField(params.opts.algorithmic,'EMIterations');
params.opts.algorithmic.EMIterations = touchField(params.opts.algorithmic.EMIterations,'maxIter',100);			% max no of EM iterations
params.opts.algorithmic.EMIterations = touchField(params.opts.algorithmic.EMIterations,'maxCPUTime',1200);		% max CPU time for EM
params.opts.algorithmic.EMIterations = touchField(params.opts.algorithmic.EMIterations,'progTolvarBound',1e-6);     	% progress tolerance on var bound per data time bin


%%%% set parameters for initialization methods %%%%

params.opts = touchField(params.opts,'initMethod','params');

switch params.opts.initMethod

   case 'params'
   	% do nothing
	
   case 'PLDSID'

   	params.opts.algorithmic = touchField(params.opts.algorithmic,'PLDSID');
	params.opts.algorithmic.PLDSID = touchField(params.opts.algorithmic.PLDSID,'algo','SVD');
	params.opts.algorithmic.PLDSID = touchField(params.opts.algorithmic.PLDSID,'hS',xDim);
	params.opts.algorithmic.PLDSID = touchField(params.opts.algorithmic.PLDSID,'minFanoFactor',1.01);
	params.opts.algorithmic.PLDSID = touchField(params.opts.algorithmic.PLDSID,'minEig',1e-4);
	params.opts.algorithmic.PLDSID = touchField(params.opts.algorithmic.PLDSID,'useB',0);
	params.opts.algorithmic.PLDSID = touchField(params.opts.algorithmic.PLDSID,'doNonlinTransform',1);

   case 'ExpFamPCA'

   	params.opts.algorithmic = touchField(params.opts.algorithmic,'ExpFamPCA');
	params.opts.algorithmic.ExpFamPCA = touchField(params.opts.algorithmic.ExpFamPCA,'dt',10);				% rebinning factor, choose such that roughly E[y_{kt}] = 1 forall k,t
	params.opts.algorithmic.ExpFamPCA = touchField(params.opts.algorithmic.ExpFamPCA,'lam',1);		  		% regularization coeff for ExpFamPCA
	params.opts.algorithmic.ExpFamPCA = touchField(params.opts.algorithmic.ExpFamPCA,'options');				% options for minFunc
	params.opts.algorithmic.ExpFamPCA.options = touchField(params.opts.algorithmic.ExpFamPCA.options,'display','none');
	params.opts.algorithmic.ExpFamPCA.options = touchField(params.opts.algorithmic.ExpFamPCA.options,'MaxIter',10000);
	params.opts.algorithmic.ExpFamPCA.options = touchField(params.opts.algorithmic.ExpFamPCA.options,'maxFunEvals',50000);
	params.opts.algorithmic.ExpFamPCA.options = touchField(params.opts.algorithmic.ExpFamPCA.options,'Method','lbfgs');
	params.opts.algorithmic.ExpFamPCA.options = touchField(params.opts.algorithmic.ExpFamPCA.options,'progTol',1e-9);
	params.opts.algorithmic.ExpFamPCA.options = touchField(params.opts.algorithmic.ExpFamPCA.options,'optTol',1e-5); 

   otherwise
	
	warning('Unknown PLDS initialization method, cannot set parameters')

end

function params = PLDSsetDefaultParameters(params,xDim,yDim)
%
% params = PLDSsetDefaultParameters(params,xDim,yDim)
%


%%%%%%%%%%% set standard parameters %%%%%%%%%%%%%%%%%%%%

params = touchField(params,'A',0.9*eye(xDim));
params = touchField(params,'Q',(1-0.9.^2)*eye(xDim));
params = touchField(params,'Q0',eye(xDim));
params = touchField(params,'x0',zeros(xDim,1));
params = touchField(params,'C',randn(yDim,xDim)./sqrt(xDim));
params = touchField(params,'d',zeros(yDim,1)-2.0);


%%%%%%%%%%% set standard observation model handles for variational inference %%%%%%%%%%%%%%%%%%%%

params = touchField(params,'likeHandle',@ExpPoissonHandle);
params = touchField(params,'dualHandle',@ExpPoissonDualHandle);



%%%%%%%%%%% set standard algorithmic parameters %%%%%%%%%%%%%%%%%%%%

params = touchField(params,'algorithmic');

%%%% set parameters for Variational Inference %%%%

params.algorithmic = touchField(params.algorithmic,'VarInfX');
params.algorithmic.VarInfX = touchField(params.algorithmic.VarInfX,'minFuncOptions');

params.algorithmic.VarInfX.minFuncOptions = touchField(params.algorithmic.VarInfX.minFuncOptions,'display',	'none');
params.algorithmic.VarInfX.minFuncOptions = touchField(params.algorithmic.VarInfX.minFuncOptions,'maxFunEvals',	50000);
params.algorithmic.VarInfX.minFuncOptions = touchField(params.algorithmic.VarInfX.minFuncOptions,'MaxIter',	5000);
params.algorithmic.VarInfX.minFuncOptions = touchField(params.algorithmic.VarInfX.minFuncOptions,'progTol',	1e-9);
params.algorithmic.VarInfX.minFuncOptions = touchField(params.algorithmic.VarInfX.minFuncOptions,'optTol',	1e-5);
params.algorithmic.VarInfX.minFuncOptions = touchField(params.algorithmic.VarInfX.minFuncOptions,'Method',	'lbfgs');


%%%% set parameters for MStep of observation model %%%%%%%%

params.algorithmic = touchField(params.algorithmic,'MStepObservation');
params.algorithmic.MStepObservation = touchField(params.algorithmic.MStepObservation,'minFuncOptions');

params.algorithmic.MStepObservation.minFuncOptions = touchField(params.algorithmic.MStepObservation.minFuncOptions,'maxFunEvals', 500);
params.algorithmic.MStepObservation.minFuncOptions = touchField(params.algorithmic.MStepObservation.minFuncOptions,'MaxIter',	  50);
params.algorithmic.MStepObservation.minFuncOptions = touchField(params.algorithmic.MStepObservation.minFuncOptions,'Method',	  'lbfgs');
params.algorithmic.MStepObservation.minFuncOptions = touchField(params.algorithmic.MStepObservation.minFuncOptions,'display',	  'none');


%%%% set parameters for EM iterations %%%%%%%%       

params.algorithmic = touchField(params.algorithmic,'TransformType','0');                                        % transform LDS parameters after each MStep to canonical form?
params.algorithmic = touchField(params.algorithmic,'EMIterations');
params.algorithmic.EMIterations = touchField(params.algorithmic.EMIterations,'maxIter',100);			% max no of EM iterations
params.algorithmic.EMIterations = touchField(params.algorithmic.EMIterations,'progTolvarBound',1e-6);     	% progress tolerance on var bound per data time bin


%%%% set parameters for initialization methods %%%%

params.algorithmic = touchField(params.algorithmic,'initMethod','params');

switch params.algorithmic.initMethod

   case 'params'
   	% do nothing
	
   case 'PLDSID'

   	params.algorithmic = touchField(params.algorithmic,'PLDSID');
	params.algorithmic.PLDSID = touchField(params.algorithmic.PLDSID,'algo','SVD');
	params.algorithmic.PLDSID = touchField(params.algorithmic.PLDSID,'hS',xDim);
	params.algorithmic.PLDSID = touchField(params.algorithmic.PLDSID,'minFanoFactor',1.01);
	params.algorithmic.PLDSID = touchField(params.algorithmic.PLDSID,'minEig',1e-4);
	params.algorithmic.PLDSID = touchField(params.algorithmic.PLDSID,'useB',0);
	params.algorithmic.PLDSID = touchField(params.algorithmic.PLDSID,'doNonlinTransform',1);

   case 'ExpFamPCA'

   	params.algorithmic = touchField(params.algorithmic,'ExpFamPCA');
	params.algorithmic.ExpFamPCA = touchField(params.algorithmic.ExpFamPCA,'dt',10);				% rebinning factor, choose such that roughly E[y_{kt}] = 1 forall k,t
	params.algorithmic.ExpFamPCA = touchField(params.algorithmic.ExpFamPCA,'lam',10);		  		% regularization coeff for ExpFamPCA
	params.algorithmic.ExpFamPCA = touchField(params.algorithmic.ExpFamPCA,'options');				% options for minFunc
	params.algorithmic.ExpFamPCA.options = touchField(params.algorithmic.ExpFamPCA.options,'display','none');
	params.algorithmic.ExpFamPCA.options = touchField(params.algorithmic.ExpFamPCA.options,'MaxIter',10000);
	params.algorithmic.ExpFamPCA.options = touchField(params.algorithmic.ExpFamPCA.options,'maxFunEvals',50000);
	params.algorithmic.ExpFamPCA.options = touchField(params.algorithmic.ExpFamPCA.options,'Method','lbfgs');
	params.algorithmic.ExpFamPCA.options = touchField(params.algorithmic.ExpFamPCA.options,'progTol',1e-9);
	params.algorithmic.ExpFamPCA.options = touchField(params.algorithmic.ExpFamPCA.options,'optTol',1e-5); 

   otherwise
	
	warning('Unknown PLDS initialization method, cannot set parameters')

end

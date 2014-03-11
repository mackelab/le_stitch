function params = PLDSInitialize(seq,xDim,initMethod,params)
%
% function params = PLDSInitialize(params,seq) 
%
% inititalize parameters of Population-LDS model with different methods. At
% the moment, focusses on exponential link function and Poisson
% observations.
%
%input: 
% seq:      standard data struct
% xdim:     desired latent dimensionality
% initMethods:
%
% - params											% just initialize minimal undefiened fields with standard values
% - PLDSID											% moment conversion + Ho-Kalman SSID
% - ExpFamPCA											% exponential family PCA
% - NucNormMin											% nuclear norm minimization, see [Robust learning of low-dimensional dynamics from large neural ensembles David Pfau, Eftychios A. Pnevmatikakis, Liam Paninski. NIPS2013] 
% params: if initialization method 'params' is chosen, the params-struct
% that one should use
%
% (c) L Buesing 2014


yDim       = size(seq(1).y,1);                                                                           
Trials     = numel(seq);
params.opts.initMethod = initMethod;
params     = PLDSsetDefaultParameters(params,xDim,yDim);					% set standard parameter values


switch initMethod

   case 'params'
   	% do nothing
	disp('Initializing PLDS parameters with given parameters')

       
   case 'PLDSID'
   	% !!! debugg SSID stuff separately & change to params.model convention
	disp('Initializing PLDS parameters using PLDSID')
	if params.model.useB
	  warning('SSID initialization with external input: not implemented yet!!!')
	end
	PLDSIDoptions = struct2arglist(params.opts.algorithmic.PLDSID);
	params = FitPLDSParamsSSID(seq,xDim,'params',params,PLDSIDoptions{:});


   case 'ExpFamPCA'     
   	% this replaces the FA initializer from the previous verions...
   	disp('Initializing PLDS parameters using exponential family PCA')
	
	dt = params.opts.algorithmic.ExpFamPCA.dt;
	Y  = [seq.y];
	[Cpca, Xpca, dpca] = ExpFamPCA(Y,xDim,'dt',dt,'lam',params.opts.algorithmic.ExpFamPCA.lam,'options',params.opts.algorithmic.ExpFamPCA.options);     
	params.model.C = Cpca;
	params.model.d = dpca-log(dt);								% compensate for rebinning

	if params.model.useB; u = [seq.u];else;u = [];end
	params.model = LDSObservedEstimation(Xpca,params.model,dt,u);
	

   case 'NucNormMin'
	disp('Initializing PLDS parameters using Nuclear Norm Minimization')
	
        dt = params.opts.algorithmic.NucNormMin.dt;
	seqRebin = rebinRaster(seq,dt);
        Y  = [seqRebin.y];
	options = params.opts.algorithmic.NucNormMin.options;
	options.lambda = options.lambda*sqrt(size(Y,1)*size(Y,2));
	[Y,Xu,Xs,Xv,d] = MODnucnrmminWithd( Y, options );
	params.model.d = d-log(dt);

	if ~params.opts.algorithmic.NucNormMin.fixedxDim
	   disp('Variable dimension; still to implement!')
	else
	   params.model.C = Xu(:,1:xDim)*Xs(1:xDim,1:xDim);
	   if params.model.useB; u = [seq.u];else;u = [];end
	   params.model = LDSObservedEstimation(Xv(:,1:xDim)',params.model,dt,u);
	end
	

   otherwise
	warning('Unknown PLDS initialization method')

end

if params.model.useB && (numel(params.model.B)<1)
  params.model.B = zeros(xDim,size(seq(1).u,1));
end

params = LDSTransformParams(params,'TransformType',params.opts.algorithmic.TransformType);	% clean up parameters
params.modelInit = params.model;

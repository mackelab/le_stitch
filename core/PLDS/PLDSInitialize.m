function params = PLDSInitialize(seq,xDim,initMethod,params)
%
% function params = PLDSInitialize(params,seq) 
%
% initMethods:
%
% - params						% just initialize minimal undefiened fields with standard values
% - PLDSID						% moment conversion + Ho-Kalman SSID, default
% - ExpFamPCA						% exponential family PCA, if trials are short use this
% - others... to implement: NucNormMin + SSID 
%


yDim       = size(seq(1).y,1);                                                                           
Trials     = numel(seq);
params.opts.initMethod = initMethod;
params     = PLDSsetDefaultParameters(params,xDim,yDim);  % set standard parameter values


switch initMethod

   case 'params'
   	% do nothing
	disp('Initializing PLDS parameters with given parameters')

       
   case 'PLDSID'
   	% !!! debugg SSID stuff separately & change to params.model convention
	disp('Initializing PLDS parameters using PLDSID')
	PLDSIDoptions = struct2arglist(params.opts.algorithmic.PLDSID);
	params = FitPLDSParamsSSID(seq,xDim,'params',params,PLDSIDoptions{:});


   case 'ExpFamPCA'     
   	% this replaces the FA initializer from the previous verions...
   	disp('Initializing PLDS parameters using exponential family PCA')
	
	dt = params.opts.algorithmic.ExpFamPCA.dt;
	Y  = [seq.y];
	[Cpca, Xpca, dpca] = ExpFamPCA(Y,xDim,'dt',dt,'lam',params.opts.algorithmic.ExpFamPCA.lam,'options',params.opts.algorithmic.ExpFamPCA.options);     
	params.model.C = Cpca;
	params.model.d = dpca-log(dt);				% compensate for rebinning

	Tpca = size(Xpca,2);
	params.model.Pi = Xpca*Xpca'./Tpca;

	params.model.A  = Xpca(:,2:end)/Xpca(:,1:end-1);
	params.model.A  = diag(min(max((diag(abs(params.model.A))).^(1/dt),0.5),0.98));
	params.model.Q  = params.model.Pi - params.model.A*params.model.Pi*params.model.A';	% set innovation covariance Q such that stationary dist matches the ExpFamPCA solution params.model.Pi
	[Uq Sq Vq] = svd(params.model.Q);				% ensure that Q is pos def
	params.model.Q  = Uq*diag(max(diag(Sq),1e-3))*Uq';
	params.model.x0 = zeros(xDim,1);
	params.model.Q0 = dlyap(params.model.A,params.model.Q);			% set initial distribution to stationary distribution, this could prob be refined
	

   otherwise
	warning('Unknown PLDS initialization method')

end

%%% !!! maybe parameter cleanup steps here
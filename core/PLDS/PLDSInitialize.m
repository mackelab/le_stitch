function params = PLDSInitialize(seq,xDim,varargin)
%
% function params = PLDSInitialize(params,seq) 
%
% Methods:
%
% - params						% just initialize minimal undefiened fields with standard values
% - PLDSID						% moment conversion + Ho-Kalman SSID, default
% - ExpFamPCA						% exponential family PCA, if trials are short use this
% - others... to implement: NucNormMin + SSID 
%


initMethod = 'PLDSID';
params     = [];

assignopts(who,varargin);

yDim       = size(seq(1).y,1);                                                                           
Trials     = numel(seq);
params.algorithmic.initMethod = initMethod;
params     = PLDSsetDefaultParameters(params,xDim,yDim);  % set standard parameter values


switch initMethod

   case 'params'
   	% do nothing
	disp('Initializing PLDS parameters with given parameters')

       
   case 'PLDSID'
   	% !!! rewrite to take out all parameters and put them into PLDSsetDefaultParameters
	disp('Initializing PLDS parameters using PLDSID')
	PLDSIDoptions = struct2arglist(params.algorithmic.PLDSID);
	params = FitPLDSParamsSSID(seq,xDim,'params',params,PLDSIDoptions{:});


   case 'ExpFamPCA'     % this should replace the FA initializer from the previous verions...
   	% !!! also package all parameters into PLDSsetDefaultParameters
   	disp('Initializing PLDS parameters using exponential family PCA')
	
	dt = params.algorithmic.ExpFamPCA.dt;
	Y  = [seq.y];
	[Cpca, Xpca, dpca] = ExpFamPCA(Y,xDim,'dt',dt,'lam',params.algorithmic.ExpFamPCA.lam,'options',params.algorithmic.ExpFamPCA.options);     
	params.C = Cpca;
	params.d = dpca-log(dt);				% compensate for rebinning
	Tpca = size(Xpca,2);
	params.Pi = (Xpca*Xpca')/Tpca;				% set stationary covariance to the empirical one

	[Upi Spi Vpi] = svd(params.Pi);                         % whiten latent space
        Tpi = diag(1./sqrt(diag(Spi)))*Upi';
	
	if cond(Tpi)>1e2
	   warning('Trying to rescale latent dimensions with a ill-conditioned transformation')
	end

        params.Pi = Tpi*params.Pi*Tpi';
        params.C  = params.C/Tpi;
	Xpca      = Tpi*Xpca;
	
	params.A  = Xpca(:,2:end)/Xpca(:,1:end-1);
	params.A  = diag(min(max((diag(abs(params.A))).^(1/dt),0.5),0.98));
	params.Q  = eye(xDim)-params.A*params.A';		% set innovation covariance Q such that stationary dist is white
	[Uq Sq Vq] = svd(params.Q);				% ensure that Q is pos def
	params.Q  = Uq*diag(max(diag(Sq),1e-3))*Uq';
	params.x0 = zeros(xDim,1);
	params.Q0 = eye(xDim);			% set initial distribution to stationary distribution, this could prob be refined
	

   otherwise
	warning('Unknown PLDS initialization method')

end

%%% !!! maybe parameter cleanup steps here
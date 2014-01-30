function [params] = LDSTransformParams(params,varargin)
%
%
% transform parameters of LDS by imposing constraints on C and the
% stationary distribution Pi.
%
%  Pi := dlyap(params.model.A,params.model.Q)   stationary distribution
%
% 0: do nothing
% 1: C'*C = eye		&&		Pi = diag    [default]
% 2: C'*C = diag 	&& 		Pi = eye
% 3: not implemented / tested yet
% 4: not implemented / tested yet
%
% (c) 2014 Lars Busing   lars@stat.columbia.edu
%
%
% also see: LDSApplyParamsTransformation(M,params,varargin)
%


TransformType   = '1';

assignopts(who,varargin);

xDim = size(params.model.A,1);

switch TransformType

  case '0'
 
	% do nothing

  case '1'

       [UC,SC,VC]    = svd(params.model.C,0);
       params        = LDSApplyParamsTransformation(SC*VC',params);

       params.model.Pi     = dlyap(params.model.A,params.model.Q);
       [UPi SPi VPi] = svd(params.model.Pi);
       params        = LDSApplyParamsTransformation(UPi',params);

  case '2'  
  
	params.model.Pi     = dlyap(params.model.A,params.model.Q);
	[UPi SPi VPi] = svd(params.model.Pi);
	M    	      = diag(1./sqrt(diag(SPi)))*UPi';
       	params        = LDSApplyParamsTransformation(M,params);    	

        [UC,SC,VC]    = svd(params.model.C,0);
        params        = LDSApplyParamsTransformation(VC',params);  	

  otherwise
	
	warning('Unknow paramter transformation type')

end

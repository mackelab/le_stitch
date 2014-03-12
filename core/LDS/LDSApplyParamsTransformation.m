function [params] = LDSApplyParamsTransformation(M,params,varargin)
%
% [params] = LDSTransformParams(M,params,varargin)
%
% Applies M from left and inv(M)/M' from the right
%
% to A,Q,Q0,x0,B,C
%
%
% L Buesing 2014


assignopts(who,varargin); 

if cond(M)>1e3
   warning('Attempting LDSApplyParamsTransformation with ill-conditioned transformation')
end

params.model.C  =     params.model.C  / M;
params.model.A  = M * params.model.A  / M;
params.model.Q  = M * params.model.Q  * M';
params.model.Q0 = M * params.model.Q0 * M';
params.model.x0 = M * params.model.x0;

if isfield(params.model,'B')
  params.model.B = M*params.model.B;
end

if isfield(params.model,'Pi')
   params.model.Pi = dlyap(params.model.A,params.model.Q);
end

function [params] = LDSApplyParamsTransformation(M,params,varargin)
%
% [params] = LDSTransformParams(M,params,varargin)
%
% Applies M from left and inv(M)/M' from the right
%
% to A,Q,Q0,x0,B,C

assignopts(who,varargin); 

if cond(M)>1e3
   warning('Attempting LDSApplyParamsTransformation with ill-conditioned transformation')
end

params.C  =     params.C  / M;
params.A  = M * params.A  / M;
params.Q  = M * params.Q  * M';
params.Q0 = M * params.Q0 * M';
params.x0 = M * params.x0;

if isfield(params,'B')
  params.B = M*params.B;
end

if isfield(params,'Pi')
   params.Pi = M * params.Pi * M';
end

function seq = PLDSsample(params,T,Trials,varargin)
%
% seq = PLDSsample(params,T,Trials)
%
% sample from a poisson model with exponential nonlinearity
% uses LDSsample
%
% (c) L Buesing 2014
%

yMax = inf;
assignopts(who,varargin);

seq = LDSsample(params,T,Trials,varargin{:});

for tr=1:Trials
  seq(tr).yr   = seq(tr).y;
  seq(tr).y    = poissrnd(exp(seq(tr).yr));
  seq(tr).y(:) = min(yMax,seq(tr).y(:));
end


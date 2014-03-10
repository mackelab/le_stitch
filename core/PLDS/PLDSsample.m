function seq = PLDSsample(params,T,Trials)
%
% seq = PLDSsample(params,T,Trials)
%
% sample from a poisson model with exponential nonlinearity
%
% (c) L Buesing 2014

if numel(T)==1
   T = ones(Trials,1)*T;
end

Trials = numel(T);

[yDim xDim] = size(params.model.C);
CQ          = chol(params.model.Q);
CQ0         = chol(params.model.Q0);


for tr=1:Trials

  if params.model.useB
    uDim = size(params.model.B,2);
    gpsamp = sampleGPPrior(1,T(tr),uDim-1);
    tpsamp = (vec(repmat(rand(1,floor(T(tr)/10))>0.5,10,1))-0.5); tpsamp = [tpsamp' zeros(1,T(tr)-floor(T(tr)/10)*10)];
    seq(tr).u = [gpsamp{1}/3;tpsamp];
  end

  seq(tr).x = zeros(xDim,T(tr));
  seq(tr).x(:,1) = params.model.x0+CQ0'*randn(xDim,1);
  if params.model.useB; seq(tr).x(:,1) = seq(tr).x(:,1)+params.model.B*seq(tr).u(:,1);end;
  for t=2:T(tr)
      seq(tr).x(:,t) = params.model.A*seq(tr).x(:,t-1)+CQ'*randn(xDim,1);
      if params.model.useB; seq(tr).x(:,t) = seq(tr).x(:,t)+params.model.B*seq(tr).u(:,t);end;
  end
  seq(tr).yr = bsxfun(@plus,params.model.C*seq(tr).x,params.model.d);
  seq(tr).y  = poissrnd(exp(seq(tr).yr));
  seq(tr).T  = T(tr);
end

function seq = sampleLDS(params,T,Trials)
%
% sampleLDS(params,T,Trials)
%
% sample from linear dynamical system model

if numel(T)==1
   T = ones(Trials,1)*T;
end

Trials = numel(T);


[yDim xDim] = size(params.model.C);
CQ          = chol(params.model.Q);
CQ0         = chol(params.model.Q0);
try
       CR = chol(params.model.R);
catch
       disp('No pos def R, adding zero Gaussian observation noise');
       CR = zeros(yDim);
end

for tr=1:Trials
  seq(tr).x = zeros(xDim,T(tr));
  seq(tr).x(:,1) = params.model.x0+CQ0'*randn(xDim,1);
  for t=2:T(tr)
      seq(tr).x(:,t) = params.model.A*seq(tr).x(:,t-1)+CQ'*randn(xDim,1);
  end
  seq(tr).y = bsxfun(@plus,params.model.C*seq(tr).x,params.model.d)+CR'*randn(yDim,T(tr));
  seq(tr).T = T(tr);
end

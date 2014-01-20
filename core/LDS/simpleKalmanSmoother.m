function [seq Lambda LambdaPost] = simpleKalmanSmoother(params,seq)
%
% simplest Kalman smoother in O(T)
%
% assume all trials are of the same length T
%


[yDim xDim] = size(params.C);  

Trials      = numel(seq);
T           = size(seq(1).y,2);


%%%%%%%%%%%%% covariances %%%%%%%%%%%%

Lambda = buildPriorPrecisionMatrixFromLDS(params,T); % prior precision
LambdaPost = Lambda;                                 % posterior precision

CRC    = zeros(xDim*T,xDim);
CRinvC = params.C'*pinv(params.R)*params.C;
for t=1:T
  xidx = ((t-1)*xDim+1):(t*xDim);
  CRC(xidx,:) = CRinvC;
  LambdaPost(xidx,xidx)  = LambdaPost(xidx,xidx) + CRinvC;
end

[Vsm, VVsm] = smoothedKalmanMatrices(params,CRC);


%%%%%%%%%%%%%%%%% means %%%%%%%%%%%%%%%%

Mu = zeros(xDim,T);
Mu(:,1) = params.x0;

for t=2:T
    Mu(:,t) = params.A*Mu(:,t-1);
end

Mu = Lambda*vec(Mu);


for tr=1:Trials
    Y = bsxfun(@minus,seq(tr).y,params.d);
    Y = params.R\Y;
    Y = params.C'*Y;
    seq(tr).posterior.xsm = reshape(LambdaPost\(Mu+vec(Y)),xDim,T);
    seq(tr).posterior.Vsm = Vsm;
    seq(tr).posterior.VVsm = VVsm;
end

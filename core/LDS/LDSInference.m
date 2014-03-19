function [seq Lambda LambdaPost] = LDSInference(params,seq)
%
% simplest Kalman smoother in O(T)
%
% assume all trials are of the same length T
%
% !!! align this with convention for returning cost such that it works with PopSpikeEM.m


[yDim xDim] = size(params.model.C);  

Trials      = numel(seq);
T           = size(seq(1).y,2);


%%%%%%%%%%%%% covariances %%%%%%%%%%%%

Lambda     = buildPriorPrecisionMatrixFromLDS(params,T); % prior precision
LambdaPost = Lambda;                                 % posterior precision

CRC    = zeros(xDim*T,xDim);
CRinvC = params.model.C'*pinv(params.model.R)*params.model.C;
for t=1:T
  xidx = ((t-1)*xDim+1):(t*xDim);
  CRC(xidx,:) = CRinvC;
  LambdaPost(xidx,xidx)  = LambdaPost(xidx,xidx) + CRinvC;
end

[Vsm, VVsm] = smoothedKalmanMatrices(params.model,CRC);


%%%%%%%%%%%%%%%%% means %%%%%%%%%%%%%%%%

for tr=1:Trials
  Mu = zeros(xDim,T);
  Mu(:,1) = params.model.x0;
  if params.model.notes.useB; Mu(:,1) = Mu(:,1)+params.model.B*seq(tr).u(:,1);end;

  for t=2:T
    Mu(:,t) = params.model.A*Mu(:,t-1);
     if params.model.notes.useB; Mu(:,t) = Mu(:,t)+params.model.B*seq(tr).u(:,t);end;
  end

  Mu = Lambda*vec(Mu);
  Y  = bsxfun(@minus,seq(tr).y,params.model.d);
  if params.model.notes.useS
    Y = Y-seq.s;
  end
  Y  = params.model.R\Y;
  Y  = params.model.C'*Y;
  seq(tr).posterior.xsm  = reshape(LambdaPost\(Mu+vec(Y)),xDim,T);
  seq(tr).posterior.Vsm  = Vsm;
  seq(tr).posterior.VVsm = VVsm;
end

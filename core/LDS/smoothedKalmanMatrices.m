function [Vsm VVsm F0 F1] = smoothedKalmanMatrices(params,CRinvC)
%
% [Vsm VVsm F0 F1] = smoothedKalmanMatrices(params,CRinvC)
%
% 
% computes posterior covariances by Kalman smoothing
%
% INPUT:
%
%  - CRinvC is a array of dimension (xDim*T) x (xDim) where
%    CRinvC((t-1)*xDim+1:t*xDim) = C'/Rt*C where Rt is observation noise
%    covariance matrix at time t and C is loading matrix
%
%  - params with fields
%    - A
%    - Q
%
%
% OUTPUT: all matrices but VVsm are of dimension (xDim*T) x (xDim)
%
% the t-th block of dimension (xDim) x (xDim) is defined as:
% F0   = predictive cov Cov(x_t|y_{1:t-1})
% F1   = filtering cov  Cov(x_t|y_{1:t})
% Vsm  = smoothed cov   Cov(x_t|y_{1:T})
%
% VVsm is of dimension (xDim*(T-1)) x (xDim)
%
% VVsm = smoothed cov   Cov(x_{t+1},x_t}|y_{1:T})
% 
%
%
% (c) Lars Buesing 2013,2014
%


xDim = size(params.A,1);
T    = round(size(CRinvC,1)/xDim);
Vsm  = zeros(xDim*T,xDim);
VVsm = zeros(xDim*(T-1),xDim);
F0   = zeros(xDim*T,xDim);
F1   = zeros(xDim*T,xDim);


% forward pass

F0(1:xDim,1:xDim) = params.Q0;
for t=1:(T-1)
  xidx = ((t-1)*xDim+1):(t*xDim);
  %F1(xidx,:) = pinv(eye(xDim)+F0(xidx,:)*CRinvC(xidx,:))*F0(xidx,:);  % debug line, do not use
  F1(xidx,:) = (eye(xDim)+F0(xidx,:)*CRinvC(xidx,:))\F0(xidx,:);
  F0(xidx+xDim,:) = params.A*F1(xidx,:)*params.A'+params.Q;
end
t=T;xidx = ((t-1)*xDim+1):(t*xDim);
F1(xidx,:) = eye(xDim)/(eye(xDim)/(F0(xidx,:))+CRinvC(xidx,:));


% backward pass using Rauch–Tung–Striebel smoother

t = T; xidx = ((t-1)*xDim+1):(t*xDim);
Vsm(xidx,:) = F1(xidx,:);
for t=(T-1):(-1):1
  xidx = ((t-1)*xDim+1):(t*xDim);
  Ck = F1(xidx,:)*params.A'/F0(xidx+xDim,:);
  Vsm(xidx,:)  = F1(xidx,:)+Ck*(Vsm(xidx+xDim,:)-F0(xidx+xDim,:))*Ck';
  VVsm(xidx,:) = (Ck*Vsm(xidx+xDim,:))';
end

function [ypred xpred xpredCov seqInf] = PLDSPredictRange(params,y,condRange,predRange,varargin);
%
% function [ypred xpred xpredCov] = PLDSPredictRange(params,y);
%
%
% Lars Buesing, 2014
%

lamInit = [];

assignopts(who,varargin);
 
[yDim xDim] = size(params.C);

Tcond = numel(condRange);
Tpred = numel(predRange);
tcRlo = min(condRange);
tcRhi = max(condRange);
tpRlo = min(predRange);
tpRhi = max(predRange);

if size(y,2)<Tcond
   error('Conditioning range larger than data, aborting')
elseif size(y,2)>Tcond
   y = y(:,condRange);
end


paramsInf = params;
for t=2:tcRlo                                                   % get the starting distribution right
    paramsInf.x0 = paramsInf.A*paramsInf.x0;
    paramsInf.Q0 = paramsInf.A*paramsInf.Q0*paramsInf.A'+paramsInf.Q;
end

seqInf.y  = y;
seqInf.T  = size(seqInf.y,2);
if numel(lamInit)==(yDim*Tcond)
   disp('Warm-starting predction inference')
   seqInf.posterior.lamOpt = lamInit;
end

seqInf   = PLDSVariationalInference(paramsInf,seqInf);
xpred    = zeros(xDim,Tpred);
xpredCov = zeros(xDim,xDim,Tpred);
ypred    = zeros(yDim,Tpred);

xNow     = seqInf(1).posterior.xsm(:,end);
xCovNow  = seqInf(1).posterior.Vsm(end+1-xDim:end,:); 

for t = (tcRhi+1):tpRhi    % progagate prediction
    xNow    = paramsInf.A*xNow;
    xCovNow = paramsInf.A*xCovNow*paramsInf.A'+paramsInf.Q;
    if t>=tpRlo
       xpred(:,t-tpRlo+1) = xNow;
       xpredCov(:,:,t-tpRlo+1) = xCovNow;
       ypred(:,t-tpRlo+1) = exp(params.C*xNow+params.d+0.5*diag(params.C*xCovNow*params.C'));
    end
end

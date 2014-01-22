function [f, df] = PLDSMStepObservationCost(vecCd,seq,params)
%
% function [f, df] = PLDSMStepObservationCost(vecCd,seq,params)
%
% Mstep for observation parameters C,d for standard PLDS with exp-Poisson observations
%
% Input:
%	- convention Cd = [C d] = [C^e C^i d] and vecCd = vec(Cd)
%
% to do: 
%
%       0) analyze run time
%



Trials  = numel(seq);
yDim    = size(seq(1).y,1);
xDim    = size(params.A,1);


CdMat   = reshape(vecCd,yDim,xDim+1);
C       = CdMat(:,1:xDim);
d       = CdMat(:,end);

CC      = zeros(yDim,xDim^2);
for yd=1:yDim
  CC(yd,:) = vec(C(yd,:)'*C(yd,:));
end


f   = 0;				% current value of the cost function = marginal likelihood
df  = zeros(size(C));			% derviative wrt C
dfd = zeros(yDim,1);			% derivative wrt d, concatenate later with df

for tr=1:Trials
 
  T    = size(seq(tr).y,2);
  y    = seq(tr).y;
  m    = seq(tr).posterior.xsm;
  Vsm  = reshape(seq(tr).posterior.Vsm',xDim,xDim,T);
  
  VsmPop = reshape(Vsm,xDim.^2,T);
    
  h    = bsxfun(@plus,C*m,d);
  rho  = CC*VsmPop;
  yhat = exp(h+rho/2);

  f    = f+sum(vec(y.*h-yhat));
  
  TT = yhat*VsmPop';
  TT = reshape(TT,yDim*xDim,xDim);
  TT = squeeze(sum(reshape(bsxfun(@times,TT,vec(C)),yDim,xDim,xDim),2));
     
  df = df  + (y-yhat)*m'-TT;
  dfd= dfd + sum((y-yhat),2);

  
end

f  = -f;
df = -vec([df dfd]);

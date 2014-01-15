function [seq] = VariationalInferenceDualLDS(params,seq,optparams)
%
% [seq] = VariationalInferenceDualLDS(params,seq);
%
% Assume here that all trials are of same length
%
% do this for different normalizer than that of Poisson --> introduce base measure handle
%


Trials      = numel(seq);
[yDim xDim] = size(params.C);


% set up parameters for variational inference

VarInfparams    = params;
VarInfparams.CC = zeros(xDim,xDim,yDim);
for yy=1:yDim
  VarInfparams.CC(:,:,yy) = params.C(yy,:)'*params.C(yy,:);
end
VarInfparams.CC = reshape(VarInfparams.CC,xDim^2,yDim);


% iterate over trials

for tr = 1:Trials
  
  T = size(seq(tr).y,2);

  VarInfparams.mu = zeros(xDim,T); %prior mean
  VarInfparams.mu(:,1) = params.x0;
  for t=2:T; VarInfparams.mu(:,t) = params.A*VarInfparams.mu(:,t-1); end
  VarInfparams.mu = vec(VarInfparams.mu);
  
  Cl = {}; for t=1:T; Cl = {Cl{:} params.C}; end
  VarInfparams.W      = sparse(blkdiag(Cl{:})); %stacked loading matrix
  
  VarInfparams.y      = seq(tr).y;
  VarInfparams.Lambda = buildPriorPrecisionMatrixFromLDS(params,T);  % generate prior precision matrix
  VarInfparams.WlamW  = sparse(zeros(xDim*T)); %allocate sparse observation matrix
  % fix this: optparams.dualParams{tr} should default to 0
  VarInfparams.dualParams      = optparams.dualParams{tr};   
  VarInfparams.DataBaseMeasure = -sum(log(gamma(vec(seq(tr).y)+1))); %!!! make this more general
  
  % init value
  if isfield(seq(tr),'posterior')&&isfield(seq(tr).posterior,'lamInit')
    lamInit = seq(tr).posterior.lamInit;
  else
    lamInit = zeros(yDim*T,1)+mean(vec(seq(tr).y));
    %lamInit = 0.01+repmat(mean(seq(tr).y,2),T,1); % doesn't work that well
  end
  % warm start inference if possible
  if isfield(seq(tr),'posterior')&&isfield(seq(tr).posterior,'lamOpt')
    [~, ~, LlamOpt]  = VariationalInferenceDualCost(seq(tr).posterior.lamOpt,VarInfparams);
    [~, ~, LlamInit] = VariationalInferenceDualCost(lamInit,VarInfparams);
    %if LlamOpt>LlamInit;                      % this criterion prob doesn't make sense, as we're comparing upper bounds 
      lamInit = seq(tr).posterior.lamOpt; 
    %end
  end
  
  lamOpt = minFunc(@VariationalInferenceDualCost,lamInit,optparams.minFuncOptions,VarInfparams);
    
  [DualCost, ~, varBound, m_ast, invV_ast, Vsm, VVsm, over_m, over_v] = VariationalInferenceDualCost(lamOpt,VarInfparams);


  seq(tr).posterior.xsm        = reshape(m_ast,xDim,T);	      % posterior mean   E[x(t)|y(1:T)]
  seq(tr).posterior.Vsm        = Vsm;			      % posterior covariances Cov[x(t),x(t)|y(1:T)]
  seq(tr).posterior.VVsm       = VVsm;			      % posterior covariances ...
  seq(tr).posterior.lamOpt     = lamOpt;		      % optimal value of dual variable
  seq(tr).posterior.lamInit    = lamInit;
  seq(tr).posterior.varBound   = varBound;		      % variational lower bound for trial
  seq(tr).posterior.DualCost   = DualCost;
  seq(tr).posterior.over_m     = over_m;		      % C*xsm+d
  seq(tr).posterior.over_v     = over_v;		      % C*Vsm*C'
  
  
end

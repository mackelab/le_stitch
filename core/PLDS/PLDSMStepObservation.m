function params = PLDSMStepObservation(params,seq)
%
% function params = PLDSMStepObservation(params,seq)
%


minFuncOptions = params.algorithmic.MStepObservation.minFuncOptions;

[yDim xDim] = size(params.C);


CdInit = vec([params.C params.d]); % warm start at current parameter values
MStepCostHandle = @PLDSMStepObservationCost;

%%% optimization %%%

CdOpt    = minFunc(MStepCostHandle,CdInit,minFuncOptions,seq,params);
CdOpt    = reshape(CdOpt,yDim,xDim+1);

params.C = CdOpt(:,1:xDim);
params.d = CdOpt(:,end);

function [seq] = PLDSVariationalInference(params,seq)
%
% [seq] = PLDSVariationalInferenceDualLDS(params,seq)
%

Trials = numel(seq);

for tr=1:Trials
  optparams.dualParams{tr} = [];
end
optparams.minFuncOptions = params.algorithmic.VarInfX.minFuncOptions;

[seq] = VariationalInferenceDualLDS(params,seq,optparams);
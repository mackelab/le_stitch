function params = PLDSMStep(params,seq)
%
% params = PLDSMStep(params,seq) 
%


params = LDSMStep(params,seq);
params = PLDSMStepObservation(params,seq);

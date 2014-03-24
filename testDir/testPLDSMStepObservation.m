clear all
close all


xDim   = 3;
yDim   = 50;
T      = 100;
Trials = 5;


tp  = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',-1.5);
tp  = LDSApplyParamsTransformation(randn(xDim)+eye(xDim)*0.3,tp);
seq = PLDSsample(tp,T,Trials);


params = tp;
%params.model.R = params.model.R*0;
tic
seq = PLDSVariationalInference(params,seq);
toc

% checking posterior
plotPosterior(seq,1,tp);


% do MStep
params.opts.algorithmic.MStepObservation.minFuncOptions.display = 'iter';
params = PLDSMStepObservation(params,seq);


% look at some invariant comparison statistics

subspace(tp.model.C,params.model.C)
figure
plot(vec(tp.model.C),vec(params.model.C),'xr')

figure
plot(tp.model.d,params.model.d,'xr')

figure; hold on
plot(tp.model.R,params.model.R,'xr')
plot([0 1],[0 1]) 



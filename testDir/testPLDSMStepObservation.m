clear all
close all


xDim   = 10;
yDim   = 100;
T      = 100;
Trials = 20;


trueparams = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',0.0);
trueparams = LDSApplyParamsTransformation(randn(xDim)+eye(xDim)*0.3,trueparams);
seq = PLDSsample(trueparams,T,Trials);
tp = trueparams;

tic
seq = PLDSVariationalInference(tp,seq);
toc

% checking posterior
plotPosterior(seq,1,tp);


% do MStep
params = PLDSMStepObservation(tp,seq);


% look at some invariant comparison statistics

subspace(tp.C,params.C)
figure
plot(vec(tp.C),vec(params.C),'xr')

figure
plot(tp.d,params.d,'xr')

clear all
close all


xDim   = 10;
yDim   = 100;
T      = 100;
Trials = 10;


trueparams = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',-1.5);
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

subspace(tp.model.C,params.model.C)
figure
plot(vec(tp.model.C),vec(params.model.C),'xr')

figure
plot(tp.model.d,params.model.d,'xr')

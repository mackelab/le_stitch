clear all
close all


xDim   = 3;
yDim   = 30;
T      = 100;
Trials = 100;


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
params = LDSMStep(tp,seq);


% look at some invariant comparison statistics

sort(abs(eig(tp.A)))
sort(abs(eig(params.A)))
figure
plot(vec(tp.A),vec(params.A),'xr')

tp.Pi     = dlyap(tp.A,tp.Q);
params.Pi = dlyap(params.A,params.Q);
figure
plot(vec(tp.Pi),vec(params.Pi),'xr')

figure
plot(vec(tp.Q0),vec(params.Q),'xr')

figure
plot(tp.x0,params.x0,'xr')

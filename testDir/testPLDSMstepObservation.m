clear all
close all


xDim   = 5;
yDim   = 50;
T      = 100;
Trials = 50;


params = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim);
seq = PLDSsample(params,T,Trials);

tic;seq = PLDSVariationalInference(seq,params);toc;

plotPosterior(seq,1,params);


% 

ESTparams = PLDSMStepObservation(params,seq)

figure
plot(params.C,ESTparams.C,'xr')

figure
plot(params.d,ESTparams.d,'xr')                                           


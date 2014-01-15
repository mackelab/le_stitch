clear all
close all


xDim   = 30;
yDim   = 100;
T      = 100;
Trials = 50;


params = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim);
seq = PLDSsample(params,T,Trials);

tic
seq = PLDSVariationalInference(seq,params);
toc

plotPosterior(seq,1,params);


% works!

ESTparams = MStepLDS(params,seq);

figure
imagesc([params.A ESTparams.A])
figure
imagesc([params.Q ESTparams.Q])
figure
imagesc([params.Q0 ESTparams.Q0])
figure
plot(params.x0,ESTparams.x0,'xr')


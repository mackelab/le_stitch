clear all
close all

xDim   = 10;
yDim   = 50;
T      = 50;
Trials = 5000;

params = generateLDS('xDim',xDim,'yDim',yDim);
params = LDSApplyParamsTransformation(randn(xDim)+0.1*eye(xDim),params);
seq    = sampleLDS(params,T,Trials);
seq    = simpleKalmanSmoother(params,seq);


%%%%%%%%%%%%%%%%%%% test LDS Mstep %%%%%%%%%%%%%%%%%%


ESTparams = LDSMStep(params,seq);

figure
plot(params.A,ESTparams.A,'xr')
figure
imagesc([params.A ESTparams.A])
figure
plot(params.Q,ESTparams.Q,'xr')
figure
imagesc([params.Q ESTparams.Q])
figure
plot(params.Q0,ESTparams.Q0,'xr')
figure
imagesc([params.Q0 ESTparams.Q0])
figure
plot(params.x0,ESTparams.x0,'xr')
 



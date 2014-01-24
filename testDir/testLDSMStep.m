clear all
close all

xDim   = 10;
yDim   = 50;
T      = 50;
Trials = 50;

trueparams = LDSgenerateExample('xDim',xDim,'yDim',yDim);
trueparams = LDSApplyParamsTransformation(randn(xDim)+0.1*eye(xDim),trueparams);
seq = LDSsample(trueparams,T,Trials);
seq = LDSInference(trueparams,seq);
tp  = trueparams;

%%%%%%%%%%%%%%%%%%% test LDS Mstep %%%%%%%%%%%%%%%%%%


params = LDSMStepLDS(tp,seq);

figure
plot(tp.model.A,params.model.A,'xr')
figure
imagesc([tp.model.A params.model.A])
figure
plot(tp.model.Q,params.model.Q,'xr')
figure
imagesc([tp.model.Q params.model.Q])
figure
plot(tp.model.Q0,params.model.Q0,'xr')
figure
imagesc([tp.model.Q0 params.model.Q0])
figure
plot(tp.model.x0,params.model.x0,'xr')
 



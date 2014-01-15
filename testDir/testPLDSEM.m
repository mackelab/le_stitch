clear all
close all


xDim   = 10;
yDim   = 100;
T      = 100;
Trials = 100;


params = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim);

params.PiY = params.C*dlyap(params.A,params.Q)*params.C';

seq = PLDSsample(params,T,Trials);
ESTparams = params;

ESTparams.A  = eye(xDim)*0.9;
ESTparams.Q  = (1-0.9^2)*eye(xDim);
ESTparams.Q0 = dlyap(ESTparams.A,ESTparams.Q);
ESTparams.x0 = zeros(xDim,1);
ESTparams.C  = randn(yDim,xDim)*0.1/sqrt(xDim);
ESTparams.d  = -1.7*ones(yDim,1);


[NOWparams varBound] = PLDSEM(params,seq);


figure
plotMatrixSpectrum(params.A)
figure
plotMatrixSpectrum(ESTparams.A)

params.Pi = dlyap(params.A,params.Q)
ESTparams.Pi = dlyap(ESTparams.A,ESTparams.Q)
figure
plot(vec(params.C*params.Pi*params.C'),vec(ESTparams.C*ESTparams.Pi*ESTparams.C'),'xr')

figure
plot(vec(params.C*params.A*params.Pi*params.C'),vec(ESTparams.C*ESTparams.A*ESTparams.Pi*ESTparams.C'),'xr')


figure
plot(vec(params.C*params.Q0*params.C'),vec(ESTparams.C*ESTparams.Q*ESTparams.C'),'xr')

figure
plot(params.C*params.x0,ESTparams.C*ESTparams.x0,'xr')

subspace(params.C,ESTparams.C)



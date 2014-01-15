clear all
close all


xDim   = 5;
yDim   = 100;
T      = 200;
Trials = 20;


trueparams = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim);
trueparams.PiY = trueparams.C*dlyap(trueparams.A,trueparams.Q)*trueparams.C';
seq = PLDSsample(trueparams,T,Trials);
tp = trueparams;

params = PLDSInitialize(seq,xDim);


[params varBound] = PLDSEM(params,seq);


plotMatrixSpectrum(tp.A);
plotMatrixSpectrum(params.A);

tp.Pi = dlyap(tp.A,tp.Q);
params.Pi = dlyap(params.A,params.Q);
figure
plot(vec(tp.C*tp.Pi*tp.C'),vec(params.C*params.Pi*params.C'),'xr')

figure
plot(vec(tp.C*tp.A*tp.Pi*tp.C'),vec(params.C*params.A*params.Pi*params.C'),'xr')


figure
plot(vec(tp.C*tp.Q0*tp.C'),vec(params.C*params.Q*params.C'),'xr')

figure
plot(tp.C*tp.x0,params.C*params.x0,'xr')

subspace(tp.C,params.C)



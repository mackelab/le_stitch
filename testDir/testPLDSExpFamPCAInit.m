clear all
close all


xDim   = 5;
yDim   = 100;
T      = 100;
Trials = 50;


%%%% generate data

trueparams = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim);
trueparams.PiY = trueparams.C*dlyap(trueparams.A,trueparams.Q)*trueparams.C';
seq = PLDSsample(trueparams,T,Trials);
tp = trueparams;


%%% initialize with exponential family PCA

params = PLDSInitialize(seq,xDim,'params',params,'initMethod','ExpFamPCA');
subspace(tp.C,params.C)


%%% do 10 EM iterations

params.algorithmic.EMIterations.maxIter = 10;
[params varBound] = PLDSEM(params,seq);
subspace(tp.C,params.C)


%%% plot some diagnostics

%plotMatrixSpectrum(tp.A);
%plotMatrixSpectrum(params.A);

tp.Pi = dlyap(tp.A,tp.Q);
params.Pi = dlyap(params.A,params.Q);
figure
plot(vec(tp.C*tp.Pi*tp.C'),vec(params.C*params.Pi*params.C'),'xr')

figure
plot(vec(tp.C*tp.A*tp.Pi*tp.C'),vec(params.C*params.A*params.Pi*params.C'),'xr')


figure
plot(vec(tp.C*tp.Q0*tp.C'),vec(params.C*params.Q0*params.C'),'xr')

figure
plot(tp.C*tp.x0,params.C*params.x0,'xr')



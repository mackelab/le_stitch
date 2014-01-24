clear all
close all

 
xDim    = 3;
yDim    = 30;
T       = 100;
Trials  = 10;
maxIter = 25;

%%%% ground truth model

trueparams = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',0.0);
seqOrig    = PLDSsample(trueparams,T,Trials);
tp         = trueparams;


%%%% fitting a model to data

seq    = seqOrig;
params = [];
params = PLDSInitialize(seq,xDim,'initMethod','ExpFamPCA','params',params);

params.algorithmic.EMIterations.maxIter    = maxIter;
params.algorithmic.EMIterations.maxCPUTime = 50;
tic; [params seq varBound EStepTimes MStepTimes] = PLDSEM(params,seq); toc


%%%% compare models

subspace(tp.C,params.C)

sort(eig(tp.A))
sort(eig(params.A))

tp.Pi     = dlyap(tp.A,tp.Q);
params.Pi = dlyap(params.A,params.Q);

figure
plot(vec(tp.C*tp.Pi*tp.C'),vec(params.C*params.Pi*params.C'),'xr')

figure
plot(vec(tp.C*tp.A*tp.Pi*tp.C'),vec(params.C*params.A*params.Pi*params.C'),'xr')

figure
plot(tp.d,params.d,'rx');

figure
plot(vec(tp.C*tp.Q0*tp.C'),vec(params.C*params.Q*params.C'),'xr')

figure
plot(tp.C*tp.x0,params.C*params.x0,'xr')

clear all
close all


xDim   = 10;
yDim   = 100;
T      = 100;
Trials = 1500;


%%%% generate data

trueparams = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim);
trueparams.model.PiY = trueparams.model.C*dlyap(trueparams.model.A,trueparams.model.Q)*trueparams.model.C';
seq = PLDSsample(trueparams,T,Trials);
tp = trueparams;


%%% initialize with exponential family PCA

params = PLDSInitialize(seq,xDim,'ExpFamPCA',[]);


%%% do 10 EM iterations
%
%params.startParams = params;
%params.opts.algorithmic.EMIterations.maxIter = 10;
%[params varBound] = PLDSEM(params,seq);


%%% plot some diagnostics

subspace(tp.model.C,params.model.C)

sort(eig(tp.model.A))
sort(eig(params.model.A))

%{
tp.model.Pi     = dlyap(tp.model.A,tp.model.Q);
params.model.Pi = dlyap(params.model.A,params.model.Q);

figure
plot(vec(tp.model.C*tp.model.Pi*tp.model.C'),vec(params.model.C*params.model.Pi*params.model.C'),'xr')

figure
plot(vec(tp.model.C*tp.model.A*tp.model.Pi*tp.model.C'),vec(params.model.C*params.model.A*params.model.Pi*params.model.C'),'xr')

figure
plot(tp.model.d,params.model.d,'rx');

figure
plot(vec(tp.model.C*tp.model.Q0*tp.model.C'),vec(params.model.C*params.model.Q0*params.model.C'),'xr')

figure
plot(tp.model.C*tp.model.x0,params.model.C*params.model.x0,'xr')
%}


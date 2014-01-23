% This script presents an  example how to use the PLDS toolbox.
% 
%
%
% Lars Buesing, 2014
%


clear all
close all


% set parameters for the ground truth model
 
xDim    = 5;                % latent dimensiom
yDim    = 100;		    % observed dimension = no of neurons
T       = 100;		    % no of time bins per trial; here a time step is approx 10ms 
Trials  = 20;		    % no trials
maxIter = 25;		    % max no of EM iterations for fitting the model

%%%% ground truth model

trueparams = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',0.0);		% create ground truth model parameters
seqOrig    = PLDSsample(trueparams,T,Trials);								% sample from the model
tp         = trueparams;										


%%%% fitting a model to data

seq    = seqOrig;
params = [];
params = PLDSInitialize(seq,xDim,'initMethod','ExpFamPCA','params',params);				% initialize paramters

params.algorithmic.EMIterations.maxIter = maxIter;
tic; [params seq varBound EStepTimes MStepTimes] = PLDSEM(params,seq); toc				% EM iterations


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

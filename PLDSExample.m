% This script presents an  example how to use the PLDS toolbox.
% 
%
%
% Lars Buesing, 2014
%


clear all
close all


% set parameters for the ground truth model
 
xDim    = 3;			% latent dimensiom
yDim    = 30;		    	% observed dimension = no of neurons
T       = 100;		    	% no of time bins per trial; here a time step is approx 10ms 
Trials  = 5;		    	% no trials
maxIter = 5;		    	% max no of EM iterations for fitting the model

%%%% ground truth model

trueparams = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',-2.0);		% create ground truth model parameters
seqOrig    = PLDSsample(trueparams,T,Trials);								% sample from the model
tp         = trueparams;										
hist(mean([seqOrig.y]'))


%%%% fitting a model to data

seq    = seqOrig;
params = [];
params = PLDSInitialize(seq,xDim,'ExpFamPCA',params);

params.opts.algorithmic.EMIterations.maxIter     = maxIter;						%
params.opts.algorithmic.EMIterations.maxCPUTime  = 600;							% allow for 600s of EM
tic; [params seq varBound EStepTimes MStepTimes] = PopSpikeEM(params,seq); toc


%%%% compare models

subspace(tp.model.C,params.model.C)

sort(eig(tp.model.A))
sort(eig(params.model.A))

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

clear all
close all

addpath('/nfs/nhome/live/lars/projects/dynamics/pair/HNLDS/matlab/PPGPFA/core_lds')
addpath('/nfs/nhome/live/lars/projects/dynamics/pair/HNLDS/matlab/PPGPFA/util')


xDim   = 10;
yDim   = 100;

T      = 200;  % also check if the transformed systems have the same liklihood, sensitive test;
Trials = 3;    


trueparams = generateLDS('xDim',xDim,'yDim',yDim);
seqOrig    = sampleLDS(trueparams,T,Trials)

tp = trueparams;
tp.PiY = tp.C*tp.Pi*tp.C';
tp.notes.forceEqualT = true;
tp.notes.useB = false;
tp.notes.useD = false;
tp.notes.type = 'LDS';

%%%%%%%%% test parameter transformation

% random transformation works!

[params] = LDSApplyParamsTransformation(randn(xDim)+0.1*eye(xDim),tp);
[params] = LDSTransformParams(params,'TransformType','2');

params.PiY = params.C*params.Pi*params.C';

figure
plot(vec(tp.PiY),vec(params.PiY),'rx');

figure    
plot(vec(tp.C*tp.A*tp.Pi*tp.C'),vec(params.C*params.A*params.Pi*params.C'),'rx');

figure
plot(vec(tp.C*tp.Q0*tp.C'),vec(params.C*params.Q0*params.C'),'rx');

figure
plot(vec(tp.C*tp.x0),vec(params.C*params.x0),'rx');


subspace(tp.C,params.C)

sort(eig(tp.A))
sort(eig(params.A))


tp.xo = tp.x0;tp.Qo = tp.Q0;
params.xo = params.x0;params.Qo = params.Q0;
[tpseq, tpLL] = exactInferenceLDS(seqOrig, tp,'getLL',true);
[seq,   LL]   = exactInferenceLDS(seqOrig, params,'getLL',true);

sum(abs(tpLL-LL))./sum(abs(tpLL))

params.C'*params.C
dlyap(params.A,params.Q)
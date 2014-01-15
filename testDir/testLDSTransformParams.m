clear all
close all


xDim   = 5;
yDim   = 100;
T      = 100;
Trials = 3;


trueparams = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim);
trueparams.PiY = trueparams.C*dlyap(trueparams.A,trueparams.Q)*trueparams.C';
seq = PLDSsample(trueparams,T,Trials); Oseq = seq;
tp = trueparams;

Y = [seq.y];

params    = PLDSInitialize(seq,xDim,'initMethod','ExpFamPCA');
params.algorithmic.EMIterations.maxIter = 10;
iparams = params;
[params varBound] = PLDSEM(params,seq);
tparams = iparams;
tparams.algorithmic.TransformType = '2';
[tparams tvarBound] = PLDSEM(tparams,seq);



%tparams = LDSTransformParams(params,'TransformType','1'); 


%%%%%% different diagnostics if parameter transformation is equivalent
% works

subspace(tparams.C,params.C)

sort(eig(params.A))
sort(eig(tparams.A))

tparams.C'*tparams.C 

dlyap(tparams.A,tparams.Q)



%{

seq  = PLDSVariationalInference(params,Oseq);
tseq = PLDSVariationalInference(tparams,Oseq);

for tr=1:Trials
    abs(seq(tr).posterior.varBound-tseq(tr).posterior.varBound)./abs(seq(tr).posterior.varBound)

end

tparams.Pi = dlyap(tparams.A,tparams.Q);
params.Pi = dlyap(params.A,params.Q);
figure
plot(vec(tparams.C*tparams.Pi*tparams.C'),vec(params.C*params.Pi*params.C'),'xr')

figure
plot(vec(tparams.C*tparams.A*tparams.Pi*tparams.C'),vec(params.C*params.A*params.Pi*params.C'),'xr')


figure
plot(vec(tparams.C*tparams.Q0*tparams.C'),vec(params.C*params.Q0*params.C'),'xr')

figure
plot(tparams.C*tparams.x0,params.C*params.x0,'xr')
%}
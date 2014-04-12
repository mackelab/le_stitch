clear all
close all

%rng('default');

uDim   = 2;   
xDim   = 10;
yDim   = 100;
T      = 100;
Trials = 1;


params  = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'uDim',uDim,'doff',-1.75);
seqOrig = PLDSsample(params,T,Trials);


tic
seqVarInf = PLDSVariationalInference(params,seqOrig);
toc

tic
seqLpInf  = PLDSLaplaceInference(params,seqOrig);
toc

Mu = getPriorMeanLDS(params,T,'seq',seqOrig(1));
norm(seqVarInf.posterior.xsm-seqLpInf.posterior.xsm,'fro')/norm(seqVarInf.posterior.xsm-Mu,'fro')



figure;
plot(vec(seqVarInf.posterior.Vsm),vec(seqLpInf.posterior.Vsm),'rx')

figure;
plot(vec(seqVarInf.posterior.xsm),vec(seqLpInf.posterior.xsm),'rx')

figure;
plot(vec(seqVarInf.posterior.VVsm),vec(seqLpInf.posterior.VVsm),'rx')


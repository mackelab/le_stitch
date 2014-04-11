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
seqLpInf  = PLDSlpinf(params,seqOrig);
toc

Mu = getPriorMeanLDS(params,T,'seq',seqOrig(1));
norm(seqVarInf.posterior.xsm-seqLpInf.posterior.xsm,'fro')/ ...
    norm(seqVarInf.posterior.xsm-Mu,'fro')



figure;
hold on
plot(vec(seqVarInf.posterior.Vsm),vec(seqLpInf.posterior.Vsm),'rx')

%{
figure; hold on
plot(seqVarInf.posterior.xsm','k')
plot(seqLpInf.posterior.xsm','r--')
figure
plot(Mu')
%}




%{
seqVarInf2 = seqOrig;
seqVarInf2.posterior.lamOpt = seqLpInf.posterior.lamOpt;
params.opts.algorithmic.VarInfX.minFuncOptions.display = 'iter';

tic
seqVarInf2 = PLDSVariationalInference(params,seqVarInf2);
toc
%}


%{
%%%% outer loop over trials

options.Method      = 'newton';
options.MaxIter     = 100;
options.maxFunEvals = 1000;

for tr=1:Trials

    T = size(seq(tr).y,2);
    
    dlarge = repamat(params,d,T,1);
    
    Cl = {}; for t=1:T; Cl = {Cl{:} params.C}; end
    W = sparse(blkdiag(Cl{:}));

    Lambda = buildPriorPrecisionMatrixFromLDS(params,T);

    mu = zeros(xDim,T);
    mu(:,1) = params.x0;
    for t=2:T; mu(:,t) = params.A*mu(:,t-1); end

    y = vec(seq(tr).y);
    xInit = zeros(xDim*T,1);

    xsm = minFunc(@PLDSLaplaceCost,xInit,options,y,Lambda,mu,W,dlarge);

end 

%}
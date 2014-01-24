clear all
close all


xDim   = 3;
yDim   = 30;
T      = [100];
Trials = 1;


params  = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',0.0);
seqOrig = PLDSsample(params,T,Trials);
seqVar  = feval(params.inferenceHandle,params,seqOrig)
seq     = seqOrig;

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


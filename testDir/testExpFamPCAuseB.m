clear all
close all

uDim   = 2;
xDim   = 5;
yDim   = 100;

T      = 100;
Trials = 10;

tp  = PLDSgenerateExample('xDim',xDim,'yDim',yDim,'uDim',uDim);
seq = PLDSsample(tp,T,Trials);

params.model.useB = true;
params = PLDSInitialize(seq,xDim,'ExpFamPCA',params)

%{
%mean(vec([seq.y]))
%figure
%plot(seq(1).x')
%figure
%imagesc(seq(1).y)

%options.display     = 'none';
options.MaxIter     = 10000;
options.maxFunEvals = 50000;
options.Method      = 'lbfgs';
options.progTol     = 1e-9;
options.optTol      = 1e-5;


[Cest, Xest, dest] = ExpFamPCA([seq.y],xDim,'options',options,'dt',10,'lam',1);
subspace(params.model.C,Cest) 




Cest'*Cest
Xest*Xest'./T

[UC SC VC] = svd(Cest);
M = SC(1:xDim,1:xDim)*VC(:,1:xDim)';
Cest = Cest/M;
Xest = M*Xest;
U = C'*Cest;
Cest = Cest/U;
Xest = U*Xest;


figure
plot(vec(X),vec(Xest),'rx');

figure
plot(vec(d),vec(dest),'rx'); 


%}
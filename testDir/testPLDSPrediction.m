clear all
close all

 
xDim    = 3;
yDim    = 30;
T       = 200;
Trials  = 1;


%%%% ground truth model

trueparams = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',0.0);
tp         = trueparams;

UA = randn(xDim); UA = (UA-UA')/2;
tp.A       = expm(0.1*UA)*0.995*eye(xDim)
eig(tp.A)

seqOrig = PLDSsample(tp,T,Trials);
seq     = seqOrig;
params  = tp;


%%%% prediction example

y = seq(1).y;

condRange = [50:125];
predRange = [135:190];

tic; [ypred xpred xpredCov seqInf] = PLDSPredictRange(params,y,condRange,predRange); toc

figure
imagesc([y(:,predRange) ypred])


figure; hold on
plot(seq(1).x','k')
plot(condRange,seqInf(1).posterior.xsm','b')
plot(predRange,xpred','r')


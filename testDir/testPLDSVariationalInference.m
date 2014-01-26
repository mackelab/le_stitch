clear all
close all


xDim   = 10;
yDim   = 150;
T      = [150];
Trials = 1;


params = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',-2.0);
seq = PLDSsample(params,T,Trials);

tic
seq = PLDSVariationalInference(params,seq);
toc

plotPosterior(seq,1,params);

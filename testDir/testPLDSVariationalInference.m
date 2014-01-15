clear all
close all


xDim   = 5;
yDim   = 50;
T      = 150;
Trials = 2;


params = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim);
seq = PLDSsample(params,T,Trials);

tic
seq = PLDSVariationalInference(seq,params);
toc

plotPosterior(seq,1,params);
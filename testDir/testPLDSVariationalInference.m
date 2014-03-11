clear all
close all

uDim   = 3;
xDim   = 10;
yDim   = 150;
T      = [150];
Trials = 1;


params = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',-3.5,'uDim',uDim);
params.model.B = params.model.B*2;
seq = PLDSsample(params,T,Trials);
max(vec([seq.y]))


tic
seq = PLDSVariationalInference(params,seq);
toc

plotPosterior(seq,1,params);


mu = zeros(xDim,T);
mu(:,1) = params.model.B*seq(1).u(:,1);
for t=2:T
  mu(:,t) = params.model.A*mu(:,t-1)+params.model.B*seq(1).u(:,t);
end
figure
plot(mu')


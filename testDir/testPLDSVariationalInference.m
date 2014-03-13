clear all
close all

uDim   = 0;
xDim   = 10;
yDim   = 150;
T      = [150];
Trials = 1;

for tr=1:Trials
  s{tr} = (vec(repmat(rand(1,floor(T(tr)/10))>0.5,10,1))-0.5);
  s{tr} = [s{tr}' zeros(1,T(tr)-floor(T(tr)/10)*10)];
  s{tr} = repmat(s{tr},yDim,1);
  s{tr}(1:20,:) = s{tr}(1:20,:)*2;
end


params = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',-1.5,'uDim',uDim);
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


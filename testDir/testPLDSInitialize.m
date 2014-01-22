clear all
close all


xDim   = 10;
yDim   = 100;
T      = 100;
Trials = 50;


trueparams = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',-2.0);
seq = PLDSsample(trueparams,T,Trials);
mean(vec([seq.y]))
tp  = trueparams;

params = [];
params = PLDSInitialize(seq,xDim,'initMethod','ExpFamPCA','params',params);
% params = PLDSInitialize(seq,xDim,'initMethod','params','params',params);
params = PLDSInitialize(seq,xDim,'initMethod','PLDSID','params',params);

% [params varBound] = PLDSEM(params,seq);

subspace(tp.C,params.C)
sort(abs(eig(tp.A)))
sort(abs(eig(params.A)))

tp.Pi     = dlyap(tp.A,tp.Q);
params.Pi = dlyap(params.A,params.Q);
figure
plot(vec(tp.C*tp.Pi*tp.C'),vec(params.C*params.Pi*params.C'),'xr')

figure
plot(vec(tp.C*tp.A*tp.Pi*tp.C'),vec(params.C*params.A*params.Pi*params.C'),'xr')

figure
plot(tp.d,params.d,'rx');

%% useless
%figure
%plot(vec(tp.C*tp.Q0*tp.C'),vec(params.C*params.Q*params.C'),'xr')

%figure
%plot(tp.C*tp.x0,params.C*params.x0,'xr')




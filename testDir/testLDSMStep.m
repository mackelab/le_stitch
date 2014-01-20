clear all
close all

xDim   = 10;
yDim   = 25;
T      = 150;
Trials = 50;

params = generateLDS('xDim',xDim,'yDim',yDim);
seq = sampleLDS(params,T,Trials);


%%%%%%%%%%%%%%%%%5 test Kalman smoother %%%%%%%%%%%%%%%%
% ---> works

[seq Lambda LambdaPost] = simpleKalmanSmoother(params,seq);
plotPosterior(seq,1,params)

Sig = pinv(full(LambdaPost));
VsmFull = zeros(xDim*T,xDim);

for t=1:T
    xidx = ((t-1)*xDim+1):(t*xDim);
    VsmFull(xidx,:) = Sig(xidx,xidx);
end


max(abs(vec(VsmFull-seq(1).posterior.Vsm)))
figure
imagesc([VsmFull seq(1).posterior.Vsm])
figure
imagesc([VsmFull])
figure
imagesc([seq(1).posterior.Vsm])


%%%%%%%%%%%%%%%%%%% test LDS Mstep %%%%%%%%%%%%%%%%%%
% --> works!

ESTparams = MStepLDS(params,seq);

figure
imagesc([params.A ESTparams.A])
figure
imagesc([params.Q ESTparams.Q]) 
figure
imagesc([params.Q0 ESTparams.Q0])
figure
plot(params.x0,ESTparams.x0,'xr')
 



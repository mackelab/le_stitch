clear all
close all


xDim   = 3;
yDim   = 100;
T      = 200;
Trials = 3;
Iters  = 100;

params = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim);
params.R = diag(rand(yDim,1))+0.1;

seqOrig = sampleLDS(params,T,Trials);

seq = seqOrig;
[seq Lambda LambdaPost] = simpleKalmanSmoother(params,seq);

%plotPosterior(seq,1);

tic
SigFull = pinv(full(LambdaPost));
toc


VsmFull  = zeros(size(seq(1).posterior.Vsm));
VVsmFull = zeros(size(seq(1).posterior.VVsm));

for t=1:T
    xidx = (t-1)*xDim+1:t*xDim;
    VsmFull(xidx,:) = SigFull(xidx,xidx);
end

for t=1:T-1
    xidx = (t-1)*xDim+1:t*xDim;
    VVsmFull(xidx,:) = SigFull(xidx+xDim,xidx);
end

max(abs(vec(seq(1).posterior.Vsm-VsmFull)))
max(abs(vec(seq(1).posterior.VVsm-VVsmFull)))


% compare to previous code

paramsC = params;
paramsC.Qo = paramsC.Q0;
paramsC.xo = paramsC.x0;
seqC = seqOrig;
paramsC.notes.forceEqualT= true;
addpath('/nfs/nhome/live/lars/projects/dynamics/pair/HNLDS/matlab/PPGPFA/core_lds')
addpath('/nfs/nhome/live/lars/projects/dynamics/pair/HNLDS/matlab/PPGPFA/util')  
[seqC] = exactInferenceLDS(seqC, paramsC)

%seqC = kalmanSmootherLDS(seqC, paramsC);

figure; hold on
plot(seq(1).posterior.xsm(1,:))
plot(seqC(1).xsm(1,:),'r')

seqC(1).Vsm = reshape(seqC(1).Vsm,xDim,xDim*T)';
seqC(1).VVsm = reshape(permute(seqC(1).VVsm(:,:,2:end),[2 1 3]),xDim,xDim*(T-1))';

figure
imagesc(seqC(1).VVsm(1:xDim,1:xDim))
figure
imagesc(seq(1).posterior.VVsm(1:xDim,1:xDim))


max(abs(vec(seq(1).posterior.Vsm-seqC(1).Vsm)))
max(abs(vec(seq(1).posterior.VVsm-seqC(1).VVsm)))
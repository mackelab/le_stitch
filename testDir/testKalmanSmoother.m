clear all
close all


xDim   = 10;
yDim   = 100;
T      = 100;
Trials = 3;
Iters  = 100;

params = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim);

Lambda = buildPriorPrecisionMatrixFromLDS(params,T);

Rinv = diag(rand(yDim,1)+0.1);

CRinvCFull = zeros(T*xDim,T*xDim);
CRinvC     = zeros(T*xDim,xDim);

crinvc     = params.C'*Rinv*params.C;

for t=1:T
    xidx = (t-1)*xDim+1:t*xDim;
    CRinvCFull(xidx,xidx) = crinvc;
    CRinvC(xidx,:) = crinvc;

end


SigFull = pinv(Lambda+CRinvCFull);
[Vsm VVsm] = smoothedKalmanMatrices(params,CRinvC);

VsmFull = zeros(size(Vsm));
VVsmFull = zeros(size(VVsm));

for t=1:T
    xidx = (t-1)*xDim+1:t*xDim;
    VsmFull(xidx,:) = SigFull(xidx,xidx);
end

for t=1:T-1
    xidx = (t-1)*xDim+1:t*xDim;
    VVsmFull(xidx,:) = SigFull(xidx+xDim,xidx);
end

max(abs(vec(Vsm-VsmFull)))
max(abs(vec(VVsm-VVsmFull)))

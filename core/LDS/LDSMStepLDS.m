function params = MStepLDS(params,seq)
%
% function params = MStepLDS(params,seq)
%
% to do:
%
%	- stimulus drive B 
%
% Parameters to update: A,Q,Q0,x0,B
%
%
% (c) L Buesing 2014


xDim    = size(params.model.A,1);
Trials  = numel(seq);


%% compute posterior statistics

S11 = zeros(xDim,xDim);
S01 = zeros(xDim,xDim);
S00 = zeros(xDim,xDim);

x0 = zeros(xDim,Trials);
Q0 = zeros(xDim,xDim);

Tall = [];

for tr = 1:Trials

    T = size(seq(tr).y,2);
    Tall  = [Tall T];

    Vsm   = reshape(seq(tr).posterior.Vsm' ,xDim,xDim,T);
    VVsm  = reshape(seq(tr).posterior.VVsm',xDim,xDim,T-1);
    
    MUsm0 = seq(tr).posterior.xsm(:,1:T-1);
    MUsm1 = seq(tr).posterior.xsm(:,2:T);

    S00   = S00 + sum(Vsm(:,:,1:T-1),3)  + MUsm0*MUsm0';
    S01   = S01 + sum(VVsm(:,:,1:T-1),3) + MUsm0*MUsm1';
    S11   = S11 + sum(Vsm(:,:,2:T),3)    + MUsm1*MUsm1';

    x0(:,tr) = MUsm0(:,1);
    Q0 = Q0 + Vsm(:,:,1);

end

S00 = (S00+S00')/2;
S11 = (S11+S11')/2;

params.model.A  = S01'/S00;
params.model.Q  = (S11+params.model.A*S00*params.model.A'-S01'*params.model.A'-params.model.A*S01)./(sum(Tall)-Trials);
params.model.Q  = (params.model.Q+params.model.Q')/2;

params.model.x0 = mean(x0,2);
x0dev     = bsxfun(@minus,x0,params.model.x0);
params.model.Q0 = (Q0 + x0dev*x0dev')./Trials;
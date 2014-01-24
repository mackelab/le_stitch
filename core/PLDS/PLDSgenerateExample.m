function params = PLDSgenerateExample(varargin)
%
% trueparams = PLDSgenerateExample(varargin)
%

xDim     = 10;
yDim     = 100;

Aspec    = 0.99;
Arand    = 0.03;
Q0max    = 0.3;
BernFlag = false;
doff     = -1.9;

assignopts(who,varargin);


%%%%%%%%%  generate parameters %%%%%%%%% 

A  = eye(xDim)+Arand*randn(xDim);
A  = A./max(abs(eig(A)))*Aspec;
Q  = diag(rand(xDim,1));
Q0 = dlyap(A,Q);
M  = diag(1./sqrt(diag(Q0)));
A  = M*A*pinv(M);
Q  = M*Q*M'; Q=(Q+Q)/2;

O  = orth(randn(xDim));
Q0 = O*diag(rand(xDim,1)*Q0max)*O'/3;
x0 = randn(xDim,1)/3;

C  = randn(yDim,xDim)./sqrt(3*xDim);
d  = 0.3*randn(yDim,1)+doff;

params.model.A    = A;
params.model.Q    = Q;
params.model.Q0   = Q0;
params.model.x0   = x0;
params.model.C    = C;
params.model.d    = d;


if BernFlag
    params.model.dualHandle = @LogisticBernoulliDualHandle;
    params.model.likeHandle = @LogisticBernoulliHandle;
else
    params.model.dualHandle = @ExpPoissonDualHandle;
    params.model.likeHandle = @ExpPoissonHandle;
end

params.model.inferenceHandle = @PLDSVariationalInference;

params = PLDSsetDefaultParameters(params,xDim,yDim);

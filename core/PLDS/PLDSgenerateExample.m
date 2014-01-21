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
d  = 0.3*doff*randn(yDim,1)+doff;

params.A    = A;
params.Q    = Q;
params.Q0   = Q0;
params.x0   = x0;
params.C    = C;
params.d    = d;


if BernFlag
    params.dualHandle = @LogisticBernoulliDualHandle;
    params.likeHandle = @LogisticBernoulliHandle;
else
    params.dualHandle = @ExpPoissonDualHandle;
    params.likeHandle = @ExpPoissonHandle;
end

params = PLDSsetDefaultParameters(params,xDim,yDim);

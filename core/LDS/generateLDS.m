function params = generateLDS(varargin)
%
% trueparams = generateLDS(varargin)
%

xDim     = 10;
yDim     = 100;

Aspec    = 0.99;
Arand    = 0.03;
Q0max    = 0.3;
Rmin     = 0.1;
Rmax     = 0.1;


T        = 100;
Trials   = 10;


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
Q0 = O*diag(rand(xDim,1)*Q0max)*O'/100;
x0 = randn(xDim,1);

C  = randn(yDim,xDim)./sqrt(xDim);
R  = diag(rand(yDim,1)*Rmax+Rmin);
d  = randn(yDim,1);

params.A    = A;
params.Q    = Q;
params.Q0   = Q0;
params.x0   = x0;
params.C    = C;
params.d    = d;
params.R    = R;
params.Pi   = dlyap(params.A,params.Q);

function [C, X, d] = ExpFamPCA(Y,xDim,varargin)
%
% [C, X, d] = ExpFamPCA(Y,xDim)
%
% to do:
%
%  - generalize to different observation models
%


dt  = 10;    % rebin factor %!!! this is very much dependent on the firing rates
lam = 1e-1;  % penalizer

options.display     = 'none';
options.MaxIter     = 10000;
options.maxFunEvals = 50000;
options.Method      = 'lbfgs';
options.progTol     = 1e-9;
options.optTol      = 1e-5;

assignopts(who,varargin);


seqDum.y = Y;
seqDum = rebinRaster(seqDum,dt);
Y = seqDum.y;

[yDim T] = size(Y);

my = mean(Y,2);
[Uy Sy Vy] = svd(bsxfun(@minus,Y,my),0);
my = max(my,0.1);

Cinit = Uy(:,1:xDim);
Xinit = zeros(xDim,T);
dinit = log(my);%zeros(yDim,1);

CXdinit = [vec([Cinit; Xinit']); dinit];
CXdOpt  = minFunc(@ExpFamPCACost,CXdinit,options,Y,xDim,lam); 

d  = CXdOpt(end-yDim+1:end);
CX = reshape(CXdOpt(1:end-yDim),yDim+T,xDim);
C  = CX(1:yDim,:);
X  = CX(yDim+1:end,:)';


% transform for C to have orthonormal columns
[UC SC VC] = svd(C);
M = SC(1:xDim,1:xDim)*VC(:,1:xDim)';
C = C/M;
X = M*X;

% transform for X to have orthogonal rows
[MU MS MV] = svd((X*X')./T);
M = MU';
C = C/M;
X = M*X;
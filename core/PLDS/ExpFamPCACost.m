function [f df] = ExpFamPCACost(CXd,Y,xDim,lambda)
%
% [f df] = ExpFamPCACost(CXd,Y,xDim,lambda)
%
% (c) L Buesing 2014

[yDim T] = size(Y);


d  = CXd(end-yDim+1:end);
CX = reshape(CXd(1:end-yDim),yDim+T,xDim);
C  = CX(1:yDim,:);
X  = CX(yDim+1:end,:)';

nu = bsxfun(@plus,C*X,d);
Yhat = exp(nu);

f = sum(vec(-Y.*nu+Yhat))+lambda/2*(norm(C,'fro')^2+norm(X,'fro')+norm(d)^2);

YhatmY = Yhat-Y;

gX = C'*YhatmY+lambda*X;
gC = YhatmY*X'+lambda*C;
gd = sum(YhatmY,2)+lambda*d;

df = [vec([gC;gX']);gd];


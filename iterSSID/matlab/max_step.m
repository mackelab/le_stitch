function theta_max = max_step(X,dX)
% theta = MAX_STEP(X,dX)
%
% Given n-by-n symmetric matrices, X and dX, where X is positive definite,
% MAX_STEP returns the largest theta_max > 0 such that
%
%   X + theta*dX
%
% is positive definite for all 0 < theta < theta_max. If X + theta*dX is
% positive definite for all theta, then theta_max = Inf.
x = max(eig(-dX,X,'chol'));
if x > 0, theta_max = 1/x; else theta_max = Inf; end
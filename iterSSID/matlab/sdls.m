function [X,Y,norm_res,muv,tt,iter,fail]  = sdls(A,B,X0,Y0,tol,verbose)
% [X,Y,norm_res,muv,tt,iter,fail] = SDLS(A,B,X0,Y0,tol,verbose)
%
% Uses a stanadard path-following interior-point method based on the
% AHO search direction to solve the symmetric semidefinite constrained
% least squares problem:
%
%   min  norm(A*X-B,’fro’)
%   s.t. X symm. pos. semidef.
%
% where A and B are real m-by-n matrices, and X is a real n-by-n matrix.
%
% X0 and Y0 are n-by-n initial strictly feasible matrices, which means
% that X0 and Y0 are symmetric positive definite.
% Set as [] for the default value of eye(n).
%
% tol is the zero tolerance described below.
% Set as [] for the default value of 1e-10.
%
% Set verbose = 1 for screen output during algorithm execution,
% otherwise set vebose = 0 for no output.
%
% SDLS returns approximate optimal solutions to the above primal
% problem and its associated dual problem so that
%
%   norm(res,’fro’)  <=  sqrt(tol)*norm(res0,’fro’)
%        trace(X*Y)  <=  tol*trace(X0*Y0)
%
% where res = (Z+Z’)/2-Y, Z =  A’*(A*X-B), and res0 is res evaluated Appendix B.  Matlab M-files
% at X0, Y0.
%
% SDLS optionally returns:
%
%  norm_res : norm(res,’fro’) at the final iterate,
%  muv  : a vector of the duality gaps for each iteration
%  tt   : the total running time of the algorithm
%  iter : the number of iterations required
%  fail : fail = 1 if the algorithm failed to achieve the desired
%         tolerances within the maximum number of iterations allowed;
%         otherwise fail = 0
% Nathan Krislock, University of British Columbia, 2003.
%
% N. Krislock. Numerical solution of semidefinite constrained least
% squares problems.  M.Sc. thesis, University of British Columbia,
% Vancouver, British Columbia, Canada, 2003.
tic;  % Start the preprocessing timer
MaxIt = 100;  % max iteration
[m,n] = size(A);
AA = A'*A;  AB = A'*B;  I = eye(n);
if isempty(X0), X = I; else X = X0; end
if isempty(Y0), Y = I; else Y = Y0; end
if isempty(tol), tol = 1e-10; end
XAA = X*AA;  XY = X*Y;
Z = XAA' - AB;  Z = (Z+Z')/2;  R = Z - Y;
norm_res = norm(R,'fro');  mu = trace(XY)/n;  muv = mu;
tol1 = sqrt(tol)*norm_res;  tol2 = tol*mu;
r = 0;  theta = 0;
if verbose==1
disp(' ');
disp(['  r      sigma         theta        norm(res)     <X,Y>/n']);
disp([' ---   ----------    ----------    ----------    ----------']);
ol = sprintf('%3d',r);
ol = [ ol, sprintf('                            ') ];
ol = [ ol, sprintf('  %12.4e',[norm_res, mu]) ];
disp(ol);
end
pretime = toc;  % End the preprocessing timer
while ( norm_res > tol1 || mu > tol2 ) && r < MaxIt
    tic;  % Start the iteration timer
    % Compute sigma and tau
    if norm_res < tol1
    sigma = 1/n^2;
    else
    sigma = 1-1/n^2;
    end
    tau = sigma*mu;
    % Compute the AHO search direction (dX,dY)
    E = (kron(I,Y)+kron(Y,I))/2;  % F = (kron(I,X)+kron(X,I))/2;
    XZ = X*Z;
    M = (kron(I,XAA)+kron(AA,X)+kron(X,AA)+kron(XAA,I))/4 + E;
    % M = F*kronAA + E;
    d = vec(tau*I - (XZ+XZ')/2);
    % d = F*vec(-R) + vec(tau*I-(X*Y+Y*X)/2);
    [L,U,P] = lu(M);
    dx = U\(L\(P*d));
    dX = mat(dx);  dX = (dX+dX')/2;  AAdX = AA*dX;
    dY = (AAdX+AAdX')/2 + R;  dY = (dY+dY')/2;
    % Compute the step length theta
    c = 0.9 + 0.09*theta;
    theta1 = max_step(X,dX);  theta2 = max_step(Y,dY);
    theta_max = min([theta1, theta2]);
    theta = min([c*theta_max,1]);
    % Update
    X = X + theta*dX;
    Y = Y + theta*dY;
    XAA = X*AA;  XY = X*Y;
    Z = XAA' - AB;  Z = (Z+Z')/2;  R = Z - Y;
    norm_res = norm(R,'fro');  mu = trace(XY)/n;  muv(r+2) = mu;
    r = r + 1;
    if verbose==1
    ol = sprintf('%3d',r);
    ol = [ ol, sprintf('  %12.4e',[sigma, theta, norm_res, mu]) ];
    disp(ol);
    end
    t(r) = toc;  % End the iteration timer
end
if r==MaxIt && ( norm_res > tol1 || mu > tol2 )
fail = 1;
else
fail = 0;
end
avt = mean(t);  tt = sum(t) + pretime;  iter = r;
if verbose==1
if fail==1,
disp(' ');
disp('Failed to reach desired tolerance.');
end
disp(' ');
disp(sprintf('Average iteration time:\t%2.4f seconds', avt));
disp(sprintf('Total time:\t\t%2.4f seconds', tt));
disp(' ');
end

function [Y,Xu,Xs,Xv,d] = MODnucnrmminWithd( S, opts )
%
% [Y,Xu,Xs,Xv,d] = MODnucnrmminWithd( S, opts )
%
%  obj function is -log(S|x) + lambda||X||_* + Tr[Z'(x-d-X)] + rho/2||x-d-X||^2_F
% Nuclear Norm Minimization with additive offset to model mean firing
% 
% opts -
%   rho - dual gradient ascent rate for ADMM
%   eps_abs - absolute threshold for ADMM
%   eps_rel - relative threshold for ADMM
%   lambda - strength of the nuclear norm penalty. multiplied by the square
%       root of the number of elements in y to make sure it properly scales
%       as the size of the problem increases
%   maxIter - maximum number of iterations to run if the stopping threshold
%       is not reached
%   nlin - the nonlinearity to be used in the Poisson log likelihood:
%       exp - exponential
%       soft - exponential for negative x, linear otherwise
%       logexp - log(1+exp(x)), a smooth version of "soft"
%   verbose - verbosity level.
%       0 - print nothing
%       1 - print every outer loop of ADMM
%       2 - print every inner loop of Newton's method and ADMM
%
% David Pfau, 2012-2013
% minor modifications by Lars Buesing, 2014
%

% set default values
rho     = 0;
eps_abs = 0;
eps_rel = 0;
maxIter = 0;
nlin    = 0;
lambda  = 0;
verbose = 0;


for field = {'rho','eps_abs','eps_rel','maxIter','nlin','lambda','verbose'}
    eval([field{1} ' = opts.' field{1} ';'])
end

%nz = logical(sum(S,2));
%S = S(nz,:); % remove rows with no spikes

[N,T] = size(S);


switch nlin
    case 'exp'
        f = @exp;
    case 'soft'
        f   = @(x) exp(x).*(x<0) + (1+x).*(x>=0);
        df  = @(x) exp(x).*(x<0) + (x>=0);
        d2f = @(x) exp(x).*(x<0);
    case 'logexp'
        f   = @(x) log(1+exp(x));
        df  = @(x) 1./(1+exp(-x));
        d2f = @(x) exp(-x)./(1+exp(-x)).^2;
end

switch nlin  % crude ADMM initialization
    case 'exp'
        x = log(max(S,1));
    case {'soft','logexp'}
        x = max(S-1,0);
end

X = zeros(N,T);
Z = zeros(N,T);
d = mean(x,2);
x = bsxfun(@minus,x,d);

nr_p = Inf; nr_d= Inf;
e_p = 0;    e_d = 0;
iter = 0;
if verbose>0;
   fprintf('Iter:\t Nuc nrm:\t Loglik:\t Objective:\t dX:\t\t r_p:\t\t e_p:\t\t r_d:\t\t e_d:\n')
end

while ( nr_p > e_p || nr_d > e_d ) && iter < maxIter % Outer loop of ADMM
    stopping = Inf; % stopping criterion
    x_old = x;
    if verbose == 2, fprintf('\tNewton:\t Obj\t\t Stopping\t\n'); end
    while stopping/norm(x,'fro') > 1e-6 % Outer loop of Newton's method
        switch nlin
            case 'exp'
                h =  exp( bsxfun(@plus,x,d) ); % diagonal of Hessian
                g =  exp( bsxfun(@plus,x,d) ) - S; % gradient
            otherwise
		warning('not implemented')
                h =  d2f( x ) - S .* ( d2f( x ) .* f( x ) - df( x ).^2 ) ./ f( x ).^2;
                g =  df ( x ) - S .* df( x ) ./ f( x );
                g(isnan(g)) = 0;
                h(isnan(h)) = 0;
        end
        
        grad = g + rho*(x-X) + Z;
        dx = -inv_hess_mult(h,grad);
        
	gd = sum(g,2)+lambda*d;
	hd = sum(h,2)+lambda;
	dd = -gd./hd;

	% upadate
	x = x + dx;
	d = d + dd;	

        stopping = abs(grad(:)'*dx(:)+dd'*gd);
        if verbose == 2 % verbosity level
            fprintf('\t\t %1.2e \t %1.2e\n', obj(x), stopping)
        end
    end
    dx = norm(x_old-x,'fro')/norm(x,'fro');

        
    if T > N
        [v,s,u] = svd( x' + Z'/rho, 0 );
    else
        [u,s,v] = svd( x + Z/rho, 0);
    end

    Xs = max( s - eye(min(N,T))*lambda/rho, 0 );
    Xu = u;
    Xv = v;

    X_ = u*max( s - eye(min(N,T))*lambda/rho, 0 )*v';
    
    Z_ = Z + rho * ( x - X_ );
    
    % compute residuals and thresholds
    r_p = x - X_;
    r_d = rho * ( X - X_ );
    e_p = sqrt(N*T) * eps_abs + eps_rel * max( norm( x, 'fro' ), norm( X_, 'fro' ) );
    e_d = sqrt(N*T) * eps_abs + eps_rel * norm( Z , 'fro' );
    nr_p = norm( r_p, 'fro' );
    nr_d = norm( r_d, 'fro' );
    
    % heuristics to adjust dual gradient ascent rate to balance primal and
    % dual convergence. Seems to work pretty well: we almost always
    % converge in under 100 iterations.
    if nr_p > 10*nr_d
        rho = 2*rho;
    elseif nr_d > 10*nr_p
        rho = rho/2;
    end
    
    % update
    X = X_;
    Z = Z_;
    
    fval = sum( sum( f( bsxfun(@plus,x,d) ) - S .* log( f( bsxfun(@plus,x,d) ) ) ) );
    nn = sum( svd( x ) );
    
    % print
    iter = iter + 1;
    if verbose>0
       fprintf('%i\t %1.4e\t %1.4e\t %1.4e\t %1.4e\t %1.4e\t %1.4e\t %1.4e\t %1.4e\n',iter, nn, fval, lambda*nn+fval, dx, nr_p, e_p, nr_d, e_d);
    end

end

%Y = zeros(length(nz),T);
Y = x;
%Y(~nz,:) = -Inf; % set firing rates to zero for rows with no data. Used to make sure the returned value is aligned with the input

    function y = inv_hess_mult(H,x)
            y = x./(H + rho);
    end

    function y = obj(x)
        
        foo = S.*log(f( bsxfun(@plus,x,d) ));
        foo(isnan(foo)) = 0; % 0*log(0) = 0 by convention
        y = sum(sum(f( bsxfun(@plus,x,d)  ) - foo)) + sum(sum(Z.*x)) + rho/2*norm(x-X,'fro')^2;
        
    end
end
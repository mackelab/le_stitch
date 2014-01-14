function [params, LL, FILENAME] = ...
    learn_kalman(data, input, s, h, params, max_iter, diagQ, diagR, ARmode, constr_fun)
% LEARN_KALMAN Find the ML parameters of a stochastic Linear Dynamical System using EM.
%
% [A, B, C, D, Q, R, INITX, INITV, LL] = LEARN_KALMAN(DATA, INPUT, A0, B0, C0, D0, Q0, R0, INITX0, INITV0) fits
% the parameters which are defined as follows
%   x(t+1) = A*x(t) + B*u(t) + w(t),  w ~ N(0, Q),  x(0) ~ N(init_x, init_V)
%   y(t)   = C*x(t) + D*u(t) + v(t),  v ~ N(0, R)
% A0 is the initial value, A is the final value, etc.
% DATA(:,t,l) is the observation vector at time t for sequence l. If the sequences are of
% different lengths, you can pass in a cell array, so DATA{l} is an O*T matrix.
% LL is the "learning curve": a vector of the log lik. values at each iteration.
% LL might go positive, since prob. densities can exceed 1, although this probably
% indicates that something has gone wrong e.g., a variance has collapsed to 0.
%
% There are several optional arguments, that should be passed in the following order.
% LEARN_KALMAN(DATA, INPUT, A0, C0, Q0, R0, INITX0, INITV0, MAX_ITER, DIAGQ, DIAGR, ARmode)
% MAX_ITER specifies the maximum number of EM iterations (default 10).
% DIAGQ=1 specifies that the Q matrix should be diagonal. (Default 0).
% DIAGR=1 specifies that the R matrix should also be diagonal. (Default 0).
% ARMODE=1 specifies that C=I, R=0. i.e., a Gauss-Markov process. (Default 0).
% This problem has a global MLE. Hence the initial parameter values are not important.
%
% LEARN_KALMAN(DATA, INPUT, A0, C0, Q0, R0, INITX0, INITV0, MAX_ITER, DIAGQ, DIAGR, F, P1, P2, ...)
% calls [A,C,Q,R,initx,initV] = f(A,C,Q,R,initx,initV,P1,P2,...) after every M step. f can be
% used to enforce any constraints on the params.
%
% For details, see
% - Ghahramani and Hinton, "Parameter Estimation for LDS", U. Toronto tech. report, 1996
% - Digalakis, Rohlicek and Ostendorf, "ML Estimation of a stochastic linear system with the EM
%      algorithm and its application to speech recognition",
%       IEEE Trans. Speech and Audio Proc., 1(4):431--442, 1993.

%    learn_kalman(data, input, A, B, C, D, Q, R, initx, initV, max_iter, diagQ, diagR, ARmode, constr_fun, varargin)
if ~exist('max_iter','var') || isempty(max_iter),
    max_iter = 10;
end
if ~exist('diagQ','var') || isempty(diagQ),
    diagQ = false;
end
if ~exist('diagR','var') || isempty(diagR),
    diagR = false;
end
if ~exist('ARmode','var') || isempty(ARmode),
    ARmode = false;
end
if ~exist('constr_fun','var'),
    constr_fun = [];
end
verbose = 1;
% thresh = 1e-10;
thresh = 2e-4;

if( ~isempty(params.filename) )
    FILENAME = [params.filename '_' datestr(now,30)];
    if( exist('Kalman_Code', 'dir') == 7 )
        mkdir(FILENAME)
    end
else
    FILENAME = 0;
end

if ~iscell(data)
    N = size(data, 3);
    data = num2cell(data, [1 2]); % each elt of the 3rd dim gets its own cell
    input = num2cell(input, [1 2]);
else
    N = length(data);
end

N = length(data);
ss = size(params.A, 1); % statespace dimensionality
is = size(input{1}, 1); % control input dimensionality
os = size(params.C, 1); % response (data) dimensionality
ds = size(params.D, 2)*os; % feedback dimensionality
es = size(params.E, 2); % h dimensionality

suffstats.yt_yt = zeros(os, os);
suffstats.fut_fut = zeros(ss, ss);
suffstats.yt_xt = zeros(os, ss);
suffstats.xt_xt = zeros(ss, ss);
suffstats.xt_xtminus1 = zeros(ss, ss);
suffstats.xt_fut = zeros(ss, ss);
suffstats.xt_fut_1 = zeros(ss, ss);
suffstats.xt_fut_T = zeros(ss, ss);

suffstats.xt_xt_1 = zeros(ss, ss);
suffstats.xt_xt_T = zeros(ss, ss);
suffstats.xtminus1_fut = zeros(ss, ss);
suffstats.fut_fut_1 = zeros(ss, ss);

suffstats.yt_st = zeros(os, ds);
suffstats.yt_ht = zeros(os, es);

suffstats.xt_st = zeros(ss, ds);
suffstats.xt_ht = zeros(ss, es);
suffstats.st_st = zeros(ds, ds);
suffstats.st_ht = zeros(ds, es);
suffstats.ht_ht = zeros(es, es);

suffstats.x1 = zeros(ss, 1);
suffstats.V1 = zeros(ss, ss);
suffstats.T = 0;

for ex = 1:N
    y = data{ex};
    
    suffstats.T = suffstats.T + size(y,2);
    suffstats.yt_yt = suffstats.yt_yt + y*y';
    
    % setting up suffstats for s and h
    suffstats.st_st = s*s';
    suffstats.ht_ht = h*h';
    suffstats.st_ht = s*h';
    suffstats.yt_st = y*s';
    suffstats.yt_ht = y*h';
    
end
suffstats.T_1 = suffstats.T - N;

previous_loglik = -inf;
loglik = 0;
converged = 0;
num_iter = 1;
LL = [];

if(params.disable_dynamics == 1)
    params.A = zeros(size(params.A));%eye(size(params.A));
end


% Convert to inline function as needed.
if ~isempty(constr_fun)
    constr_fun = fcnchk(constr_fun,length(varargin));
end

% while ~converged & (num_iter <= max_iter)
while (num_iter <= max_iter)
    
    %%% E step
    suffstats.yt_xt(:) = 0;
    suffstats.xt_xt(:) = 0;
    suffstats.xt_ut(:) = 0;
    suffstats.xt_xtminus1(:) = 0;
    %   suffstats.xt_futminus1(:) = 0;
    
    suffstats.xt_xt_1(:) = 0;
    suffstats.xt_xt_T(:) = 0;
    suffstats.xt_fut_T(:) = 0;
    
    suffstats.x1(:) = 0;
    suffstats.V1(:) = 0;
    
    loglik = 0;
    
    for ex = 1:N
        y = data{ex};
        u = input{ex};
        [suffstats_ex, loglik_ex, mu, V] = Estep(y, u, s, h, params, ARmode);
        for f = fieldnames(suffstats_ex)',
            suffstats.(f{1}) = suffstats.(f{1}) + suffstats_ex.(f{1});
        end
        loglik = loglik + loglik_ex;
        
        %fprintf(1, 'example %d, ll/T %5.3f\n', ex, loglik_t/T);
    end
    LL = [LL loglik];
    if verbose, fprintf(1, 'iteration %d, loglik = %f\n', num_iter, loglik); end
    %fprintf(1, 'iteration %d, loglik/NT = %f\n', num_iter, loglik/Tsum);
    num_iter = num_iter + 1;
    
    %%% M step
    
    %% Update initial paramters
    
    params.initx = suffstats.x1;
    params.initV = suffstats.V1;
    
    %% Implement an iterative method to compute B and A!!!
    err = 1;
    while(1)
        
        % UPDATE B
        Bnew = funcmaxB(params, mu, u);
        Bold = params.fnlin.B;
        params.fnlin.B = Bnew;
        
        % UPDATE SUFFSTATS
        % once we update B we have to update the sufficient statistics
        suffstats_fut = recomputeFUMoments(params, u, mu);
        suffstats.fut_fut = suffstats_fut.fut_fut;
        suffstats.fut_fut_1 = suffstats_fut.fut_fut_1;
        suffstats.xt_fut = suffstats_fut.xt_fut;
        suffstats.xtminus1_fut = suffstats_fut.xtminus1_fut;
        suffstats.xt_fut_T = suffstats_fut.xt_fut_T;
        
        % Update A 
        Aold = params.A;
        if(params.disable_dynamics ~= 1)
            params.A = (suffstats.xt_xtminus1 - suffstats.xtminus1_fut')*inv(suffstats.xt_xt_T);
        end
        
        %% Q (ugly but general... could be improved via matrix form -- TBD)
        Qnew = (suffstats.xt_xt_1 ...
            - suffstats.xt_xtminus1*params.A' ...
            - suffstats.xt_fut_1...
            - params.A*suffstats.xt_xtminus1' ...
            + params.A * suffstats.xt_xt_T * params.A' ...
            + params.A * suffstats.xtminus1_fut  ...
            -suffstats.xt_fut_1' ...
            + suffstats.xtminus1_fut'*params.A'...
            + suffstats.fut_fut_1)/suffstats.T_1;
        
        params.Q = (Qnew + Qnew')/2;
        
        err = norm(Aold - params.A)
        if( err < 1e-3)
            break
        end
    end
    %     Qnew_lin = (suffstats.xt_xt_1 ...
    %       - suffstats.xt_xtminus1*params.A' ...
    %       - suffstats.xt_utminus1*params.B'...
    %       - params.A*suffstats.xt_xtminus1' ...
    %       + params.A * suffstats.xt_xt_T * params.A' ...
    %       + params.A * suffstats.xt_ut_T*params.B'  ...
    %       -params.B*suffstats.xt_utminus1' ...
    %       + params.B*suffstats.xt_ut_T'*params.A'...
    %       + params.B*suffstats.ut_ut_T*params.B')/suffstats.T_1;
    %   Qnew_lin = (Qnew_lin + Qnew_lin')/2
    %
    
    %% C,D,E
    % Compute parameter updates observation model equation
    XSH = [suffstats.xt_xt suffstats.xt_st suffstats.xt_ht;
        suffstats.xt_st' suffstats.st_st suffstats.st_ht;
        suffstats.xt_ht' suffstats.st_ht' suffstats.ht_ht];
    
    y_cross = [suffstats.yt_xt'; suffstats.yt_st'; suffstats.yt_ht'];
    vvv = (XSH\y_cross)';
    
    C = vvv(:, 1:ss); Dstack = vvv(:, ss + (1:ds)); E = vvv(:, ss+ds+(1:es));
    D = zeros(size(params.D));
    for idx = 1:(ds/os)
        D(:,idx) = diag(Dstack(:, os*(idx-1) + (1:os)));
    end
    params.C = C; params.D = D; params.E = E;
    
    %% R
    e = y - params.C*mu - params.E*h;
    for t = 1:N
        e(:, t) = e(:,t) - sum(params.D.*reshape(s(:,t), os, []),2);
    end
    
    params.R = (params.C*V*params.C' + e*e')/suffstats.T;
    params.R = (params.R+params.R')/2;
    params.R = diag(diag(params.R));
    
    %% Save parameters for iteration, if filename was passed.
    if(FILENAME)
        save(sprintf('%s/iter%d', FILENAME, num_iter), 'params', 'loglik')
    end
    
    converged = em_converged(loglik, previous_loglik, thresh);
    previous_loglik = loglik;
end

%%%%%%%%%

% function [xt_xt,xt_xt_1,xt_xt_T,xt_xtminus1,yt_xt,xt_ut_1,xt_utminus1, x1, V1, loglik] = ...
% function [suffstats, loglik] = Estep(y, u, A, B, C, D, Q, R, initx, initV, ARmode)
function [suffstats, loglik, xsmooth, V, Vsmooth] = Estep(y, u, s, h, params, ARmode)
%
% Compute the (expected) sufficient statistics for a single Kalman filter sequence.
%

[os T] = size(y);
% [ss is] = size(params.B);
is = size(u,1);
ss = params.fnlin.outDim;[os ss] = size(params.C);

if ARmode
    xsmooth = y;
    Vsmooth = zeros(ss, ss, T); % no uncertainty about the hidden states
    VVsmooth = zeros(ss, ss, T);
    loglik = 0;
else
%     [xsmooth, Vsmooth, VVsmooth, loglik] = kalman_smoother( ...
%         bsxfun(@minus, y, params.d), u, s, h, params);
    [xsmooth, Vsmooth, VVsmooth, loglik] = kalman_smoother(y, u, s, h, params);
end

suffstats_fut = recomputeFUMoments(params, u, xsmooth);
suffstats.fut_fut = suffstats_fut.fut_fut;
suffstats.fut_fut_1 = suffstats_fut.fut_fut_1;
% suffstats.xt_fut = suffstats_fut.xt_futminus1;
suffstats.xt_fut = suffstats_fut.xt_fut;
suffstats.xtminus1_fut = suffstats_fut.xtminus1_fut;

suffstats.yt_xt = y*xsmooth';
suffstats.xt_xt = xsmooth*xsmooth' + sum(Vsmooth,3);
suffstats.xt_xtminus1 = xsmooth(:,2:end)*xsmooth(:,1:end-1)' + sum(VVsmooth(:,:,2:end),3);

suffstats.xt_xt_1 = suffstats.xt_xt - xsmooth(:,1)*xsmooth(:,1)';
suffstats.xt_xt_T = suffstats.xt_xt - xsmooth(:,end)*xsmooth(:,end)';

suffstats.xt_st = xsmooth*s';
suffstats.xt_ht = xsmooth*h';

suffstats.x1 = xsmooth(:,1);
suffstats.V1 = Vsmooth(:,:,1);

V = sum(Vsmooth,3);

% OLD:
% suffstats.xt_utminus1 = xsmooth(:,2:end)*u(:,1:end-1)';
% suffstats.xt_ut = xsmooth*u';
% suffstats.xt_ut_T = suffstats.xt_ut - xsmooth(:,end)*u(:,end)';

function  suffstats = recomputeFUMoments(params, u, xsmooth)
% [~, suffstats.fut_fut] = computeStimNlinMoments(params.fnlin, u);
um = computeStimNlin(params.fnlin,u);
suffstats.fut_fut = um*um';
% suffstats.fut_fut_T = um*um' - um(:,end)*um(:,end)';
suffstats.fut_fut_1 = suffstats.fut_fut - um(:,1)*um(:,1)';
suffstats.xt_fut = xsmooth*um';
suffstats.xt_fut_T = xsmooth*um' - xsmooth(:,end)*um(:,end)';
% suffstats.xt_fut = xsmooth(:,2:end)*um(:,1:end-1)';
suffstats.xtminus1_fut = xsmooth(:,1:end-1)*um(:,2:end)';
% INDEX BUG?!?! switching to x_t.*u_t
% INDEX BUG?!?! switching to x_{t-1}.*u_t
% suffstats.xtminus1_fut = xsmooth(:,1:end-1)*um(:,2:end)';






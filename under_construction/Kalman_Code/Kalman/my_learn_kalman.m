function [A, C, Q, R, initx, initV, LL,M,filt] = ...
    my_learn_kalman(fulldata, A, C, Q, R, initx, initV, max_iter, diagQ, diagR, ARmode, constr_fun, varargin)
% LEARN_KALMAN Find the ML parameters of a stochastic Linear Dynamical System using EM.
%
% [A, C, Q, R, INITX, INITV, LL] = LEARN_KALMAN(DATA, A0, C0, Q0, R0, INITX0, INITV0) fits
% the parameters which are defined as follows
%   x(t+1) = A*x(t) + w(t),  w ~ N(0, Q),  x(0) ~ N(init_x, init_V)
%   y(t)   = C*x(t) + v(t),  v ~ N(0, R)
% A0 is the initial value, A is the final value, etc.
% DATA(:,t,l) is the observation vector at time t for sequence l. If the sequences are of
% different lengths, you can pass in a cell array, so DATA{l} is an O*T matrix.
% LL is the "learning curve": a vector of the log lik. values at each iteration.
% LL might go positive, since prob. densities can exceed 1, although this probably
% indicates that something has gone wrong e.g., a variance has collapsed to 0.
%
% There are several optional arguments, that should be passed in the following order.
% LEARN_KALMAN(DATA, A0, C0, Q0, R0, INITX0, INITV0, MAX_ITER, DIAGQ, DIAGR, ARmode)
% MAX_ITER specifies the maximum number of EM iterations (default 10).
% DIAGQ=1 specifies that the Q matrix should be diagonal. (Default 0).
% DIAGR=1 specifies that the R matrix should also be diagonal. (Default 0).
% ARMODE=1 specifies that C=I, R=0. i.e., a Gauss-Markov process. (Default 0).
% This problem has a global MLE. Hence the initial parameter values are not important.
% 
% LEARN_KALMAN(DATA, A0, C0, Q0, R0, INITX0, INITV0, MAX_ITER, DIAGQ, DIAGR, F, P1, P2, ...)
% calls [A,C,Q,R,initx,initV] = f(A,C,Q,R,initx,initV,P1,P2,...) after every M step. f can be
% used to enforce any constraints on the params. 
%
% For details, see
% - Ghahramani and Hinton, "Parameter Estimation for LDS", U. Toronto tech. report, 1996
% - Digalakis, Rohlicek and Ostendorf, "ML Estimation of a stochastic linear system with the EM
%      algorithm and its application to speech recognition",
%       IEEE Trans. Speech and Audio Proc., 1(4):431--442, 1993.


    data = fulldata.data;
    filt = fulldata.filt;
    X = fulldata.X;
    M = fulldata.M;

    %    learn_kalman(data, A, C, Q, R, initx, initV, max_iter, diagQ, diagR, ARmode, constr_fun, varargin)
    if nargin < 8, max_iter = 10; end
    if nargin < 9, diagQ = 0; end
    if nargin < 10, diagR = 0; end
    if nargin < 11, ARmode = 0; end
    if nargin < 12, constr_fun = []; end
    verbose = 1;
    thresh = 1e-4;

    if ~iscell(data)
      N = size(data, 3);
      data = num2cell(data, [1 2]); % each elt of the 3rd dim gets its own cell
    else
      N = length(data);
    end

    N = length(data);
    ss = size(A, 1);
    os = size(C,1);

    alpha = zeros(os, os);
    Tsum = 0;
    for ex = 1:N
      %y = data(:,:,ex);
      y = data{ex};
      %T = length(y);
      T =size(y,2);
      Tsum = Tsum + T;
      alpha_temp = zeros(os, os);
      for t=1:T
        alpha_temp = alpha_temp + y(:,t)*y(:,t)';
      end
      alpha = alpha + alpha_temp;
    end

    d = X*filt;    
    Fs = [d'.^2; d'; ones(size(d))'];


    previous_loglik = -inf;
    loglik = 0;
    converged = 0;
    num_iter = 1;
    LL = [];

    % Convert to inline function as needed.
    if ~isempty(constr_fun)
      constr_fun = fcnchk(constr_fun,length(varargin));
    end


    while ~converged & (num_iter <= max_iter) 

      %%% E step

      delta = zeros(os, ss);
      gamma = zeros(ss, ss);
      gamma1 = zeros(ss, ss);
      gamma2 = zeros(ss, ss);
      beta = zeros(ss, ss);
      P1sum = zeros(ss, ss);
      x1sum = zeros(ss, 1);
      loglik = 0;

      for ex = 1:N
        y = data{ex};
        T = length(y);
        [beta_t, gamma_t, delta_t, gamma1_t, gamma2_t, x1, V1, loglik_t, qt] = ...
        Estep(y, A, C, Q, R, initx, initV, ARmode, Fs,M);


        beta = beta + beta_t;
        gamma = gamma + gamma_t;
        delta = delta + delta_t;
        gamma1 = gamma1 + gamma1_t;
        gamma2 = gamma2 + gamma2_t;
        P1sum = P1sum + V1 + x1*x1';
        x1sum = x1sum + x1;
        %fprintf(1, 'example %d, ll/T %5.3f\n', ex, loglik_t/T);
        loglik = loglik + loglik_t; 
      end
      LL = [LL loglik];
      if verbose, fprintf(1, 'iteration %d, loglik = %f\n', num_iter, loglik); end
      %fprintf(1, 'iteration %d, loglik/NT = %f\n', num_iter, loglik/Tsum);
      num_iter =  num_iter + 1;

      %%% M step
      

      % Tsum =  N*T
      % Tsum1 = N*(T-1);
      Tsum1 = Tsum - N;
      A = beta * inv(gamma1);
      %A = (gamma1' \ beta')';
      Q = (gamma2 - A*beta') / Tsum1;
      if diagQ
        Q = diag(diag(Q));
      end
      if ~ARmode
        C = delta * inv(gamma);
        %C = (gamma' \ delta')';
        R = (alpha - C*delta') / Tsum;
        if diagR
          R = diag(diag(R));
        end
      end
      initx = x1sum / N;
      initV = P1sum/N - initx*initx';

      if ~isempty(constr_fun)
        [A,C,Q,R,initx,initV] = feval(constr_fun, A, C, Q, R, initx, initV, varargin{:});
      end
      
      
      options = optimset('GradObj','off','display','iter');
      
      if( mod(num_iter,2) )
          fprintf('optimizing M\n')
          

          M  = fminunc(@(mm) optimize_M(mm), M,options);
      else
          fprintf('optimizing filt\n') 
          
%           DerivCheck_Elts(@(ff) optimize_filter(ff), 1,filt)
%           keyboard
          filt = fminunc(@(ff) optimize_filter(ff), filt,options);
         
      end

      converged = em_converged(loglik, previous_loglik, thresh);
      previous_loglik = loglik;
    end

    function resid = optimize_M(M)
          resid = qt - (M*Fs + A*[x1 qt(:,1:end-1)]);
          resid = sum(sum(resid.*(Q\resid)));
    end

    function [resid grad] = optimize_filter(filt)
        d = X*filt;    
        Fs = [d'.^2; d'; ones(size(d))'];

        resid_v = qt - (M*Fs + A*[x1 qt(:,1:end-1)]);
        resid_vq = Q\resid_v;
        
        resid = sum(sum(resid_v.*(resid_vq)));
        
%         grad = zeros(1, length(filt));
%         for idx = 1:size(X,1)
%         	grad = grad - 2*(resid_v(:,idx)'*((inv(Q')M')*[2*d(idx)*X(idx,:) ; X(idx,:) ; zeros(1,size(X,2))]));
%         end
%         grad = sum(grad);
    end


    
end

%%%%%%%%%

function [beta, gamma, delta, gamma1, gamma2, x1, V1, loglik,xsmooth] = ...
    Estep(y, A, C, Q, R, initx, initV, ARmode, Fs, M)
%
% Compute the (expected) sufficient statistics for a single Kalman filter sequence.
%

    [os T] = size(y);
    ss = length(A);

    if ARmode
      xsmooth = y;
      Vsmooth = zeros(ss, ss, T); % no uncertainty about the hidden states
      VVsmooth = zeros(ss, ss, T);
      loglik = 0;
    else
      [xsmooth, Vsmooth, VVsmooth, loglik] = kalman_smoother(y, A, C, Q, R, initx, initV, 'u', Fs,'B',M);
    end

    delta = zeros(os, ss);
    gamma = zeros(ss, ss);
    beta = zeros(ss, ss);
    for t=1:T
      delta = delta + y(:,t)*xsmooth(:,t)';
      gamma = gamma + xsmooth(:,t)*xsmooth(:,t)' + Vsmooth(:,:,t);
      if t>1 beta = beta + xsmooth(:,t)*xsmooth(:,t-1)' + VVsmooth(:,:,t); end
    end
    gamma1 = gamma - xsmooth(:,T)*xsmooth(:,T)' - Vsmooth(:,:,T);
    gamma2 = gamma - xsmooth(:,1)*xsmooth(:,1)' - Vsmooth(:,:,1);

    x1 = xsmooth(:,1);
    V1 = Vsmooth(:,:,1);

    qt = xsmooth;

end


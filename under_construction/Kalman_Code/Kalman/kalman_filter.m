function [x, V, VV, loglik, ypred] = kalman_filter(y, u, s, h, d, A, C, D, E, Q, R, init_x, init_V, fnlin)
% Kalman filter.
% [x, V, VV, loglik] = kalman_filter(y, A, C, Q, R, init_x, init_V, ...)
%
% INPUTS:
% y(:,t)   - the observation at time t
% A - the system matrix
% C - the observation matrix 
% Q - the system covariance 
% R - the observation covariance
% init_x - the initial state (column) vector 
% init_V - the initial state covariance 
%
% OPTIONAL INPUTS (string/value pairs [default in brackets])
% 'model' - model(t)=m means use params from model m at time t [ones(1,T) ]
%     In this case, all the above matrices take an additional final dimension,
%     i.e., A(:,:,m), C(:,:,m), Q(:,:,m), R(:,:,m).
%     However, init_x and init_V are independent of model(1).
% 'u'     - u(:,t) the control signal at time t [ [] ]
% 'B'     - B(:,:,m) the input regression matrix for model m
%
% OUTPUTS (where X is the hidden state being estimated)
% x(:,t) = E[X(:,t) | y(:,1:t)]
% V(:,:,t) = Cov[X(:,t) | y(:,1:t)]
% VV(:,:,t) = Cov[X(:,t), X(:,t-1) | y(:,1:t)] t >= 2
% loglik = sum{t=1}^T log P(y(:,t))
%
% If an input signal is specified, we also condition on it:
% e.g., x(:,t) = E[X(:,t) | y(:,1:t), u(:, 1:t)]
% If a model sequence is specified, we also condition on it:
% e.g., x(:,t) = E[X(:,t) | y(:,1:t), u(:, 1:t), m(1:t)]

[os T] = size(y);
ss = size(A,1); % size of state space

% set default params
model = ones(1,T);
x = zeros(ss, T);
ypred = zeros(os, T);
V = zeros(ss, ss, T);
VV = zeros(ss, ss, T);

loglik = 0;
for t=1:T
  m = model(t);
  if t==1
    prevx = init_x;
    prevV = init_V;
    initial = 1;
  else
    prevx = x(:,t-1);
    prevV = V(:,:,t-1);
    initial = 0;
  end
      
  [x(:,t), V(:,:,t), LL, VV(:,:,t), ypred(:,t)] = ...
      kalman_update(y(:,t), u(:,t), s(:,t), h(:,t), d, A(:,:,m), C(:,:,m), D(:,:,m), E(:,:,m), Q(:,:,m), R(:,:,m), prevx, prevV, initial, fnlin);

  loglik = loglik + LL;
  
end






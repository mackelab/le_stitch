function [x,y] = sample_lds(A, C, Q, R, init_state, T, models, B, u)
%function [x,y] = sample_lds(A, C, Q, R, init_state, T, models, B, u)
% SAMPLE_LDS Simulate a run of a (switching) stochastic linear dynamical system.
% [x,y] = switching_lds_draw(A, C, Q, R, init_state, models, B, u)
% 
%   x(t+1) = A*x(t) + B*u(t) + w(t),  w ~ N(0, Q),  x(0) = init_state
%   y(t) =   C*x(t) + v(t),  v ~ N(0, R)
%
% Input:
% A(:,:,i) - the transition matrix for the i'th model
% C(:,:,i) - the observation matrix for the i'th model
% Q(:,:,i) - the transition covariance for the i'th model
% R(:,:,i) - the observation covariance for the i'th model
% init_state(:,i) - the initial mean for the i'th model
% T - the num. time steps to run for
%
% Optional inputs:
% models(t) - which model to use at time t. Default = ones(1,T)
% B(:,:,i) - the input matrix for the i'th model. Default = 0.
% u(:,t)   - the input vector at time t. Default = zeros(1,T)
%
% Output:
% x(:,t)    - the hidden state vector at time t.
% y(:,t)    - the observation vector at time t.


if ~iscell(A)
  A = num2cell(A, [1 2]);
  C = num2cell(C, [1 2]);
  Q = num2cell(Q, [1 2]);
  R = num2cell(R, [1 2]);
end

M = length(A);
%T = length(models);

if nargin < 7 || isempty(models)
  models = ones(1,T);
end
if nargin < 8,
  B = num2cell(repmat(0, [1 1 M]));
  u = zeros(1,T);
else 
    if ~iscell(B)
        B = num2cell(B, [1 2]);
    end
end

[os ss] = size(C{1});
state_noise_samples = cell(1,M);
obs_noise_samples = cell(1,M);
for i=1:M
  state_noise_samples{i} = sample_gaussian(zeros(length(Q{i}),1), Q{i}, T)';
  obs_noise_samples{i} = sample_gaussian(zeros(length(R{i}),1), R{i}, T)';
end

x = zeros(ss, T);
y = zeros(os, T);

m = models(1);
x(:,1) = init_state(:,m);
y(:,1) = C{m}*x(:,1) + obs_noise_samples{m}(:,1);

for t=2:T
  m = models(t);
  x(:,t) = A{m}*x(:,t-1) + B{m}*u(:,t-1) + state_noise_samples{m}(:,t);
  y(:,t) = C{m}*x(:,t)  + obs_noise_samples{m}(:,t);
end



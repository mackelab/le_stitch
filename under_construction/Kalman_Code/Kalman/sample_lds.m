function [x,y] = sample_lds(params)
% function [x,y] = sample_lds(params)
%function [x,y] = sample_lds(A, C, Q, R, initx, T, models, B, u)
% SAMPLE_LDS Simulate a run of a (switching) stochastic linear dynamical system.
% [x,y] = switching_lds_draw(A, C, Q, R, initx, models, B, u)
% 
%   x(t+1) = A*x(t) + B*u(t) + w(t),  w ~ N(0, Q),  x(0) = initx
%   y(t) =   C*x(t) + v(t),  v ~ N(0, R)
%
% Input:
% A(:,:,i) - the transition matrix for the i'th model
% C(:,:,i) - the observation matrix for the i'th model
% Q(:,:,i) - the transition covariance for the i'th model
% R(:,:,i) - the observation covariance for the i'th model
% initx(:,i) - the initial mean for the i'th model
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

POSSIBLE_INPUT_PARAMETERS = {'A', 'C', 'D', 'E', 'Q', 'R', 'initx', 'models', 'u', 's', 'h', 'initx'};

for ndx = 1:length(POSSIBLE_INPUT_PARAMETERS)
   if( ~isfield(params, POSSIBLE_INPUT_PARAMETERS{ndx}) )
       error(sprintf('\tPlease specify parameter %s!\n', POSSIBLE_INPUT_PARAMETERS{ndx}))
   end
end

% Length of simulation is defined implicitly through the lengths of inputs.
% Assure that time dimension is consistently represented across inputs. 
T = size(params.u,2);
[os ss] = size(params.C);

assert( size(params.s, 2) == T)
assert( size(params.h, 2) == T)

if( ~isfield(params, 'd'))
    params.d = zeros(os,1);
end

if ~iscell(params.A)
    params.A = num2cell(params.A, [1 2]);
    params.C = num2cell(params.C, [1 2]);
    params.D = num2cell(params.D, [1 2]);
    params.E = num2cell(params.E, [1 2]);
    params.Q = num2cell(params.Q, [1 2]);
    params.R = num2cell(params.R, [1 2]);
    params.d = num2cell(params.d, [1 2]);
end


M = length(params.A);

% ============================
% Commented out. No longer permit default options set within function!
% ============================
%T = length(models);
% if nargin < 7 || isempty(models)
%   models = ones(1,T);
% end
% if nargin < 8,
%   B = num2cell(repmat(0, [1 1 M]));
%   u = zeros(1,T);
% else 
%     if ~iscell(B)
%         B = num2cell(B, [1 2]);
%     end
% end

state_noise_samples = cell(1,M);
obs_noise_samples = cell(1,M);
for i=1:M
  state_noise_samples{i} = mvnrnd(zeros(1,length(params.Q{i})), params.Q{i}, T)'; %sample_gaussian(zeros(length(params.Q{i}),1), params.Q{i}, T)';
  obs_noise_samples{i} = mvnrnd(zeros(1,length(params.R{i})), params.R{i}, T)';%sample_gaussian(zeros(length(params.R{i}),1), params.R{i}, T)';
end

x = zeros(ss, T);
y = zeros(os, T);

m = params.models(1);
x(:,1) = params.initx(:,m)  + params.fnlin.f(params.fnlin, params.u(:,1)) + state_noise_samples{m}(:,1);
y(:,1) = params.C{m}*x(:,1) + obs_noise_samples{m}(:,1);

for t=2:T
  m = params.models(t);
  x(:,t) = params.A{m}*x(:,t-1) + params.fnlin.f(params.fnlin, params.u(:,t)) + state_noise_samples{m}(:,t);
  if(isempty(params.D) && isempty(params.E))
      y(:,t) = params.C{m}*x(:,t) + obs_noise_samples{m}(:,t);
  else
      y(:,t) = params.C{m}*x(:,t)   + sum(params.D{m}.*reshape(params.s(:,t), os, []),2) + params.E{m}*params.h(:,t) + obs_noise_samples{m}(:,t);
  end
end



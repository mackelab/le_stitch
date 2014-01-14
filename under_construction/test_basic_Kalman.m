startup

%% 1) simulate timeseries with latent LDS dynamics 

nStateDim = 2; V = orth(rand(nStateDim));
nStimDim  = 2; 
nObsDim   = 3;
nSDim = 0;
nHDim = 0;
nT  = 5e3; 
u = randn(nT, nStimDim);          % stimuli
% u = repmat([1 1 0 0 -1 -1 0 0]', 600, 1); nT = length(u);
s = zeros(nT, nSDim*nObsDim);          % "history"
h = zeros(nT, nHDim);          % more "history"
A = V*diag(rand(nStateDim,1))*V'; % transition matrix
% A = zeros(nStateDim); 
C = randn(nObsDim,nStateDim);      % observation matrix
% D = randn(nObsDim,nSDim);      % single-neuron "history" filter
% E = randn(nObsDim,nHDim);      % "shared" history filter
d = randn(nObsDim,1);

% zero these out for debugging purposes
D = randn(nObsDim,nSDim);      % single-neuron "history" filter
E = randn(nObsDim,nHDim);      % "shared" history filter

Q = diag( randn(nStateDim, 1).^2 );
R = diag( randn(nObsDim, 1).^2 );

x0 = rand(nStateDim,1);

%% Test linear function

params.B = randn(nStateDim, nStimDim);
params.outDim = nStateDim;
params.inDim = nStimDim;
flinstr = makeStimNlin('linear', params);

%% Test quadratic functino 


% [B nbars nkt] = make_gabor_basis();
B = rand(nStimDim, nStateDim);
params.outDim = nStateDim;
params.inDim = nStimDim;
% params.inDim = nbars*nkt;

params.w = B';
params.a = rand(params.outDim,1);
params.b = rand(params.outDim,1);
params.c = rand(params.outDim,1);

fquadstr = makeStimNlin('quadratic', params);

%% 2) Get inference working. 
%       1. This should be easy.

% [x, y] = sample_lds_old(A, C, Q, R, x0, nT,[], B, u');
% 
% figure(1); clf; plot(y')
% title('simulated observations')

%% 2b) Getting new sample_lds to work

params_true.A = A;
params_true.C = C;
params_true.D = D;
params_true.E = E;
params_true.Q = Q; 
params_true.R = R;
params_true.d = d;
params_true.initx = x0;
params_true.models = ones(1,nT);
params_true.fnlin = flinstr;
% additional stuffs

params_true.u = u';
params_true.s = s';
params_true.h = h';

[x_true, y_true] = sample_lds(params_true);

figure(1); clf; plot(y_true')
title('simulated observations')


%% 3) Get EM framework. Need: 

params = [];
params.A = randn(size(A));
params.C = randn(size(C));%+.001*rand(size(C));
params.D = D;%randn(size(D));
params.E = E;%randn(size(E));
params.Q = Q;%randn(size(Q)); Q = Q*Q';
params.R = R; %randn(size(R)); R = R*R';
params.d = d; %randn(size(d));
params.initx = x0;
params.initV = eye(nStateDim);
params.fnlin = flinstr;
params.models = ones(1,nT);
params.filename = '';
params.disable_dynamics = 0;

% params.A = diag(ones(nStateDim,1)); % transition matrix
% params.C = ones(nObsDim,nStateDim);      % observation matrix
% 
% % zero these out for debugging purposes
% params.D = zeros(nObsDim,nSDim);      % single-neuron "history" filter
% params.E = zeros(nObsDim,nHDim);      % "shared" history filter
% 
% params.Q = diag(ones(nStateDim, 1));
% params.R = diag(ones(nObsDim, 1));
% 
% params.initx = zeros(nStateDim,1);
% params.initV = eye(nStateDim);
% params.d = zeros(nObsDim,1);
%
% params.fnlin = params_true.fnlin;
% params.fnlin.B = randn(size(params.fnlin.B));
% params.models = ones(1,nT);
% params.disable_dynamics = 0;
% params.filename = '';

[params_em, LL] = learn_kalman(y_true, u', s', h', params, 15);%, x_true);
params_em.models = ones(1,nT);

%% 4) compare sampled

params_tt = params_em;
params_tt.u = u';
params_tt.s = s';
params_tt.h = h';
[x_fit, y_fit] = sample_lds(params_tt);

figure(2); clf; plot(y_fit')
title('simulated observations')


addpath /home/pillowlab/v1data_Rust_Neuron2005
addpath Kalman_Code/
addpath Kalman_Code/Kalman/
addpath Kalman_Code/KPMstats/
addpath Kalman_Code/KPMtools/
addpath ~/pillowlab/lib/minFunc_2012/
addpath ~/pillowlab/lib/minFunc_2012/minFunc/
addpath ~/pillowlab/lib/minFunc_2012/minFunc/compiled/
addpath ~/pillowlab/ncclabcode/
addpath NlinStimDep/

%% 1) simulate timeseries with latent LDS dynamics 
%       1. Jakob has a simple file that does this, so just use it: 
%           PPLDS_packaged/core_lds/SampleFromPriorLDS.m

DISABLE_STIMULUS = 0;

nStateDim = 1; V = orth(rand(nStateDim));
nStimDim  = 1; 
nObsDim   = 30;
nSDim = 0;
nHDim = 0;
nT  = 5e3; 
X = randn(nT, sqrt(nStimDim));
u = makeStimRows(X, sqrt(nStimDim));          % stimuli
s = zeros(nT, nSDim*nObsDim);          % "history"
h = zeros(nT, nHDim);          % more "history"

% if( nStateDim == 2)
%     A = V*[.5 0; 0 .3]*V';
% elseif(nStateDim == 3)
%     A = V*[.6 0 0 ; 0 .3 0 ; 0 0 .01]*V';
% elseif(nStateDim == 4)
%     A = [.9 -.5 0 .1; .1 .8 -.1 0; 0 0 .7 -.2; .1 0 .02 .88];
% end
A = randn(nStateDim, nStateDim); A = A*A';
% A = V*diag(rand(nStateDim,1))*V'; % transition matrix
% B = real(dftmtx(nStimDim)); 
% B = randn(nStimDim);
% B = B(1:nStateDim,:); % smooth control input guys
C = randn(nObsDim,nStateDim); %.1*randn(nObsDim,nStateDim);      % observation matrix
% D = randn(nObsDim,nSDim);      % single-neuron "history" filter
% E = randn(nObsDim,nHDim);      % "shared" history filter

% zero these out for debugging purposes
D = zeros(nObsDim,nSDim);      % single-neuron "history" filter
E = zeros(nObsDim,nHDim);      % "shared" history filter

Q = randn(nStateDim,nStateDim); Q = Q*Q';%diag([.1; .1; .5; .5]);
R = .0001*diag(ones(nObsDim, 1));
d = zeros(nObsDim,1);

x0 = zeros(nStateDim,1);

%% Test quadratic functino 


nlparms.outDim = nStateDim;
nlparms.inDim = nStimDim;


if(DISABLE_STIMULUS)
    B = zeros(nStateDim, nStimDim);
else
    B = randn(nStateDim, nStimDim);
end

nlparms.w = B;
nlparms.a = randn(1,nStateDim);!
nlparms.b = randn(1,nStateDim);
nlparms.c = randn(1,nStateDim);

if(DISABLE_STIMULUS)
    nlparms.w(:)= 0;
    nlparms.a(:) = 0;
    nlparms.b(:) =  0;
    nlparms.c(:) = 0;
end

fquadstr = makeStimNlin('quadratic', nlparms);

%% Test linear function
clear nlparams;
nlparams.B = B;
nlparams.outDim = size(nlparams.B,1);
nlparams.inDim = size(nlparams.B,2);

flinstr = makeStimNlin('linear', nlparams);

%%
params = [];
V = orth(rand(nStateDim));
params.A = A;
params.C = C;%+.001*rand(size(C));
params.D = D;
params.E = E;
params.Q = Q;
params.R = R;
params.d = d;
params.initx = x0;
params.initV = eye(nStateDim);
params.fnlin = fquadstr;
params.models = ones(1,nT);
params.filename = '';

params_tt = params;
if(DISABLE_STIMULUS)
    params_tt.u = zeros(size(u'));
else
    params_tt.u = u';
end

params_tt.s = s';
params_tt.h = h';
[real_x, real_y] = sample_lds(params_tt);
real_h=u;
if(DISABLE_STIMULUS)
    save('popsim_realparams_blind', 'real_h', 'real_y', 'params_tt', 'real_x')
else
    save('popsim_realparams_linear', 'real_h', 'real_y', 'params_tt', 'real_x')
end

%% 

fit_params = [];
fit_params.A = rand(size(A));
% fit_params.B = B;%+.001*rand(size(B));
fit_params.C = rand(size(C));%+.001*rand(size(C));
fit_params.D = rand(size(D));
fit_params.E = rand(size(E));
fit_params.Q = rand(size(Q)); fit_params.Q = diag(diag(fit_params.Q*fit_params.Q'));
fit_params.R = rand(size(R)); fit_params.R = diag(diag(fit_params.R*fit_params.R'));
fit_params.d = d;
fit_params.initx = zeros(nStateDim,1);
fit_params.initV = eye(nStateDim);
fit_params.fnlin = fquadstr;
if(DISABLE_STIMULUS)    
    fit_params.fnlin.B = 0*rand(size(fit_params.fnlin.B));
else
    fit_params.fnlin.B = rand(size(fit_params.fnlin.B));
end
fit_params.models = ones(1,nT);
if(DISABLE_STIMULUS)
    fit_params.filename = 'DEBUG_SIMULATION_NO_STIMULUS';
else
    fit_params.filename = 'DEBUG_SIMULATION';
end
fit_params.disable_dynamics = 0;

if(DISABLE_STIMULUS)
    [params_em, LL] = learn_kalman(real_y, zeros(size(u')), s', h', fit_params, 50);
else
    [params_em, LL] = learn_kalman(real_y, u', s', h', fit_params, 50);
end
params_em.models = ones(1,nT);

%% Make lots of repeats (real)

Nrpt = 50;
NrptSize = 1000;

Y = zeros([nObsDim NrptSize Nrpt]);
X = zeros([nStateDim NrptSize Nrpt]);

% params_tt = fit.params
% params_tt.u = orig.params_tt.u
% params_tt.s = orig.params_tt.s
% params_tt.h = orig.params_tt.h
% params_tt.models = orig.params_tt.models

params_rep = params_tt;
params_rep.u = params_tt.u(:,1:NrptSize);
params_rep.s = params_tt.s(:,1:NrptSize);
params_rep.h = params_tt.h(:,1:NrptSize);
params_rep.models = params_tt.models(1:NrptSize);
params_rep.R = diag(diag(params_tt.R));

for  idx = 1:Nrpt
    idx
    [X(:,:,idx), Y(:,:,idx)] = sample_lds(params_rep);
end
rep_h = params_rep.u;

if(DISABLE_STIMULUS)
    save('popsim_real_repeatstimulus_blind', 'Y', 'X', 'rep_h', 'params_tt')
else
    save('popsim_real_repeatstimulus_linear', 'Y', 'X', 'rep_h', 'params_tt')
end

%% Make lots of repeats (fit)

Nrpt = 50;
NrptSize = 1000;

Y = zeros([nObsDim NrptSize Nrpt]);
X = zeros([nStateDim NrptSize Nrpt]);

% params_tt = fit.params
% params_tt.u = orig.params_tt.u
% params_tt.s = orig.params_tt.s
% params_tt.h = orig.params_tt.h
% params_tt.models = orig.params_tt.models

params_rep = params_em;
params_rep.u = params_tt.u(:,1:NrptSize);
params_rep.s = params_tt.s(:,1:NrptSize);
params_rep.h = params_tt.h(:,1:NrptSize);
params_rep.models = params_tt.models(1:NrptSize);
params_rep.R = diag(diag(params_tt.R));
params_rep.initx = randn(size(params_rep.initx));

for  idx = 1:Nrpt
    idx
    [X(:,:,idx), Y(:,:,idx)] = sample_lds(params_rep);
end
rep_h = params_rep.u;

if(DISABLE_STIMULUS)
    save('popsim_fit_repeatstimulus_blind', 'Y', 'X', 'rep_h', 'params_em')
else
    save('popsim_fit_repeatstimulus_linear', 'Y', 'X', 'rep_h', 'params_em')
end

% %%
% 
% fit=load('DEBUG_SIMULATION_NO_STIMULUS_20131118T223545/iter72.mat');
% orig = load('popsim_realparams_blind.mat'); orig.params = orig.params_tt;
% 
% fitz = fit.params.fnlin.get_params(fit.params.fnlin)
% origz = orig.params.fnlin.get_params(orig.params.fnlin)
% 
% b = (fitz.w' \ origz.w')
% recon = b'*fitz.w;
% figure(12301);clf;
% for idx= 1:4
%     subplot(4,2,2*idx-1)
%     imagesc(reshape(origz.w(idx,:), 4,4));colormap gray
%     subplot(4,2,2*idx)
%     imagesc( reshape(recon(idx,:), 4,4));colormap gray
% end


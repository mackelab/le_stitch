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

nkt = 8;
nbars = 8;

nStateDim = 4;
nStimDim  = 64; 
nObsDim   = 30;
nSDim = 0;
nHDim = 0;
nT  = 5e3; 

%%
X = randn(nT, nkt);
u = makeStimRows(X, nbars);          % stimuli
s = zeros(nT, nSDim*nObsDim);          % "history"
h = zeros(nT, nHDim);          % more "history"

if(nStateDim == 4)
    A = [.9 -.5 0 .1; .1 .8 -.1 0; 0 0 .7 -.2; .1 0 .02 .88];
    Q = diag([.1; .1; .5; .5]);
else
    A = randn(nStateDim,nStateDim);
    Q = randn(nStateDim); Q = Q*Q';
end
C = bsxfun(@times, randn(nObsDim,nStateDim), [.3 .3 .3 .3]); %.1*randn(nObsDim,nStateDim);      % observation matrix
% zero these out for debugging purposes
D = zeros(nObsDim,nSDim);      % single-neuron "history" filter
E = zeros(nObsDim,nHDim);      % "shared" history filter

R = 3*diag(ones(nObsDim, 1));
d = zeros(nObsDim,1);

x0 = randn(nStateDim,1);

%% Test quadratic functino 


B = make_gabor_basis(nkt,nbars);
nlparms.outDim = nStateDim;
nlparms.inDim = nbars*nkt;

% For this simulation only we're going to take 2 filters, and then set one
% of them to be 0. Later on we'll want to be able to handle state variables
% with no stimulus drive, but for now (and for this simulation), this hack
% should be good enough. 

B = B(:,[2 1 3 3]); B(:,4) = 0;

nlparms.w = B';
nlparms.a = [ 1; -1;  1;  0];
nlparms.b = [-1;  1; -1;  0];
nlparms.c = [ 0;  0;  0;  0];

fquadstr = makeStimNlin('quadratic', nlparms);

%%
params = [];
params.A = A;
params.C = C;%+.001*rand(size(C));
params.D = D;
params.E = E;
params.Q = Q;
params.R = R;
params.d = d;
params.initx = zeros(nStateDim,1);
params.initV = eye(nStateDim);
params.fnlin = fquadstr;
params.models = ones(1,nT);
params.filename = '';

%% 4) compare sampled

% params_tt = params;
% params_tt.u = u';
% params_tt.s = s';
% params_tt.h = h';
[real_x, real_y] = sample_lds(params_tt);
real_h=u;
save('popsim_realparams', 'real_h', 'real_y', 'params_tt', 'real_x')

% figure(2); clf; plot(y')
% title('simulated observations')

%% Compute STA's

% nkt = 5;
% X = makeStimRows(u, nkt);
Nrow = nObsDim/5; Ncol = 5;
for idx = 1:nObsDim
%     [STA,STC] = simpleSTC(u, real_y(idx,:)', 8);

% % Compute spike-triggered mean and covariance
% spvec = real_y(idx,:)';
% nsp = sum(spvec);
% SS = params_tt.u';
% STA = (spvec'*SS)'/nsp;
% STC = SS'*(bsxfun(@times, SS, spvec))/(nsp-1) - STA*STA'*nsp/(nsp-1);

STA = reshape(params_tt.u * real_y(idx,:)', nkt,nbars);

    figure(1);
    subplot(Nrow, Ncol, idx)
%     imagesc(reshape(y(idx,:)*u, 8, []))
    imagesc(STA); colormap gray
%     pause(.1)
end

%% Write figure.
set(gca, 'TickDirMode', 'auto', 'TickDir', 'out', 'box', 'off')

impath = 'figs';
set(gcf, 'PaperPosition', [0 0 Ncol*2.3 Nrow*2.3])
print(gcf, '-depsc',sprintf('%s/SimFiga.eps', impath))
unix(sprintf('./makePdfFiles.sh %s', impath));

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
fit_params.fnlin.B = rand(size(fit_params.fnlin.B));
fit_params.models = ones(1,nT);
fit_params.filename = 'even_better_population_simulation';
fit_params.disable_dynamics = 0;

save('popsim_initparams', 'fit_params')

[params_em, LL] = learn_kalman(real_y, u', s', h', fit_params, 100);
params_em.models = ones(1,nT);

%% OR load from a file

% file_name = 'better_population_simulation_20131118T010206/iter44.mat';
% params = load(file_name);
% params_em = params.params;
%% 
nTsim = 5e3;
% X = randn(nTsim, 8);
params_em_tt = params_em;
params_em_tt.u = u;
params_em_tt.s = zeros(nTsim, nSDim*nObsDim)';
params_em_tt.h = zeros(nTsim, nHDim)';
params_em_tt.models = ones(1,nTsim);
[fit_x, fit_y] = sample_lds(params_em_tt);
fit_h = params_em_tt.u;
save('popsim_fitparams', 'fit_h', 'fit_y', 'params_em_tt')

%%
nTsim = 5e3;
params_nostim_tt = params_tt;
params_nostim_tt.u = zeros(nTsim, nStimDim)';
params_nostim_tt.s = zeros(nTsim, nSDim*nObsDim)';
params_nostim_tt.h = zeros(nTsim, nHDim)';
% params_nostim_tt.C(:) = 0;
params_nostim_tt.models = ones(1,nTsim);
[nostim_x, nostim_y] = sample_lds(params_nostim_tt);

save('popsim_realparams_nostim', 'nostim_y', 'nostim_x', 'params_nostim_tt')
% fprintf('saved')

%% Make lots of repeats

Nrpt = 50;
NrptSize = 1000;

Y = zeros([nObsDim NrptSize Nrpt]);
X = zeros([nStateDim NrptSize Nrpt]);

% params_tt = orig.params;
% params_tt.u = orig.params_tt.u;
% params_tt.s = orig.params_tt.s;
% params_tt.h = orig.params_tt.h;
% params_tt.models = orig.params_tt.models;

params_rep = params_em;
params_rep.u = params_tt.u(:,1:NrptSize);
params_rep.s = params_tt.s(:,1:NrptSize);
params_rep.h = params_tt.h(:,1:NrptSize);
params_rep.models = params_tt.models(1:NrptSize);

for  idx = 1:Nrpt
    idx
    [X(:,:,idx), Y(:,:,idx)] = sample_lds(params_rep);
end
rep_h = params_rep.u;

save('popsim_fit_repeatstimulus', 'Y', 'X', 'rep_h')

%%
% 
fit = load('../even_better_population_simulation_20131210T013920/iter101.mat')
orig = load('../popsim_realparams.mat'); orig.params = orig.params_tt;
params_em = fit.params;
params_tt = orig.params_tt;
u = params_tt.u;
s = params_tt.s;
h = params_tt.h;

% orig = []; orig.params = params_tt;
% fit = [];
% fit.params = params_em;

fitz = fit.params.fnlin.get_params(fit.params.fnlin)
origz = orig.params.fnlin.get_params(orig.params.fnlin)

b = (fitz.w' \ origz.w')
recon = b'*fitz.w;
figure(12301);clf;
for idx= 1:4
    subplot(4,2,2*idx-1)
    imagesc(reshape(origz.w(idx,:), 8,8));colormap gray
    subplot(4,2,2*idx)
    imagesc( reshape(recon(idx,:), 8,8));colormap gray
%     caxis([-.4 .4])
end

set(gca, 'TickDirMode', 'auto', 'TickDir', 'out', 'box', 'off')

impath = 'figs';
set(gcf, 'PaperPosition', [0 0 2*2.3 4*2.3])
print(gcf, '-depsc',sprintf('%s/SimFigab.eps', impath))
unix(sprintf('./makePdfFiles.sh %s', impath));

%%






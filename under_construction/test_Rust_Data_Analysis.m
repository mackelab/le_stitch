addpath /Users/evan/Desktop/RUSTDATA/v1data_Rust_Neuron2005/
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

cnum = 21;

switch cnum
    case 9
        load 543l036.p09_stc.mat
    case 18 
        load 544l027.p18_stc.mat
    case 21
        load 544l029.p21_stc.mat
    case 27
        load 541l009.p27_stc.mat
    otherwise
        error('Not a cell I am planning for!')
end

% NOTE: possible cells: 21, 27, ...

nT  = 20e3;

stim = stim(1:nT,:);
spikes_per_frm = spikes_per_frm(1:nT);

ii=6:min(21, size(stim,2));
Stim = stim(:,ii);
sps = spikes_per_frm';
sps = [sps(3:end);0;0];
nsp = sum(sps);

global RefreshRate;
RefreshRate = 100;  % (in Hz)

nstim = length(stim);  % number of stimuli
nkt=12; %dimension of filter
nbars = size(Stim,2);  % stimulus width

ttk = -nkt+1:0;
xk = 1:nbars;

% Compute STA & STA
[sta,stc,rawmu,rawcov] = simpleSTC(Stim,sps,nkt); 
sta = sta./norm(sta(:));  % normalize sta

% Do STC analysis
[u,s,v] = svd(stc);
evals = diag(s);

ex_evec1 = reshape(u(:,1),nkt,nbars); 
ex_evec2 = reshape(u(:,2),nkt,nbars); 
ex_evec3 = reshape(u(:,3),nkt,nbars);
ex_evec4 = reshape(u(:,4),nkt,nbars); 
ex_evec5 = reshape(u(:,5),nkt,nbars); 
ex_evec6 = reshape(u(:,6),nkt,nbars); 
ex_evec7 = reshape(u(:,7),nkt,nbars); 
ex_evec8 = reshape(u(:,8),nkt,nbars);

in_evec1 = reshape(u(:,length(evals)),nkt,nbars);
in_evec2 = reshape(u(:,length(evals)-1),nkt,nbars); 
in_evec3 = reshape(u(:,length(evals)-2),nkt,nbars);
in_evec4 = reshape(u(:,length(evals)-3),nkt,nbars); 
in_evec5 = reshape(u(:,length(evals)-4),nkt,nbars);
in_evec6 = reshape(u(:,length(evals)-5),nkt,nbars);
in_evec7 = reshape(u(:,length(evals)-6),nkt,nbars);
in_evec8 = reshape(u(:,length(evals)-7),nkt,nbars); 

% ndims = 8;  % (Only need 2, but compute 10 for demonstration purposes)
% [vecs, vals, DD] = compiSTAC(sta, stc, rawmu, rawcov, ndims);
% % istac1 = reshape(vecs(:,1),nkt,nbars); 
% istac2 = reshape(vecs(:,2),nkt,nbars); 
% istac3 = reshape(vecs(:,3),nkt,nbars); 
% istac4 = reshape(vecs(:,4),nkt,nbars); 
% istac5 = reshape(vecs(:,5),nkt,nbars);
% istac6 = reshape(vecs(:,6),nkt,nbars); 
% istac7 = reshape(vecs(:,7),nkt,nbars); 
% istac8 = reshape(vecs(:,8),nkt,nbars);

%figure;
%plot(evals, 'o');
%title('eigenvalues of STC');

figure; %STC analysis - excitatory
subplot(631); imagesc(ex_evec1); colormap gray; title('EXC #1'); axis square;
subplot(634); imagesc(ex_evec2); colormap gray; title('EXC #2'); axis square;
subplot(6,3,7); imagesc(ex_evec3); colormap gray; title('EXC #3'); axis square;
subplot(6,3,10); imagesc(ex_evec4); colormap gray; title('EXC #4'); axis square;
subplot(6,3,13); imagesc(ex_evec5); colormap gray; title('EXC #5'); axis square;
subplot(6,3,16); imagesc(ex_evec6); colormap gray; title('EXC #6'); axis square;

%figure; %STC analysis - inhibitory
subplot(632); imagesc(in_evec1); colormap gray; title('INH #1'); axis square;
subplot(635); imagesc(in_evec2); colormap gray; title('INH #2'); axis square;
subplot(6,3,8); imagesc(in_evec3); colormap gray; title('INH #3'); axis square;
subplot(6,3,11); imagesc(in_evec4); colormap gray; title('INH #4'); axis square;
subplot(6,3,14); imagesc(in_evec5); colormap gray; title('INH #5'); axis square;
subplot(6,3,17); imagesc(in_evec6); colormap gray; title('INH #6'); axis square;

subplot(6,3,[9 12]); plot(evals,'o'); title('eigenvalues of STC');

eigvecs = u;
%% Set up Dynamics estimation parameters

% [B nbars nkt] = make_gabor_basis();
nlparm.outDim = 4;
nlparm.inDim = nbars*nkt;

if(nlparm.outDim == 4)
    nlparm.w = eigvecs(:, [1:2 (end-1):end])';
elseif(nlparm.outDim == 6)
    nlparm.w = eigvecs(:, [1:3 (end-2):end])';
end

nlparm.a = .1*rand(nlparm.outDim,1);
nlparm.b = .1*rand(nlparm.outDim,1);
nlparm.c = .1*rand(nlparm.outDim,1);

fquadstr = makeStimNlin('quadratic', nlparm);

%% 

nStateDim = nlparm.outDim; 
nStimDim  = nlparm.inDim; 
nObsDim   = 1;
nSDim = 0;
nHDim = 0;
y = sps(1:nT)';
u = makeStimRows(Stim(1:nT,:), nkt);          % stimuli
s = zeros(nT, nSDim*nObsDim);          % "history"
h = zeros(nT, nHDim);          % more "history"
params.A = diag(ones(nStateDim,1)); % transition matrix
params.C = ones(nObsDim,nStateDim);      % observation matrix

% zero these out for debugging purposes
params.D = zeros(nObsDim,nSDim);      % single-neuron "history" filter
params.E = zeros(nObsDim,nHDim);      % "shared" history filter

params.Q = diag(ones(nStateDim, 1));
params.R = diag(ones(nObsDim, 1));

params.initx = zeros(nStateDim,1);
params.initV = eye(nStateDim);
params.d = zeros(nObsDim,1);

params.fnlin = fquadstr;
params.models = ones(1,nT);
params.disable_dynamics = 0;

%%
params.filename = sprintf('Rust_cell%d_%d_%dfilt_BetterIndex', cnum, nT, nlparm.outDim);
[params_em, LL] = learn_kalman(y, u', s', h', params, 5);
params_em.models = ones(1,nT);

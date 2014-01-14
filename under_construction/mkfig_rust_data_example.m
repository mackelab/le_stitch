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

%%
load 544l029.p21_stc.mat
%load 541l009.p27_stc.mat

% NOTE: possible cells: 21, 27, ...

nT  = 60000;

% stim = stim(1:nT,:);
% spikes_per_frm = spikes_per_frm(1:nT);

%Stim = stim;
%sps = spikes_per_frm';
%nsp = sum(sps);

ii=6:min(21, size(stim,2));
Stim = stim(1:nT,ii);
sps = spikes_per_frm(1:nT)';
sps = [sps(3:end);0;0];
nsp = sum(sps);

global RefreshRate;
RefreshRate = 100;  % (in Hz)

nstim = length(stim);  % number of stimuli
nkt=12; %dimension of filter
nbars = size(Stim,2);  % stimulus width

ttk = -nkt+1:0;
xk = 1:nbars;

%% Compute STA/STC
 
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

eigvecs = u;

Nrow = 6; Ncol = 2;
figure(1); %STC analysis - excitatory
subplot(Nrow,Ncol,1); imagesc(ex_evec1); colormap gray; title('EXC #1'); axis square;ylabel('time')
subplot(Nrow,Ncol,3); imagesc(ex_evec2); colormap gray; title('EXC #2'); axis square;ylabel('time')
subplot(Nrow,Ncol,5); imagesc(ex_evec3); colormap gray; title('EXC #3'); axis square;ylabel('time')
subplot(Nrow,Ncol,7); imagesc(ex_evec4); colormap gray; title('EXC #4'); axis square;ylabel('time')
subplot(Nrow,Ncol,9); imagesc(ex_evec5); colormap gray; title('EXC #5'); axis square;ylabel('time')
subplot(Nrow,Ncol,11); imagesc(ex_evec6); colormap gray; title('EXC #6'); axis square;ylabel('time');
xlabel('bars'); 

%figure; %STC analysis - inhibitory
subplot(Nrow,Ncol,2); imagesc(in_evec1); colormap gray; title('INH #1'); axis square;
subplot(Nrow,Ncol,4); imagesc(in_evec2); colormap gray; title('INH #2'); axis square;
subplot(Nrow,Ncol,6); imagesc(in_evec3); colormap gray; title('INH #3'); axis square;
subplot(Nrow,Ncol,8); imagesc(in_evec4); colormap gray; title('INH #4'); axis square;
subplot(Nrow,Ncol,10); imagesc(in_evec5); colormap gray; title('INH #5'); axis square;
subplot(Nrow,Ncol,12); imagesc(in_evec6); colormap gray; title('INH #6'); axis square;
xlabel('bars'); 

%% Write figure.
set(gca, 'TickDirMode', 'auto', 'TickDir', 'out')

impath = 'figs';
set(gcf, 'PaperPosition', [0 0 Ncol*2 Nrow*2])
set(gcf, 'PaperPosition', [0 0 Ncol*2 Nrow*2])
print(gcf, '-depsc',sprintf('%s/RustFiga1.eps', impath))
unix(sprintf('./makePdfFiles.sh %s', impath));

%% 

% fname_model = '../Rust_cell21_60e3_6filt_20131115T175732/iter11.mat';
% fname_model = '../Rust_cell21_60000_6filt_20131116T161242/iter9.mat';
% fname_model = '../Rust_cell21_80000_4filt_20131116T161647/iter7.mat';
% fname_model = '../Rust_cell21_60000_6filt_20131116T161242/iter36.mat';
% fname_model = '../Rust_cell21_60000_4filt_20131116T210319/iter59.mat';

% fname_model = '../Rust_cell21_60000_6filt_20131116T211155/iter59.mat';
% fname_model = '../Rust_cell21_60000_4filt_20131117T114338/iter6.mat';

fname_model = '../Rust_cell21_60000_4filt_BUGFIX_20131122T121133/iter15.mat';

params = load(fname_model); params = params.params;

pvec = params.fnlin.get_params(params.fnlin);
model_vecs = reshape(pvec.w', nkt,nbars, []);

Nrow = 3; Ncol = 2;
figure(2); %STC analysis - excitatory
subplot(Nrow,Ncol,1); imagesc(model_vecs(:,:,1)); colormap gray; title('EXC #1'); axis square;ylabel('time')
subplot(Nrow,Ncol,3); imagesc(model_vecs(:,:,2)); colormap gray; title('EXC #2'); axis square;ylabel('time')
% subplot(Nrow,Ncol,5); imagesc(model_vecs(:,:,3)); colormap gray; title('EXC #3'); axis square;ylabel('time')
% subplot(Nrow,Ncol,7); imagesc(ex_evec4); colormap gray; title('EXC #4'); axis square;ylabel('time')
% subplot(Nrow,Ncol,9); imagesc(ex_evec5); colormap gray; title('EXC #5'); axis square;ylabel('time')
% subplot(Nrow,Ncol,11); imagesc(ex_evec6); colormap gray; title('EXC #6'); axis square;ylabel('time');
xlabel('bars'); 

%figure; %STC analysis - inhibitory
subplot(Nrow,Ncol,2); imagesc(model_vecs(:,:,3)); colormap gray; title('INH #1'); axis square;
subplot(Nrow,Ncol,4); imagesc(model_vecs(:,:,4)); colormap gray; title('INH #2'); axis square;
% subplot(Nrow,Ncol,6); imagesc(model_vecs(:,:,6)); colormap gray; title('INH #3'); axis square;
% subplot(Nrow,Ncol,8); imagesc(in_evec4); colormap gray; title('INH #4'); axis square;
% subplot(Nrow,Ncol,10); imagesc(in_evec5); colormap gray; title('INH #5'); axis square;
% subplot(Nrow,Ncol,12); imagesc(in_evec6); colormap gray; title('INH #6'); axis square;
xlabel('bars'); 

%% Write figure.
set(gca, 'TickDirMode', 'auto', 'TickDir', 'out')

impath = 'figs';
set(gcf, 'PaperPosition', [0 0 Ncol*2 Nrow*2])
set(gcf, 'PaperPosition', [0 0 Ncol*2 Nrow*2])
print(gcf, '-depsc',sprintf('%s/RustFiga2.eps', impath))
unix(sprintf('./makePdfFiles.sh %s', impath));

%% Locality

stc_vecs = zeros([size(ex_evec1) 6]);

stc_vecs(:,:,1) = ex_evec1;
stc_vecs(:,:,2) = ex_evec2;
stc_vecs(:,:,3) = ex_evec3;
stc_vecs(:,:,4) = ex_evec4;
stc_vecs(:,:,5) = ex_evec5;
stc_vecs(:,:,6) = ex_evec6;

loc_pow_stc =  mean(mean(stc_vecs.^2,2),3); % accumulate the power in each time bin
loc_pow_lds =  mean(mean(model_vecs.^2,2),3); % accumulate the power in each time bin

figure(4); clf; 
plot(flipud(loc_pow_stc/norm(loc_pow_stc)), 'linewidth', 3); hold on;
plot(flipud(loc_pow_lds/norm(loc_pow_lds)), 'r', 'linewidth', 3)
axis tight

ylabel('average pixel magnitude across filters')
xlabel('time lag')

%% Write figure.
set(gca, 'TickDirMode', 'auto', 'TickDir', 'out', 'box', 'off')

impath = 'figs';
set(gcf, 'PaperPosition', [0 0 2.3 2.3])
print(gcf, '-depsc',sprintf('%s/RustFigc.eps', impath))
unix(sprintf('./makePdfFiles.sh %s', impath));











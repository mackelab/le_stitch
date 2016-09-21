%% set up

clear all
addpath(genpath('/home/mackelab/Desktop/Projects/Stitching/code/code_le_stitch/figure_generation'))

abs_path_to_data = '/home/mackelab/Desktop/Projects/Stitching/results/cosyne_poster/';
exps = 'experiment_1';

cd([abs_path_to_data, exps])


p = 1000;
overlap = 200; % data exists for overlap = [1, 3, 9]

N = 2400;
%% load raw data
spikes = load('data/spikes_20trials_10msBins');
data = [];
for i=1:length(spikes.spikes_out)
    data = [data, spikes.spikes_out{i}];
end
covy_full = cov( [data(:,2:end);data(:,1:end-1)]' );
covy = covy_full(1:p,1:p);
covy_tl = covy_full(1:p,p+(1:p));

%% get neuron_shuffle
tmp = load('data/neuron_shuffle');
neuron_shuffle = tmp.neuron_shuffle+1;
neuron_unshuffle = tmp.neuron_unshuffle+1;

%%
subplot(1,2,1)
imagesc(covy(neuron_shuffle,neuron_shuffle))
title('cov(y), neurons in order as fitted')

subplot(1,2,2)
imagesc(covy(:,:))
title('cov(y), neurons in original order')

%%
shuffle11 = sort(neuron_shuffle(1:400));
shuffle12 = sort(neuron_shuffle(401:600));
shuffle22 = sort(neuron_shuffle(601:end));

%shuffle11 = sort(neuron_shuffle(neuron_shuffle<=400));
%shuffle12 = sort(neuron_shuffle(neuron_shuffle>400 & neuron_shuffle<=600));
%shuffle22 = sort(neuron_shuffle(neuron_shuffle>600));
shuffle = [shuffle11,shuffle12,shuffle22];

idx_stitched = ones(1000);
idx_stitched(1:600,1:600) = 0; idx_stitched(401:1000,401:1000) = 0;
idx_stitched = logical(idx_stitched);
%% get param EM fit
loadfile = load(['fits/params_naive_p1000_iter',num2str(N),'.mat']);
T = loadfile.T;
pars_hat = loadfile.estPars;
[p,n] = size(pars_hat.C);
pars_hat.R = diag(pars_hat.R);
stats_hat = loadfile.stats_h;
obs_scheme.obs_pops = loadfile.obs_pops;
obs_scheme.obs_time = loadfile.obs_time;
obs_scheme.sub_pops = loadfile.sub_pops;
likes = loadfile.ll;

pars_hat.Pi_h = direct_dlyap(pars_hat.A,pars_hat.Q);
covy_h = pars_hat.C * pars_hat.Pi_h * pars_hat.C' + diag(pars_hat.R);
covy_h = covy_h(neuron_unshuffle, neuron_unshuffle);

m = min([min(covy(:)), min(covy_h(:))]);
%M = max([max(covy(:)), max(covy_h(:))]);
M = 0.02;
figure;
subplot(1,2,1)
imagesc(covy(shuffle,shuffle), [m,M])
hold on
line([0,600]+.5,[600,600]+.5, 'color', 'g','linewidth',2)
line([600,600]+.5,[0,600]+.5, 'color', 'g','linewidth',2)
line([400,1000]+.5,[400,400]+.5, 'color', 'c','linewidth',2)
line([400,400]+.5,[400,1000]+.5, 'color', 'c','linewidth',2)
set(gca, 'TickDir', 'out')
box off
axis square
title('empirical instantaneous covariances')

subplot(1,2,2)
imagesc(covy_h(shuffle,shuffle), [m,M])
hold on
line([0,600]+.5,[600,600]+.5, 'color', 'g','linewidth',2)
line([600,600]+.5,[0,600]+.5, 'color', 'g','linewidth',2)
line([400,1000]+.5,[400,400]+.5, 'color', 'c','linewidth',2)
line([400,400]+.5,[400,1000]+.5, 'color', 'c','linewidth',2)
set(gca, 'TickDir', 'out')
box off
axis square
title('estimated instantaneous covariances')
colorbar
%%
covy_e = covy(shuffle,shuffle);
covy_e_tl = covy_tl(shuffle,shuffle);

lls = zeros(2400,1); pars_hats = cell(24,1);
for i = 1:24.
    loadfile = load(['fits/params_naive_p1000_iter',num2str(100*i),'.mat']);
    if i == 1
        initPars = loadfile.initPars;
    end
    pars_hats{i} = loadfile.estPars;
    lls((i-1)*100+(1:100)) = loadfile.ll;
end

perf_yy = zeros(25,2);
perf_yy_tl = zeros(24,1);

pars_hat = initPars;
covy_h = pars_hat.C * pars_hat.Pi * pars_hat.C' + diag(pars_hat.R);
covy_h = covy_h(neuron_unshuffle, neuron_unshuffle); % fully unshuffle        
covy_h = covy_h(shuffle,shuffle);   % re-shuffle accross groups
covy_h_tl =  pars_hat.C * pars_hat.A * pars_hat.Pi * pars_hat.C';
covy_h_tl = covy_h_tl(neuron_unshuffle, neuron_unshuffle); % fully unshuffle        
covy_h_tl = covy_h_tl(shuffle,shuffle);   % re-shuffle accross groups

tmp = corrcoef(covy_e(~idx_stitched), covy_h(~idx_stitched));
perf_yy(1,1) = tmp(1,2);
tmp = corrcoef(covy_e(idx_stitched), covy_h(idx_stitched));
perf_yy(1,2) = tmp(1,2);

tmp = corrcoef(covy_e_tl(~idx_stitched), covy_h_tl(~idx_stitched));
perf_yy_tl(1,1) = tmp(1,2);
tmp = corrcoef(covy_e_tl(idx_stitched), covy_h_tl(idx_stitched));
perf_yy_tl(1,2) = tmp(1,2);

for i = 1:24
    
    pars_hat = pars_hats{i};
    pars_hat.Pi_h = direct_dlyap(pars_hat.A,pars_hat.Q);
    
    covy_h = pars_hat.C * pars_hat.Pi_h * pars_hat.C' + pars_hat.R;
    covy_h = covy_h(neuron_unshuffle, neuron_unshuffle); % fully unshuffle        
    covy_h = covy_h(shuffle,shuffle);   % re-shuffle accross groups
    
    covy_h_tl =  pars_hat.C * pars_hat.A * pars_hat.Pi_h * pars_hat.C';
    covy_h_tl = covy_h_tl(neuron_unshuffle, neuron_unshuffle); % fully unshuffle        
    covy_h_tl = covy_h_tl(shuffle,shuffle);   % re-shuffle accross groups

    tmp = corrcoef(covy_e(idx_stitched), covy_h(idx_stitched));
    perf_yy(i+1,2) = tmp(1,2);
    tmp = corrcoef(covy_e(~idx_stitched), covy_h(~idx_stitched));
    perf_yy(i+1,1) = tmp(1,2);

    tmp = corrcoef(covy_e_tl(idx_stitched), covy_h_tl(idx_stitched));
    perf_yy_tl(i+1,2) = tmp(1,2);
    tmp = corrcoef(covy_e_tl(~idx_stitched), covy_h_tl(~idx_stitched));
    perf_yy_tl(i+1,1) = tmp(1,2);
end
%%
figure;
bar([0,1], perf_yy(end,:), 'faceColor', [1,1,0]);
hold on
bar([0,1], perf_yy(1,:), 'faceColor', 0.7*[1,1,1]);
bar([2,3], perf_yy_tl(end,:), 'faceColor', [1,1,0]);
bar([2,3], perf_yy_tl(1,:), 'faceColor', 0.7*[1,1,1]);
box off
set(gca, 'TickDir', 'out')
set(gca, 'XTick', 0:3)
set(gca, 'XTickLabel', {'observed', 'non-obs.','observed', 'non-obs.'})



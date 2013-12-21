% addpath /Users/evan/Desktop/RUSTDATA/v1data_Rust_Neuron2005/
% addpath /home/pillowlab/v1data_Rust_Neuron2005
% addpath Kalman_Code/
% addpath Kalman_Code/Kalman/
% addpath Kalman_Code/KPMstats/
% addpath Kalman_Code/KPMtools/
% addpath ~/pillowlab/lib/minFunc_2012/
% addpath ~/pillowlab/lib/minFunc_2012/minFunc/
% addpath ~/pillowlab/lib/minFunc_2012/minFunc/compiled/
% addpath ~/pillowlab/ncclabcode/
% addpath NlinStimDep/
% 
% load 544l029.p21_stc.mat
% %load 541l009.p27_stc.mat
% 
% % NOTE: possible cells: 21, 27, ...
% 
% nT  = 60000;
% 
% % stim = stim(1:nT,:);
% % spikes_per_frm = spikes_per_frm(1:nT);
% 
% %Stim = stim;
% %sps = spikes_per_frm';
% %nsp = sum(sps);
% 
% ii=6:min(21, size(stim,2));
% Stim = stim(1:nT,ii);
% sps = spikes_per_frm(1:nT)';
% sps = [sps(3:end);0;0];
% nsp = sum(sps);
% 
% global RefreshRate;
% RefreshRate = 100;  % (in Hz)
% 
% nstim = length(stim);  % number of stimuli
% nkt=12; %dimension of filter
% nbars = size(Stim,2);  % stimulus width
% 
% ttk = -nkt+1:0;
% xk = 1:nbars;
% 
% ex_evec1 = reshape(u(:,1),nkt,nbars); 
% ex_evec2 = reshape(u(:,2),nkt,nbars); 
% ex_evec3 = reshape(u(:,3),nkt,nbars);
% ex_evec4 = reshape(u(:,4),nkt,nbars); 
% ex_evec5 = reshape(u(:,5),nkt,nbars); 
% ex_evec6 = reshape(u(:,6),nkt,nbars); 
% ex_evec7 = reshape(u(:,7),nkt,nbars); 
% ex_evec8 = reshape(u(:,8),nkt,nbars);
% 
% in_evec1 = reshape(u(:,length(evals)),nkt,nbars); 
% in_evec2 = reshape(u(:,length(evals)-1),nkt,nbars); 
% in_evec3 = reshape(u(:,length(evals)-2),nkt,nbars);
% in_evec4 = reshape(u(:,length(evals)-3),nkt,nbars); 
% in_evec5 = reshape(u(:,length(evals)-4),nkt,nbars);
% in_evec6 = reshape(u(:,length(evals)-5),nkt,nbars);
% in_evec7 = reshape(u(:,length(evals)-6),nkt,nbars);
% in_evec8 = reshape(u(:,length(evals)-7),nkt,nbars); 
% 
% eigvecs = u;

%% Compute correlation of test data in GQM with increasing number of filters. 

nT_test = 1000;
test_ii = nT + (1:nT_test);

eStim = Stim; tStim = stim(test_ii,ii);
esps  = sps;  tsps  = [spikes_per_frm(test_ii+2)']; %tsps = tsps(1:end-2);

% trStim = stim(nT+1:end,ii);
% trsps = tsps;
NSpHist = 5;

NmaxFilt = 10;

tcorr_gqm = zeros(NmaxFilt,1);
tcorr_rgqm = zeros(NmaxFilt,1);
tcorr_qlds = zeros(NmaxFilt,1);

for idx = 1:NmaxFilt
   ndx = floor(idx/2); 
   B = eigvecs(:, [1:(ndx+mod(idx,2)) (end-ndx+1):end]);
   fo = multiconv(eStim, reshape(B,nkt,nbars,idx));
   fot = multiconv(tStim, reshape(B,nkt,nbars,idx));
      
   rfo = [fo makeStimRows([0;esps(1:end-1)], NSpHist)];
   rfot = [fot makeStimRows([0;tsps(1:end-1)], NSpHist)];
   
   tvdat = nlin_quadratic(fo,esps,fot);
   r = corrcoef(tvdat, tsps);
   tcorr_gqm(idx) = r(1,2)   
   
   [rtvdat gqm_param] = nlin_quadratic(rfo,esps,rfot);
   r = corrcoef(rtvdat, tsps);
   tcorr_rgqm(idx) = r(1,2)
   
end

%% Check gqm using lds filters

nlp = params.fnlin.get_params(params.fnlin);
fo_lds = multiconv(eStim, reshape(nlp.w',nkt,nbars,params.fnlin.outDim));
fot_lds = multiconv(tStim, reshape(nlp.w',nkt,nbars,params.fnlin.outDim));

[tvdat_ldsgqm ldsgqmparam]= nlin_quadratic(fo_lds,esps,fot_lds);
r = corrcoef(tvdat, tsps)

%% Predict from kalman filter 

tStim_stack = makeStimRows(tStim, nkt);

nStateDim = params.fnlin.outDim; 
nStimDim  = params.fnlin.inDim; 
nObsDim   = 1;
nSDim = 0;
nHDim = 0;
s = zeros(length(tsps), nSDim*nObsDim);          % "history"
h = zeros(length(tsps), nHDim);          % more "history"

%% Simulate
[xx, ~, ~, ~, tvdat_lds] = kalman_filter(tsps', tStim_stack', s', h', params.d, params.A, params.C, params.D, params.E, params.Q, params.R, params.initx, params.initV, params.fnlin);

r = corrcoef(tvdat_lds, tsps)

%% Try simulation again

yy = zeros(1,length(tsps));
xx = zeros(size(xx));
for t = 1:length(tsps)
    if(t == 1)
        xx(:,1) = params.initx;
    else
        xx(:,t) = params.A * xx(:,t-1);
    end
    yy(t) = params.C*( xx(:,t)  + params.fnlin.f(params.fnlin, tStim_stack(t,:)')) + params.d;
end

r = corrcoef(yy, tsps)

%% Make figure

figure(4); clf; hold on
plot(tcorr_gqm, 'k', 'linewidth', 2)
plot(tcorr_rgqm, 'b', 'linewidth', 2)
kk = (1:length(tcorr_gqm));
plot(kk,r(2,1)*ones(size(kk)), 'r', 'linewidth', 2);
axis tight
xlabel('number of filters')
ylabel('correlation')

%% Write figure

set(gca, 'TickDirMode', 'auto', 'TickDir', 'out')

impath = 'figs';
set(gcf, 'PaperPosition', [0 0 3 3])
set(gcf, 'PaperPosition', [0 0 3 3])
print(gcf, '-depsc',sprintf('%s/RustFigb.eps', impath))
unix(sprintf('./makePdfFiles.sh %s', impath));

%% simulate thing 

params_tt = params;
params_tt.u = tStim_stack';
params_tt.s = s';
params_tt.h = h';
params_tt.models = ones(1,length(tStim_stack));
[x, y] = sample_lds(params_tt);

%% 

nlag = 10;
figure(5); clf; hold on; 
% plot(-nlag:nlag,xcorr(tvdat_lds-mean(tvdat_lds), nlag, 'coeff'), 'r', 'linewidth', 3)
plot(-nlag:nlag,xcorr(y-mean(y), nlag, 'coeff'), 'r', 'linewidth', 3)
plot(-nlag:nlag,xcorr(rtvdat - mean(rtvdat), nlag, 'coeff'), 'b', 'linewidth', 3)
plot(-nlag:nlag,xcorr(tvdat - mean(tvdat), nlag, 'coeff'), 'm', 'linewidth', 3)
plot(-nlag:nlag,xcorr(esps - mean(esps), nlag, 'coeff'), 'k', 'linewidth', 3)
axis tight
xlabel('lags')
ylabel('correlation')
title('autocorrelation')

set(gca, 'TickDirMode', 'auto', 'TickDir', 'out', 'box', 'off')
impath = 'figs';
set(gcf, 'PaperPosition', [0 0 2.3 2.3])
set(gcf, 'PaperPosition', [0 0 2.3 2.3])
print(gcf, '-depsc',sprintf('%s/RustFigd.eps', impath))
unix(sprintf('./makePdfFiles.sh %s', impath));


%% Explore sim

idx = 9000:9100;

figure(13);clf; hold on;
plot(tvdat_lds(idx),'linewidth', 2)
plot(tvdat_ldsgqm(idx),'r')
plot(rtvdat(idx), 'g')
plot(tvdat(idx), 'm')
plot(tsps(idx), 'k.')
% 



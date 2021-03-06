%% Introduction
% Following code serves to compare the LDS core functionalities as 
% implemented by Lars Buesing and Jakob Macke in pop_spike_dyn, with
% the results of a python implementation by Marcel Nonnenmacher.
% It works by running the python code (not seen here) and exporting the
% setup and results into a .mat file, from which the Matlab code runs
% the exact same steps (see below). Then we compare.


clear all
close all
clc

%% load, translate, compute

addpath( ...
   genpath('/home/mackelab/Desktop/Projects/Stitching/code/pop_spike_dyn'))
%dataSet = 'LDS_save_longFitLowDimTempStitching_mediocreFit';

dataSet = 'LDS_save';

load(['/home/mackelab/Desktop/Projects/Stitching/results/test_problems/', dataSet, '.mat'])
% gives data traces x, y, data length T
% also gives true parameters {A,Q,mu0,V0,C,R} used to generate the data
% also gives E-step results (E[xt], E[x_t x_t'], E[x_t x_{t-1}']) as
%            returned by the python implementation
% also gives M-step results {A_h,Q_h,mu0_h,V0_h,C_h,R_h} as returned
%            by the python implementation (usually based on the E-step
%            results also provided)

% need to unwrap the observation scheme

numPops = length(obsScheme.subpops);
if numPops > 2
    disp(['Warning: testing code was intended for two subpopulations ',...
          'only: The full and the empty one.'])
end
popSize = zeros(numPops, 1);
idxFullPop = false(numPops,1);
for i = 1:length(obsScheme.subpops)
 popSize(i) = length(obsScheme.subpops{i});
 idxFullPop(i) = popSize(i)==size(y,1); % gives full populations? Rest is empty!
end
if any(popSize~=0 & popSize~=size(y,1))
    error(['This code cannot handle spatial stitching. Subpopulations',...
           'have to be either full or empty.'])
end
idxObs = ismember(obsScheme.obsPops+1, find(idxFullPop));
tObs   = false(T,1);
if idxObs(1)
    tObs(1:obsScheme.obsTime(1)) = 1;
end
for i = 2:length(obsScheme.obsTime)
    if idxObs(i)
        tObs(obsScheme.obsTime(i-1)+1:obsScheme.obsTime(i)) = 1;
    end
end
tObs = logical(tObs);


% squeezing parameters into Lars' formatting
trueparams.model.A  = A;
trueparams.model.B  = B;
trueparams.model.Q  = Q;
trueparams.model.x0 = mu0;
trueparams.model.Q0 = V0;
trueparams.model.C  = C;
trueparams.model.d  = d(:);
if min(size(R)) == 1
 trueparams.model.R  = diag(R);    
else
 trueparams.model.R  = R;
end
trueparams.model.Pi = 1;
trueparams.model.notes.useR = 1;
trueparams.model.notes.useS = 0;
trueparams.model.notes.useB = 0;   % this is important! Change if necessary
trueparams.model.notes.learnx0 = 1;
trueparams.model.notes.learnQ0 = 1;
trueparams.model.notes.learnA = 1;
trueparams.model.notes.learnR = 1;
trueparams.model.notes.diagR  = 1;
trueparams.model.notes.useCMask= 0;
trueparams.model.inferenceHandle =  @LDSInference;
trueparams.model.MStepHandle = @LDSMStep;
trueparams.model.ParamPenalizerHandle = @LDSemptyParamPenalizerHandle;
trueparams.opts.initMethod = 'params';
trueparams.opts.algorithmic.MStepObservation = [];
trueparams.opts.algorithmic.TransformType = '0';
trueparams.opts.algorithmic.EMIterations.maxIter = 100;
trueparams.opts.algorithmic.EMIterations.maxCPUTime = Inf;
trueparams.opts.algorithmic.EMIterations.progTolvarBound = 10^(-6);
trueparams.opts.algorithmic.EMIterations.abortDecresingVarBound = 1;

seq.x = x;   % true FULLY OBSERVED sequence (using this is cheating!)
seq.y = y;   %
seq.T = double(T);

% what the python E-step got as parameter estimate initializations
pyparamsIn.model.A = A_0;
pyparamsIn.model.B = B_0;
pyparamsIn.model.Q = Q_0;
pyparamsIn.model.x0 = mu0_0;
pyparamsIn.model.Q0 = V0_0;
pyparamsIn.model.C = C_0;
pyparamsIn.model.d = d_0(:);
if min(size(R_0)) == 1
 pyparamsIn.model.R  = diag(R_0);    
else
 pyparamsIn.model.R  = R_0;
end
pyparamsIn.model.Pi = 1;
pyparamsIn.model.notes = trueparams.model.notes;
pyparamsIn.model.inferenceHandle =  @LDSInference;
pyparamsIn.model.MStepHandle = @LDSMStep;
pyparamsIn.model.ParamPenalizerHandle = @LDSemptyParamPenalizerHandle;
pyparamsIn.opts = trueparams.opts;

% what the python M-step returned
pyparamsOut.model.A = A_1;
pyparamsOut.model.B = B_1;
pyparamsOut.model.Q = Q_1;
pyparamsOut.model.x0 = mu0_1;
pyparamsOut.model.Q0 = V0_1;
pyparamsOut.model.C = C_1;
if min(size(R_1)) == 1
 pyparamsOut.model.R  = diag(R_1);    
else
 pyparamsOut.model.R  = R_1;
end
pyparamsOut.model.d  = d_1(:);
pyparamsOut.model.Pi = 1;
pyparamsOut.model.notes = trueparams.model.notes;
pyparamsOut.model.inferenceHandle =  @LDSInference;
pyparamsOut.model.MStepHandle = @LDSMStep;
pyparamsOut.model.ParamPenalizerHandle = @LDSemptyParamPenalizerHandle;
pyparamsOut.opts = trueparams.opts;

pySeq = seq; % translates between the moments that my python code works 
             % with and the means and covariances that Lars' code uses.
xDim = size(trueparams.model.A,1);
yDim = size(trueparams.model.C,1);



pySeqObs = pySeq;
pySeqObs.x = pySeq.x(:,tObs);
pySeqObs.y = pySeq.y(:,tObs);
pySeqObs.T = sum(tObs);
seqObs     = pySeqObs;

pySeq.posterior.xsm  = zeros(xDim,T);
pySeq.posterior.Vsm  = zeros(xDim*T,    xDim);
pySeq.posterior.VVsm = zeros(xDim*(T-1),xDim);
 
for i = 1:xDim
    pySeq.posterior.xsm(i,:) = Ext(i,:);
    for j = 1:xDim
        pySeq.posterior.Vsm((0:xDim:end-1)+i, j)  = ...
                               squeeze(Extxt(i,j,:))'-(Ext(i,:).*Ext(j,:));
        pySeq.posterior.VVsm((0:xDim:end-1)+i, j) = ...
               squeeze(Extxtm1(i,j,2:end))'-(Ext(i,2:end).*Ext(j,1:end-1));        
    end
end
pySeq.u = u;

pySeqObs.posterior.xsm  = zeros(xDim,sum(tObs));
pySeqObs.posterior.Vsm  = zeros(xDim*sum(tObs),    xDim);
pySeqObs.posterior.VVsm = zeros(xDim*(sum(tObs)-1),xDim);
for i = 1:xDim
    pySeqObs.posterior.xsm(i,:) = pySeq.posterior.xsm(i,tObs);
    for j = 1:xDim
        pySeqObs.posterior.Vsm((0:xDim:end-1)+i, j)  = ...
                               (squeeze(Extxt(i,j,tObs))'-(Ext(i,tObs).*Ext(j,tObs)))';
        tmp = find(tObs)-1;
        pySeqObs.posterior.VVsm((0:xDim:end-1)+i, j) = ...
               (squeeze(Extxtm1(i,j,tmp(2:end)))'-(Ext(i,tmp(2:end)).*Ext(j,tmp(1:end-1))))';        
    end
end

clearvars -except xDim yDim trueparams pyparamsOut pyparamsIn seq seqObs pySeq pySeqObs T tObs


 
% % E-step: Get Matlab version of Ext, Extxt, Extxtm1
% seq = LDSInference(pyparamsIn,seq); % adds seq.posterior, otherwise returns
%                                     % the exact same seq 
%                                     
% % The M-step for A, Q does not change under temporal stitching!
% matparamsOut = LDSMStepLDS(pyparamsIn,seq);
% % The M-step for C,d,R for purely temporal stitching is the M-step
% % applied to the observed sequence!
% matparamsOut = LDSMStepObservation(matparamsOut,pySeqObs);
% % note that we start out from pySeq, i.seqe. the structure that contains
% % the posteriors as translated from the python-version E-step, to
% % directly compare the two M-steps. Could just as well replace pySeq
% % with seq (containing the posterior from the Matlab-version) E-step to
% % compare the results after a full E-Step-M-Step iteration.

%%
% E-step: Get Matlab version of Ext, Extxt, Extxtm1
pyparamsInObs = pyparamsIn;
pyparamsInObs.model.A = pyparamsIn.model.A * pyparamsIn.model.A;
% The following formula is only valid in 1 dimension (*could* generalise)!
pyparamsInObs.model.Q = pyparamsIn.model.A * pyparamsIn.model.Q * pyparamsIn.model.A + pyparamsIn.model.Q;
pyparamsInObs.model.Q0 = pyparamsIn.model.A * pyparamsIn.model.Q0 * pyparamsIn.model.A' + pyparamsIn.model.Q0';                   

seqObs = LDSInference(pyparamsInObs,seqObs);
                                    
% The M-step for A, Q does not change under temporal stitching!
matparamsOut = LDSMStepLDS(pyparamsIn,seqObs);
%matparamsOut.model.Q = matparamsOut.model.Q/(1+matparamsOut.model.A); 
%matparamsOut.model.A = sqrt(matparamsOut.model.A); 

% The M-step for C,d,R for purely temporal stitching is the M-step
% applied to the observed sequence!
matparamsOut = LDSMStepObservation(matparamsOut,seqObs);
% note that we start out from pySeq, i.e. the structure that contains
% the posteriors as translated from the python-version E-step, to
% directly compare the two M-steps. Could just as well replace pySeq
% with seq (containing the posterior from the Matlab-version) E-step to
% compare the results after a full E-Step-M-Step iteration.

%% make comparison figures for E-step
% The E-step for standard LDS needs to correctly reproduce three 
% types of expected values for the ensuing M-step:
% E[x_t]
% E[x_t x_t']
% E[x_t x_{t-1}']
% We compare the python results with those stored in seq.posterior from the 
% Matlab version in three consecutive figures:

figure('Units','normalized','Position',[0.05,0.1,0.5,0.9]); % check E[x_t]

clrs = copper(xDim);

for i = 1:xDim
    
%     subplot(xDim,2,1 + (i-1)*2)    % could also plot raw traces if 
%     plot(x(i,:), 'b')              % resuls seem really messed up ...
%     hold on
%     plot(Ext(i,:), 'r:')
%     plot(seq.posterior.xsm(i,:), 'g--')
%     title(['inferred latent states, dim #', num2str(i)])
%     
    subplot(ceil(xDim/2),2,i)
    m = min(seqObs.posterior.xsm(:));
    M = max(seqObs.posterior.xsm(:));
    plot(-1000, -1000, '.', 'color', clrs(end,:), 'markerSize', 10) 
    hold on                                     % just to get 
    plot(-1000, -1000, 'o', 'color', clrs(1,:)) % the legend right...    
    line([1.1*m,1.1*M], ...
     [1.1*m,1.1*M],'color','c')
    axis(1.1*[m,M+0.001,m,M+0.001])
    plot(pySeqObs.posterior.xsm(i,:), seqObs.posterior.xsm(i,:), '.', 'color', clrs(end,:), 'markerSize', 10)
    plot(pySeqObs.posterior.Vsm((0:xDim:end-1)+i, i), ...
         seqObs.posterior.Vsm((0:xDim:end-1)+i, i), ...
         'o', 'color', clrs(1,:))
    xlabel('python')
    ylabel('Matlab')
    title(['latent dim #', num2str(i)])
    if i == 1
        legend('E[X_i]', 'var(X_i)', 'location', 'NorthWest')
        legend boxoff
    end
    box off
    set(gca, 'TickDir', 'out')
    MSE_Ext = mean( (pySeqObs.posterior.xsm(i,:)-seqObs.posterior.xsm(i,:)).^2 );
    text(0.3*M, 0.6*m, ['MSE: ', num2str(MSE_Ext)])
end

%--------------------------------------- check E[x_t x_t'] ----------------
figure('Units', 'normalized','Position', [0.35,0.1,0.5,0.5]);

lgnd = cell(xDim,1);
m = min(seqObs.posterior.Vsm(:));
M = max(seqObs.posterior.Vsm(:));


for i = 1:xDim
    plot(-1000, -1000, 'o', 'color', clrs(end-(abs(1-i)),:))
    hold on
    lgnd{i} = ['|i-j| =', num2str(abs(1-i))];    
end
line([1.1*m,1.1*M], ...
     [1.1*m,1.1*M],'color','c')
axis(1.1*[m,M+0.001,m,M+0.001])
Mm = zeros(xDim,xDim,seqObs.T);
Mp = zeros(xDim,xDim,seqObs.T);
for i = 1:xDim
    for j = 1:xDim
        Mp(i,j,:) = pySeqObs.posterior.Vsm((0:xDim:end-1)+i, j);
        Mm(i,j,:) = seqObs.posterior.Vsm((0:xDim:end-1)+i, j);
        plot(squeeze(Mp(i,j,:)), ...
             squeeze(Mm(i,j,:)), 'o', 'color', clrs(end-(abs(i-j)),:))
    end
end
title(['cov[x_t, x_t^T]'])
xlabel('python')
ylabel('Matlab')
legend(lgnd, 'location','NorthWest')
legend boxoff
MSE_cov = mean( (Mp(:) - Mm(:)).^2 );
text(0.3*M, 0.6*m, ['MSE: ', num2str(MSE_cov)])
box off
set(gca, 'TickDir', 'out')

%------------------------------------------------ check E[x_t x_{t-1}'] ---
figure('Units', 'normalized','Position', [0.65,0.1,0.5,0.5]); 

clrs = copper(xDim);
m = min(seqObs.posterior.VVsm(:));
M = max(seqObs.posterior.VVsm(:));
for i = 1:xDim
    plot(-1000, -1000, 'o', 'color', clrs(end-(abs(1-i)),:))
    hold on
    lgnd{i} = ['|i-j| =', num2str(abs(1-i))];    
end
line([1.1*m,1.1*M], ...
     [1.1*m,1.1*M],'color','c')
axis(1.1*[m,M+0.001,m,M+0.001])
Mm = zeros(xDim,xDim,seqObs.T-1);
Mp = zeros(xDim,xDim,seqObs.T-1);
for i = 1:xDim
    for j = 1:xDim
        Mm(i,j,:) = seqObs.posterior.VVsm((0:xDim:end-1)+i, j);
        Mp(i,j,:) = pySeqObs.posterior.VVsm((0:xDim:end-1)+i, j);
        plot(squeeze(Mp(i,j,:)), ...
             squeeze(Mm(i,j,:)), 'o', 'color', clrs(end-(abs(i-j)),:))
        hold on
    end
end
legend(lgnd, 'location','NorthWest')
legend boxoff
title('cov[x_t, x_{t-1}^T]')
xlabel('python')
ylabel('Matlab')
MSE_covdt = mean( (Mp(:) - Mm(:)).^2 );
text(0.3*M, 0.6*m, ['MSE: ', num2str(MSE_covdt)])
box off
set(gca, 'TickDir', 'out')

%% make comparison figures for M-step
% now the M-step produces a new set of parameters \theta = {A,Q,mu0,V0,R,C}
clrs = copper(xDim);

% checking A, Q
%--------------------------------------------------------------------------
figure('Units', 'normalized','Position', [0.2,0.2,0.6,0.5]);  
m = min([min(pyparamsIn.model.A(:)), ...
         min(matparamsOut.model.A(:)), ...
         min(pyparamsOut.model.A(:))]);
M = max([max(pyparamsIn.model.A(:)), ...
         max(matparamsOut.model.A(:)), ...
         max(pyparamsOut.model.A(:))]);
subplot(2,4,1)
imagesc(pyparamsIn.model.A)
caxis([m,M])
set(gca,'TickDir', 'out')
title('A^2 (initialization)')
subplot(2,4,2)
imagesc(matparamsOut.model.A)
caxis([m,M])
set(gca,'TickDir', 'out')
title('A^2 (Matlab)')
subplot(2,4,3)
pyparamsOut.model.A2 = pyparamsOut.model.A*pyparamsOut.model.A;
imagesc(pyparamsOut.model.A2)
caxis([m,M])
set(gca,'TickDir', 'out')
title('A^2 (python)')
subplot(2,4,4)
M = max([max(matparamsOut.model.A(:)), max(pyparamsOut.model.A2(:))]);
m = min([min(matparamsOut.model.A(:)), min(pyparamsOut.model.A2(:))]);
plot(-1000,-1000, '.', 'color', clrs(end,:), 'markerSize', 10)
hold on
plot(-1000,-1000, 'o', 'color', clrs(1,:))
line([1.1*m,1.1*M], ...
     [1.1*m,1.1*M],'color','c')
axis(1.1*[m,M+0.001,m,M+0.001])
plot(pyparamsOut.model.A2(:), ...
     matparamsOut.model.A(:), '.', 'color', clrs(end,:))
plot(eig(pyparamsOut.model.A2), ...
     eig(matparamsOut.model.A), 'o', 'color', clrs(1,:))
set(gca,'TickDir', 'out')
box off
xlabel('python')
ylabel('Matlab')
legend('A^2_{ij}', '[eig(A^2)]_i', 'location', 'NorthWest')
legend boxoff
MSE_A = mean( (matparamsOut.model.A(:) - pyparamsOut.model.A2(:)).^2 );
MSE_Aeig = mean((eig(matparamsOut.model.A)-eig(pyparamsOut.model.A2)).^2 );
text(0.3*M, 0.5*m, ['MSE (eig): ', num2str(MSE_Aeig)])
text(0.3*M, 0.5*m+0.1*(M-m), ['MSE: ', num2str(MSE_A)])
corr_A = corr(matparamsOut.model.A(:), pyparamsOut.model.A2(:));
text(0.3*M, 0.5*m+0.2*(M-m), ['corr: ', num2str(corr_A)])

m = min([min(pyparamsIn.model.Q(:)), ...
         min(matparamsOut.model.Q(:)), ...
         min(pyparamsOut.model.Q(:))]);
M = max([max(pyparamsIn.model.Q(:)), ...
         max(matparamsOut.model.Q(:)), ...
         max(pyparamsOut.model.Q(:))]);
subplot(2,4,5)
imagesc(pyparamsIn.model.Q)
caxis([m,M])
set(gca,'TickDir', 'out')
title('AQA^T + Q (initialization)')
subplot(2,4,6)
imagesc(matparamsOut.model.Q)
caxis([m,M])
set(gca,'TickDir', 'out')
title('AQA^T + Q (Matlab)')
subplot(2,4,7)
pyparamsOut.model.AQAQ = ...
        pyparamsOut.model.A * pyparamsOut.model.Q * pyparamsOut.model.A ...
      + pyparamsOut.model.Q;
imagesc(pyparamsOut.model.AQAQ)
caxis([m,M])
set(gca,'TickDir', 'out')
title('AQA^T + Q (python)')
subplot(2,4,8)
M = max([max(matparamsOut.model.Q(:)), max(pyparamsOut.model.AQAQ(:))]);
m = max([min(matparamsOut.model.Q(:)), min(pyparamsOut.model.AQAQ(:))]);
plot(-1000,-1000, '.', 'color', clrs(end,:), 'markerSize', 10)
hold on
plot(-1000,-1000, 'o', 'color', clrs(1,:))
line([1.1*m,1.1*M], ...
     [1.1*m,1.1*M],'color','c')
axis(1.1*[m,M+0.001,m,M+0.001])
plot(pyparamsOut.model.AQAQ(:), ...
     matparamsOut.model.Q(:), '.', 'color', clrs(end,:))
plot(eig(pyparamsOut.model.AQAQ), ...
     eig(matparamsOut.model.Q), 'o', 'color', clrs(1,:))
set(gca,'TickDir', 'out')
box off
xlabel('python')
ylabel('Matlab')
legend('[AQA^T + Q]_{ij}', '[eig(AQA^T + Q)]_i', 'location', 'NorthWest')
legend boxoff
MSE_Q = mean( (matparamsOut.model.Q(:) - pyparamsOut.model.AQAQ(:)).^2 );
MSE_Qeig = mean( (eig(matparamsOut.model.Q)-eig(pyparamsOut.model.AQAQ)).^2 );
text(0.3*M, 0.5*m, ['MSE (eig): ', num2str(MSE_Qeig)])
text(0.3*M, 0.5*m+0.1*(M-m), ['MSE: ', num2str(MSE_Q)])
corr_Q = corr(matparamsOut.model.Q(:), pyparamsOut.model.AQAQ(:));
text(0.3*M, 0.5*m+0.2*(M-m), ['corr: ', num2str(corr_Q)])

% checking mu0, v0
%--------------------------------------------------------------------------
figure('Units', 'normalized','Position', [0.25,0.25,0.6,0.5]);  
clrs = copper(xDim);
m = min([min(pyparamsIn.model.x0(:)), ...
         min(matparamsOut.model.x0(:)), ...
         min(pyparamsOut.model.x0(:))]);
M = max([max(pyparamsIn.model.x0(:)), ...
         max(matparamsOut.model.x0(:)), ...
         max(pyparamsOut.model.x0(:))]);subplot(2,4,1)
imagesc(pyparamsIn.model.x0')
caxis([m,M])
set(gca,'TickDir', 'out')
title('\mu_0 (initialization)')
subplot(2,4,2)
imagesc(matparamsOut.model.x0)
caxis([m,M])
set(gca,'TickDir', 'out')
title('\mu_0 (Matlab)')
subplot(2,4,3)
imagesc(pyparamsOut.model.x0')
caxis([m,M])
set(gca,'TickDir', 'out')
title('\mu_0 (python)')
subplot(2,4,4)
M = max([max(matparamsOut.model.x0(:)), max(pyparamsOut.model.x0(:))]);
m = max([min(matparamsOut.model.x0(:)), min(pyparamsOut.model.x0(:))]);
plot(-1000,-1000, '.', 'color', clrs(end,:), 'markerSize', 10)
hold on
plot(-1000,-1000, 'o', 'color', clrs(1,:))
line([1.1*m,1.1*M], ...
     [1.1*m,1.1*M],'color','c')
axis(1.1*[m,M+0.001,m,M+0.001])
plot(pyparamsOut.model.x0(:), ...
     matparamsOut.model.x0(:), '.', 'color', clrs(end,:))
set(gca,'TickDir', 'out')
box off
xlabel('python')
ylabel('Matlab')
legend('\mu_{0,i}', 'location', 'NorthWest')
legend boxoff
MSE_mu0 = mean( (matparamsOut.model.x0(:) - pyparamsOut.model.x0(:)).^2 );
text(0.3*M, 0.5*m+0.1*(M-m), ['MSE: ', num2str(MSE_mu0)])
corr_mu0 = corr(matparamsOut.model.x0(:), pyparamsOut.model.x0(:));
text(0.3*M, 0.5*m+0.2*(M-m), ['corr: ', num2str(corr_mu0)])

m = min([min(pyparamsIn.model.Q0(:)), ...
         min(matparamsOut.model.Q0(:)), ...
         min(pyparamsOut.model.Q0(:))]);
M = max([max(pyparamsIn.model.Q0(:)), ...
         max(matparamsOut.model.Q0(:)), ...
         max(pyparamsOut.model.Q0(:))]);subplot(2,4,1)
subplot(2,4,5)
imagesc(pyparamsIn.model.Q0)
caxis([m,M])
set(gca,'TickDir', 'out')
title('V_0 (initialization)')
subplot(2,4,6)
imagesc(matparamsOut.model.Q0)
caxis([m,M])
set(gca,'TickDir', 'out')
title('V_0 (Matlab)')
subplot(2,4,7)
imagesc(pyparamsOut.model.Q0)
caxis([m,M])
set(gca,'TickDir', 'out')
title('V_0 (python)')
subplot(2,4,8)
M = max([max(matparamsOut.model.Q0(:)), max(pyparamsOut.model.Q0(:))]);
m = max([min(matparamsOut.model.Q0(:)), min(pyparamsOut.model.Q0(:))]);
plot(-1000,-1000, '.', 'color', clrs(end,:), 'markerSize', 10)
hold on
plot(-1000,-1000, 'o', 'color', clrs(1,:))
line([1.1*m,1.1*M], ...
     [1.1*m,1.1*M],'color','c')
axis(1.1*[m,M+0.001,m,M+0.001])
plot(pyparamsOut.model.Q0(:), ...
     matparamsOut.model.Q0(:), '.', 'color', clrs(end,:))
plot(eig(pyparamsOut.model.Q0), ...
     eig(matparamsOut.model.Q0), 'o', 'color', clrs(1,:))
set(gca,'TickDir', 'out')
box off
xlabel('python')
ylabel('Matlab')
legend('V_{0,ij}', '[eig(V_0)]_i', 'location', 'NorthWest')
legend boxoff
MSE_V0 = mean( (matparamsOut.model.Q0(:) - pyparamsOut.model.Q0(:)).^2 );
MSE_V0eig=mean((eig(matparamsOut.model.Q0)-eig(pyparamsOut.model.Q0)).^2);
text(0.3*M, 0.5*m, ['MSE (eig): ', num2str(MSE_V0eig)])
text(0.3*M, 0.5*m+0.1*(M-m), ['MSE: ', num2str(MSE_V0)])
corr_V0 = corr(matparamsOut.model.Q0(:), pyparamsOut.model.Q0(:));
text(0.3*M, 0.5*m+0.2*(M-m), ['corr: ', num2str(corr_V0)])

%% checking C, d, R
%--------------------------------------------------------------------------
figure('Units', 'normalized','Position', [0.3,0.3,0.6,0.5]);  
clrs = copper(yDim);
m = min([min(pyparamsIn.model.C(:)), ...
         min(matparamsOut.model.C(:)), ...
         min(pyparamsOut.model.C(:))]);
M = max([max(pyparamsIn.model.C(:)), ...
         max(matparamsOut.model.C(:)), ...
         max(pyparamsOut.model.C(:))]);
subplot(3,4,1)
imagesc(squeeze(pyparamsIn.model.C))
caxis([m,M])
set(gca,'TickDir', 'out')
title('C (initialization)')
subplot(3,4,2)
imagesc(squeeze(matparamsOut.model.C))
caxis([m,M])
set(gca,'TickDir', 'out')
title('C (Matlab)')
subplot(3,4,3)
imagesc(squeeze(pyparamsOut.model.C))
caxis([m,M])
set(gca,'TickDir', 'out')
title('C (python)')
subplot(3,4,4)
M = max([max(matparamsOut.model.C(:)), max(pyparamsOut.model.C(:))]);
m = max([min(matparamsOut.model.C(:)), min(pyparamsOut.model.C(:))]);
plot(-1000,-1000, '.', 'color', clrs(end,:), 'markerSize', 10)
hold on
plot(-1000,-1000, 'o', 'color', clrs(1,:))
line([1.1*m,1.1*M], ...
     [1.1*m,1.1*M],'color','c')
axis(1.1*[m,M+0.001,m,M+0.001])
plot(pyparamsOut.model.C(:), ...
     matparamsOut.model.C(:), '.', 'color', clrs(end,:))
set(gca,'TickDir', 'out')
box off
xlabel('python')
ylabel('Matlab')
legend('C_{ij}', 'location', 'NorthWest')
legend boxoff
MSE_C = mean( (matparamsOut.model.C(:) - pyparamsOut.model.C(:)).^2 );
text(0.3*M, 0.5*m+0.1*(M-m), ['MSE: ', num2str(MSE_C)])
corr_C = corr(matparamsOut.model.C(:), pyparamsOut.model.C(:));
text(0.3*M, 0.5*m+0.2*(M-m), ['corr: ', num2str(corr_C)])


m = min([min(pyparamsIn.model.d(:)), ...
         min(matparamsOut.model.d(:)), ...
         min(pyparamsOut.model.d(:))]);
M = max([max(pyparamsIn.model.d(:)), ...
         max(matparamsOut.model.d(:)), ...
         max(pyparamsOut.model.d(:))]);
subplot(3,4,5)
imagesc(squeeze(pyparamsIn.model.d))
caxis([m,M])
set(gca,'TickDir', 'out')
title('d (initialization)')
subplot(3,4,6)
imagesc(squeeze(matparamsOut.model.d))
caxis([m,M])
set(gca,'TickDir', 'out')
title('d (Matlab)')
subplot(3,4,7)
imagesc(squeeze(pyparamsOut.model.d))
caxis([m,M])
set(gca,'TickDir', 'out')
title('d (python)')
subplot(3,4,8)
M = max([max(matparamsOut.model.d(:)), max(pyparamsOut.model.d(:))]);
m = max([min(matparamsOut.model.d(:)), min(pyparamsOut.model.d(:))]);
plot(-1000,-1000, '.', 'color', clrs(end,:), 'markerSize', 10)
hold on
plot(-1000,-1000, 'o', 'color', clrs(1,:))
line([1.1*m,1.1*M], ...
     [1.1*m,1.1*M],'color','c')
axis(1.1*[m,M+0.001,m,M+0.001])
plot(pyparamsOut.model.d(:), ...
     matparamsOut.model.d(:), '.', 'color', clrs(end,:))
set(gca,'TickDir', 'out')
box off
xlabel('python')
ylabel('Matlab')
legend('d_{ij}', 'location', 'NorthWest')
legend boxoff
MSE_d = mean( (matparamsOut.model.d(:) - pyparamsOut.model.d(:)).^2 );
text(0.3*M, 0.5*m+0.1*(M-m), ['MSE: ', num2str(MSE_d)])
corr_d = corr(matparamsOut.model.d(:), pyparamsOut.model.d(:));
text(0.3*M, 0.5*m+0.2*(M-m), ['corr: ', num2str(corr_d)])

m = min([min(pyparamsIn.model.R(:)), ...
         min(matparamsOut.model.R(:)), ...
         min(pyparamsOut.model.R(:))]);
M = max([max(pyparamsIn.model.R(:)), ...
         max(matparamsOut.model.R(:)), ...
         max(pyparamsOut.model.R(:))]);     
subplot(3,4,9)
imagesc(squeeze(pyparamsIn.model.R))
caxis([m,M])
set(gca,'TickDir', 'out')
title('R (initialization)')
subplot(3,4,10)
imagesc(squeeze(matparamsOut.model.R))
caxis([m,M])
set(gca,'TickDir', 'out')
title('R (Matlab)')
subplot(3,4,11)
if min(size(pyparamsOut.model.R))==1
    imagesc(diag(squeeze(pyparamsOut.model.R)))
else   
    imagesc(squeeze(pyparamsOut.model.R))
end
caxis([m,M])
set(gca,'TickDir', 'out')
title('R (python)')
subplot(3,4,12)
M = max([max(matparamsOut.model.R(:)), max(pyparamsOut.model.R(:))]);
m = max([min(matparamsOut.model.R(:)), min(pyparamsOut.model.R(:))]);
plot(-1000,-1000, '.', 'color', clrs(end,:), 'markerSize', 10)
hold on
plot(-1000,-1000, 'o', 'color', clrs(1,:))
line([1.1*m,1.1*M], ...
     [1.1*m,1.1*M],'color','c')
axis(1.1*[m,M+0.001,m,M+0.001])
if min(size(pyparamsOut.model.R))==1
  plot(pyparamsOut.model.R(:), ...
       diag(matparamsOut.model.R), '.', 'color', clrs(end,:))    
else
  plot(pyparamsOut.model.R(:), ...
       matparamsOut.model.R(:), '.', 'color', clrs(end,:))
  plot(eig(squeeze(pyparamsOut.model.R)), ...
       eig(squeeze(matparamsOut.model.R)), 'o', 'color', clrs(1,:))
end
set(gca,'TickDir', 'out')
box off
xlabel('python')
ylabel('Matlab')
legend('R_{ij}', '[eig(R)]_i', 'location', 'NorthWest')
legend boxoff
if min(size(pyparamsOut.model.R))==1
  MSE_R = mean( (diag(matparamsOut.model.R) - pyparamsOut.model.R(:)).^2 );
  MSE_Reig = MSE_R;
  corr_R = corr(diag(matparamsOut.model.R), pyparamsOut.model.R(:));

else
  MSE_R = mean( (matparamsOut.model.R(:) - pyparamsOut.model.R(:)).^2 );
  MSE_Reig = mean( (eig(squeeze(matparamsOut.model.R))-...
                    eig(squeeze(pyparamsOut.model.R))).^2 );
  corr_R = corr(matparamsOut.model.R(:), pyparamsOut.model.R(:));
                
end
text(0.3*M, 0.5*m, ['MSE (eig): ', num2str(MSE_Reig)])
text(0.3*M, 0.5*m+0.1*(M-m), ['MSE: ', num2str(MSE_R)])
text(0.3*M, 0.5*m+0.2*(M-m), ['corr: ', num2str(corr_R)])



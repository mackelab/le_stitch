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

stats_idx = 'h'; 

%% load, translate, compute

addpath( ...
   genpath('/home/mackelab/Desktop/Projects/Stitching/code/pop_spike_dyn'))
save_file = '/home/mackelab/Desktop/Projects/Stitching/results/LDS_save';

load([save_file, '_mattjj'])

switch stats_idx
    case '0'
        stats_i = stats_0;
    case '1'
        stats_i = stats_1;
    case 'h'
        stats_i = stats_h;
    case 'true'
        stats_i = stats_true;
end

% squeezing parameters into Lars' formatting
trueparams.model.A  = truePars.A;
trueparams.model.B  = truePars.B;
trueparams.model.Q  = truePars.Q;
trueparams.model.x0 = truePars.mu0;
trueparams.model.Q0 = truePars.V0;
trueparams.model.C  = truePars.C;
trueparams.model.d  = truePars.d(:);
if min(size(truePars.R)) == 1
 trueparams.model.R  = diag(truePars.R);    
else
 trueparams.model.R  = truePars.R;
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

pySeq_mattjj.x = x';
pySeq_mattjj.y = y';
pySeq_mattjj.T = double(T);

% what the python E-step got as parameter estimate initializations
paramsIn.model.A = initPars.A;
paramsIn.model.B = initPars.B;
paramsIn.model.Q = initPars.Q;
paramsIn.model.x0 = initPars.mu0;
paramsIn.model.Q0 = initPars.V0;
paramsIn.model.C = initPars.C;
paramsIn.model.d = initPars.d(:);
if min(size(initPars.R)) == 1
 paramsIn.model.R  = diag(initPars.R);    
else
 paramsIn.model.R  = initPars.R;
end
paramsIn.model.Pi = 1;
paramsIn.model.notes = trueparams.model.notes;
paramsIn.model.inferenceHandle =  @LDSInference;
paramsIn.model.MStepHandle = @LDSMStep;
paramsIn.model.ParamPenalizerHandle = @LDSemptyParamPenalizerHandle;
paramsIn.opts = trueparams.opts;

% what the python M-step returned
pyparamsOut.model.A = firstPars.A;
pyparamsOut.model.B = firstPars.B;
pyparamsOut.model.Q = firstPars.Q;
pyparamsOut.model.x0 = firstPars.mu0;
pyparamsOut.model.Q0 = firstPars.V0;
pyparamsOut.model.C = firstPars.C;
if min(size(firstPars.R)) == 1
 pyparamsOut.model.R  = diag(firstPars.R);    
else
 pyparamsOut.model.R  = firstPars.R;
end
pyparamsOut.model.d  = firstPars.d(:);
pyparamsOut.model.Pi = 1;
pyparamsOut.model.notes = trueparams.model.notes;
pyparamsOut.model.inferenceHandle =  @LDSInference;
pyparamsOut.model.MStepHandle = @LDSMStep;
pyparamsOut.model.ParamPenalizerHandle = @LDSemptyParamPenalizerHandle;
pyparamsOut.opts = trueparams.opts;

xDim = size(trueparams.model.A,1);
yDim = size(trueparams.model.C,1);
             
pySeq_mattjj.posterior.xsm  = stats_i.mu_h';
pySeq_mattjj.posterior.Vsm  = zeros(xDim*T,    xDim);
pySeq_mattjj.posterior.VVsm = zeros(xDim*(T-1),xDim);

for i = 1:xDim
    for j = 1:xDim
      pySeq_mattjj.posterior.Vsm((0:xDim:end-1)+i, j)  = ...
       squeeze(stats_i.V_h(:,i,j));
      pySeq_mattjj.posterior.VVsm((0:xDim:end-1)+i, j) = ...
       squeeze(stats_i.extxtm1(1:end,i,j))' - ...
         (pySeq_mattjj.posterior.xsm(i,2:end).*pySeq_mattjj.posterior.xsm(j,1:end-1));   
    end
end

if size(stats_i.extxtm1,1) == 1
    pySeq_mattjj.posterior.VVsm = pySeq_mattjj.posterior.VVsm / pySeq_mattjj.T;
end

pySeq_mattjj.u = u;

trueparams_mattjj = trueparams;
paramsIn_mattjj =  paramsIn;
pyparamsOut_mattjj = pyparamsOut;

clearvars -except pySeq_mattjj trueparams_mattjj paramsIn_mattjj pyparamsOut_mattjj save_file stats_idx
%clearvars -except xDim yDim trueparams pyparamsOut paramsIn pySeq_mattjj pySeq Ext Extxt Extxtm1 T Pi Pi_h Pi_t Pi_t_h 

%%

load([save_file]) 

switch stats_idx
    case '0'
        stats_i = stats_0;
    case '1'
        stats_i = stats_1;
    case 'h'
        stats_i = stats_h;
    case 'true'
        stats_i = stats_true;
end

% squeezing parameters into Lars' formatting
trueparams.model.A  = truePars.A;
trueparams.model.B  = truePars.B;
trueparams.model.Q  = truePars.Q;
trueparams.model.x0 = truePars.mu0;
trueparams.model.Q0 = truePars.V0;
trueparams.model.C  = truePars.C;
trueparams.model.d  = truePars.d(:);
if min(size(truePars.R)) == 1
 trueparams.model.R  = diag(truePars.R);    
else
 trueparams.model.R  = truePars.R;
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

pySeq_mattjj.x = x;
pySeq_mattjj.y = y;
pySeq_mattjj.T = double(T);

% what the python E-step got as parameter estimate initializations
pyparamsIn.model.A = initPars.A;
pyparamsIn.model.B = initPars.B;
pyparamsIn.model.Q = initPars.Q;
pyparamsIn.model.x0 = initPars.mu0;
pyparamsIn.model.Q0 = initPars.V0;
pyparamsIn.model.C = initPars.C;
pyparamsIn.model.d = initPars.d(:);
if min(size(initPars.R)) == 1
 pyparamsIn.model.R  = diag(initPars.R);    
else
 pyparamsIn.model.R  = initPars.R;
end
pyparamsIn.model.Pi = 1;
pyparamsIn.model.notes = trueparams.model.notes;
pyparamsIn.model.inferenceHandle =  @LDSInference;
pyparamsIn.model.MStepHandle = @LDSMStep;
pyparamsIn.model.ParamPenalizerHandle = @LDSemptyParamPenalizerHandle;
pyparamsIn.opts = trueparams.opts;

% what the python M-step returned
pyparamsOut.model.A = firstPars.A;
pyparamsOut.model.B = firstPars.B;
pyparamsOut.model.Q = firstPars.Q;
pyparamsOut.model.x0 = firstPars.mu0;
pyparamsOut.model.Q0 = firstPars.V0;
pyparamsOut.model.C = firstPars.C;
if min(size(firstPars.R)) == 1
 pyparamsOut.model.R  = diag(firstPars.R);    
else
 pyparamsOut.model.R  = firstPars.R;
end
pyparamsOut.model.d  = firstPars.d(:);
pyparamsOut.model.Pi = 1;
pyparamsOut.model.notes = trueparams.model.notes;
pyparamsOut.model.inferenceHandle =  @LDSInference;
pyparamsOut.model.MStepHandle = @LDSMStep;
pyparamsOut.model.ParamPenalizerHandle = @LDSemptyParamPenalizerHandle;
pyparamsOut.opts = trueparams.opts;

pySeq = pySeq_mattjj; % translates between the moments that my python code works 
             % with and the means and covariances that Lars' code uses.
xDim = size(trueparams.model.A,1);
yDim = size(trueparams.model.C,1);
             
pySeq.posterior.xsm  = zeros(xDim,T);
pySeq.posterior.Vsm  = zeros(xDim*T,    xDim);
pySeq.posterior.VVsm = zeros(xDim*(T-1),xDim);

for i = 1:xDim
    pySeq.posterior.xsm(i,:) = stats_i.ext(i,:);
    for j = 1:xDim
        pySeq.posterior.Vsm((0:xDim:end-1)+i, j)  = ...
        squeeze(stats_i.extxt(i,j,:))'-(stats_i.ext(i,:).*stats_i.ext(j,:));
        pySeq.posterior.VVsm((0:xDim:end-1)+i, j) = ...
        squeeze(stats_i.extxtm1(i,j,2:end))'-(stats_i.ext(i,2:end).*stats_i.ext(j,1:end-1));        
    end
end

%% make comparison figures for E-step
% The E-step for standard LDS needs to correctly reproduce three 
% types of expected values for the ensuing M-step:
% E[x_t]
% E[x_t x_t']
% E[x_t x_{t-1}']
% We compare the python results with those stored in pySeq_mattjj.posterior from the 
% Matlab version in three consecutive figures:

figure('Units','normalized','Position',[0.05,0.1,0.5,0.9]); % check E[x_t]

clrs = copper(xDim);

for i = 1:xDim
    
%     subplot(xDim,2,1 + (i-1)*2)    % could also plot raw traces if 
%     plot(x(i,:), 'b')              % resuls seem really messed up ...
%     hold on
%     plot(Ext(i,:), 'r:')
%     plot(pySeq_mattjj.posterior.xsm(i,:), 'g--')
%     title(['inferred latent states, dim #', num2str(i)])
%     
    subplot(ceil(xDim/2),2,i)
    m = min(pySeq_mattjj.posterior.xsm(:));
    M = max(pySeq_mattjj.posterior.xsm(:));
    plot(-1000, -1000, '.', 'color', clrs(end,:), 'markerSize', 10) 
    hold on                                     % just to get 
    plot(-1000, -1000, 'o', 'color', clrs(1,:)) % the legend right...    
    line([1.1*m,1.1*M], ...
     [1.1*m,1.1*M],'color','c')
    axis(1.1*[m,M+0.001,m,M+0.001])
    plot(pySeq.posterior.xsm(i,:), pySeq_mattjj.posterior.xsm(i,:), '.', ...
                                   'color', clrs(end,:), 'markerSize', 10)
    plot(pySeq.posterior.Vsm((0:xDim:end-1)+i, i), ...
           pySeq_mattjj.posterior.Vsm((0:xDim:end-1)+i, i), ...
         'o', 'color', clrs(1,:))
    xlabel('pyRRHLDS')
    ylabel('pyLDS')
    title(['latent dim #', num2str(i)])
    if i == 1
        legend('E[X_i]', 'var(X_i)', 'location', 'NorthWest')
        legend boxoff
    end
    box off
    set(gca, 'TickDir', 'out')
    MSE_Ext = mean( (stats_i.ext(i,1:end)-pySeq_mattjj.posterior.xsm(i,1:end)).^2 );
    text(0.3*M, 0.6*m, ['MSE: ', num2str(MSE_Ext)])
end

%--------------------------------------- check E[x_t x_t'] ----------------
figure('Units', 'normalized','Position', [0.35,0.1,0.5,0.5]);

lgnd = cell(xDim,1);
m = min(pySeq_mattjj.posterior.Vsm(:));
M = max(pySeq_mattjj.posterior.Vsm(:));


for i = 1:xDim
    plot(-1000, -1000, 'o', 'color', clrs(end-(abs(1-i)),:))
    hold on
    lgnd{i} = ['|i-j| =', num2str(abs(1-i))];    
end
line([1.1*m,1.1*M], ...
     [1.1*m,1.1*M],'color','c')
axis(1.1*[m,M+0.001,m,M+0.001])
Mm = zeros(xDim,xDim,T);
Mp = zeros(xDim,xDim,T);
for i = 1:xDim
    for j = 1:xDim
        Mm(i,j,:) = pySeq.posterior.Vsm((0:xDim:end-1)+i, j);
        Mp(i,j,:) =   pySeq_mattjj.posterior.Vsm((0:xDim:end-1)+i, j);
        plot(squeeze(Mp(i,j,:)), ...
             squeeze(Mm(i,j,:)), 'o', 'color', clrs(end-(abs(i-j)),:))
    end
end
title(['cov[x_t, x_t^T]'])
xlabel('pyRRHLDS')
ylabel('pyLDS')
legend(lgnd, 'location','NorthWest')
legend boxoff
MSE_cov = mean( (Mp(:) - Mm(:)).^2 );
text(0.3*M, 0.6*m, ['MSE: ', num2str(MSE_cov)])
box off
set(gca, 'TickDir', 'out')

%------------------------------------------------ check E[x_t x_{t-1}'] ---
figure('Units', 'normalized','Position', [0.65,0.1,0.5,0.5]); 

clrs = copper(xDim);
m = min(pySeq_mattjj.posterior.VVsm(:));
M = max(pySeq_mattjj.posterior.VVsm(:));
for i = 1:xDim
    plot(-1000, -1000, 'o', 'color', clrs(end-(abs(1-i)),:))
    hold on
    lgnd{i} = ['|i-j| =', num2str(abs(1-i))];    
end
line([1.1*m,1.1*M], ...
     [1.1*m,1.1*M],'color','c')
axis(1.1*[m,M+0.001,m,M+0.001])
Mm = zeros(xDim,xDim,T-1);
Mp = zeros(xDim,xDim,T-1);
for i = 1:xDim
    for j = 1:xDim
        Mm(i,j,:) =   pySeq_mattjj.posterior.VVsm((0:xDim:end-1)+i, j);
        Mp(i,j,:) = pySeq.posterior.VVsm((0:xDim:end-1)+i, j);
        plot(squeeze(Mp(i,j,:)), ...
             squeeze(Mm(i,j,:)), 'o', 'color', clrs(end-(abs(i-j)),:))
        hold on
    end
end
legend(lgnd, 'location','NorthWest')
legend boxoff
title('cov[x_t, x_{t-1}^T]')
xlabel('pyRRHLDS')
ylabel('pyLDS')
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
         min(pyparamsOut_mattjj.model.A(:)), ...
         min(pyparamsOut.model.A(:))]);
M = max([max(pyparamsIn.model.A(:)), ...
         max(pyparamsOut_mattjj.model.A(:)), ...
         max(pyparamsOut.model.A(:))]);
subplot(2,4,1)
imagesc(pyparamsIn.model.A)
caxis([m,M])
set(gca,'TickDir', 'out')
title('A (initialization)')
subplot(2,4,2)
imagesc(pyparamsOut_mattjj.model.A)
caxis([m,M])
set(gca,'TickDir', 'out')
title('A (pyLDS)')
subplot(2,4,3)
imagesc(pyparamsOut.model.A)
caxis([m,M])
set(gca,'TickDir', 'out')
title('A (pyRRHLDS)')
subplot(2,4,4)
M = max([max(pyparamsOut_mattjj.model.A(:)), max(pyparamsOut.model.A(:))]);
m = min([min(pyparamsOut_mattjj.model.A(:)), min(pyparamsOut.model.A(:))]);
plot(-1000,-1000, '.', 'color', clrs(end,:), 'markerSize', 10)
hold on
plot(-1000,-1000, 'o', 'color', clrs(1,:))
line([1.1*m,1.1*M], ...
     [1.1*m,1.1*M],'color','c')
axis(1.1*[m,M+0.001,m,M+0.001])
plot(pyparamsOut.model.A(:), ...
     pyparamsOut_mattjj.model.A(:), '.', 'color', clrs(end,:))
plot(eig(pyparamsOut.model.A), ...
     eig(pyparamsOut_mattjj.model.A), 'o', 'color', clrs(1,:))
set(gca,'TickDir', 'out')
box off
xlabel('pyRRHLDS')
ylabel('pyLDS')
legend('A_{ij}', '[eig(A)]_i', 'location', 'NorthWest')
legend boxoff
MSE_A = mean( (pyparamsOut_mattjj.model.A(:) - pyparamsOut.model.A(:)).^2 );
MSE_Aeig = mean((eig(pyparamsOut_mattjj.model.A)-eig(pyparamsOut.model.A)).^2 );
text(0.3*M, 0.5*m, ['MSE (eig): ', num2str(MSE_Aeig)])
text(0.3*M, 0.5*m+0.1*(M-m), ['MSE: ', num2str(MSE_A)])
corr_A = corr(pyparamsOut_mattjj.model.A(:), pyparamsOut.model.A(:));
text(0.3*M, 0.5*m+0.2*(M-m), ['corr: ', num2str(corr_A)])

m = min([min(pyparamsIn.model.Q(:)), ...
         min(pyparamsOut_mattjj.model.Q(:)), ...
         min(pyparamsOut.model.Q(:))]);
M = max([max(pyparamsIn.model.Q(:)), ...
         max(pyparamsOut_mattjj.model.Q(:)), ...
         max(pyparamsOut.model.Q(:))]);
subplot(2,4,5)
imagesc(pyparamsIn.model.Q)
caxis([m,M])
set(gca,'TickDir', 'out')
title('Q (initialization)')
subplot(2,4,6)
imagesc(pyparamsOut_mattjj.model.Q)
caxis([m,M])
set(gca,'TickDir', 'out')
title('Q (pyLDS)')
subplot(2,4,7)
imagesc(pyparamsOut.model.Q)
caxis([m,M])
set(gca,'TickDir', 'out')
title('Q (pyRRHLDS)')
subplot(2,4,8)
M = max([max(pyparamsOut_mattjj.model.Q(:)), max(pyparamsOut.model.Q(:))]);
m = max([min(pyparamsOut_mattjj.model.Q(:)), min(pyparamsOut.model.Q(:))]);
plot(-1000,-1000, '.', 'color', clrs(end,:), 'markerSize', 10)
hold on
plot(-1000,-1000, 'o', 'color', clrs(1,:))
line([1.1*m,1.1*M], ...
     [1.1*m,1.1*M],'color','c')
axis(1.1*[m,M+0.001,m,M+0.001])
plot(pyparamsOut.model.Q(:), ...
     pyparamsOut_mattjj.model.Q(:), '.', 'color', clrs(end,:))
plot(eig(pyparamsOut.model.Q), ...
     eig(pyparamsOut_mattjj.model.Q), 'o', 'color', clrs(1,:))
set(gca,'TickDir', 'out')
box off
xlabel('pyRRHLDS')
ylabel('pyLDS')
legend('Q_{ij}', '[eig(Q)]_i', 'location', 'NorthWest')
legend boxoff
MSE_Q = mean( (pyparamsOut_mattjj.model.Q(:) - pyparamsOut.model.Q(:)).^2 );
MSE_Qeig = mean( (eig(pyparamsOut_mattjj.model.Q)-eig(pyparamsOut.model.Q)).^2 );
text(0.3*M, 0.5*m, ['MSE (eig): ', num2str(MSE_Qeig)])
text(0.3*M, 0.5*m+0.1*(M-m), ['MSE: ', num2str(MSE_Q)])
corr_Q = corr(pyparamsOut_mattjj.model.Q(:), pyparamsOut.model.Q(:));
text(0.3*M, 0.5*m+0.2*(M-m), ['corr: ', num2str(corr_Q)])

% checking mu0, v0
%--------------------------------------------------------------------------
figure('Units', 'normalized','Position', [0.25,0.25,0.6,0.5]);  
clrs = copper(xDim);
m = min([min(pyparamsIn.model.x0(:)), ...
         min(pyparamsOut_mattjj.model.x0(:)), ...
         min(pyparamsOut.model.x0(:))]);
M = max([max(pyparamsIn.model.x0(:)), ...
         max(pyparamsOut_mattjj.model.x0(:)), ...
         max(pyparamsOut.model.x0(:))]);subplot(2,4,1)
imagesc(pyparamsIn.model.x0')
caxis([m,M])
set(gca,'TickDir', 'out')
title('\mu_0 (initialization)')
subplot(2,4,2)
imagesc(pyparamsOut_mattjj.model.x0)
caxis([m,M])
set(gca,'TickDir', 'out')
title('\mu_0 (pyLDS)')
subplot(2,4,3)
imagesc(pyparamsOut.model.x0')
caxis([m,M])
set(gca,'TickDir', 'out')
title('\mu_0 (pyRRHLDS)')
subplot(2,4,4)
M = max([max(pyparamsOut_mattjj.model.x0(:)), max(pyparamsOut.model.x0(:))]);
m = max([min(pyparamsOut_mattjj.model.x0(:)), min(pyparamsOut.model.x0(:))]);
plot(-1000,-1000, '.', 'color', clrs(end,:), 'markerSize', 10)
hold on
plot(-1000,-1000, 'o', 'color', clrs(1,:))
line([1.1*m,1.1*M], ...
     [1.1*m,1.1*M],'color','c')
axis(1.1*[m,M+0.001,m,M+0.001])
plot(pyparamsOut.model.x0(:), ...
     pyparamsOut_mattjj.model.x0(:), '.', 'color', clrs(end,:))
set(gca,'TickDir', 'out')
box off
xlabel('pyRRHLDS')
ylabel('pyLDS')
legend('\mu_{0,i}', 'location', 'NorthWest')
legend boxoff
MSE_mu0 = mean( (pyparamsOut_mattjj.model.x0(:) - pyparamsOut.model.x0(:)).^2 );
text(0.3*M, 0.5*m+0.1*(M-m), ['MSE: ', num2str(MSE_mu0)])
corr_mu0 = corr(pyparamsOut_mattjj.model.x0(:), pyparamsOut.model.x0(:));
text(0.3*M, 0.5*m+0.2*(M-m), ['corr: ', num2str(corr_mu0)])

m = min([min(pyparamsIn.model.Q0(:)), ...
         min(pyparamsOut_mattjj.model.Q0(:)), ...
         min(pyparamsOut.model.Q0(:))]);
M = max([max(pyparamsIn.model.Q0(:)), ...
         max(pyparamsOut_mattjj.model.Q0(:)), ...
         max(pyparamsOut.model.Q0(:))]);subplot(2,4,1)
subplot(2,4,5)
imagesc(pyparamsIn.model.Q0)
caxis([m,M])
set(gca,'TickDir', 'out')
title('V_0 (initialization)')
subplot(2,4,6)
imagesc(pyparamsOut_mattjj.model.Q0)
caxis([m,M])
set(gca,'TickDir', 'out')
title('V_0 (pyLDS)')
subplot(2,4,7)
imagesc(pyparamsOut.model.Q0)
caxis([m,M])
set(gca,'TickDir', 'out')
title('V_0 (pyRRHLDS)')
subplot(2,4,8)
M = max([max(pyparamsOut_mattjj.model.Q0(:)), max(pyparamsOut.model.Q0(:))]);
m = max([min(pyparamsOut_mattjj.model.Q0(:)), min(pyparamsOut.model.Q0(:))]);
plot(-1000,-1000, '.', 'color', clrs(end,:), 'markerSize', 10)
hold on
plot(-1000,-1000, 'o', 'color', clrs(1,:))
line([1.1*m,1.1*M], ...
     [1.1*m,1.1*M],'color','c')
axis(1.1*[m,M+0.001,m,M+0.001])
plot(pyparamsOut.model.Q0(:), ...
     pyparamsOut_mattjj.model.Q0(:), '.', 'color', clrs(end,:))
plot(eig(pyparamsOut.model.Q0), ...
     eig(pyparamsOut_mattjj.model.Q0), 'o', 'color', clrs(1,:))
set(gca,'TickDir', 'out')
box off
xlabel('pyRRHLDS')
ylabel('pyLDS')
legend('V_{0,ij}', '[eig(V_0)]_i', 'location', 'NorthWest')
legend boxoff
MSE_V0 = mean( (pyparamsOut_mattjj.model.Q0(:) - pyparamsOut.model.Q0(:)).^2 );
MSE_V0eig=mean((eig(pyparamsOut_mattjj.model.Q0)-eig(pyparamsOut.model.Q0)).^2);
text(0.3*M, 0.5*m, ['MSE (eig): ', num2str(MSE_V0eig)])
text(0.3*M, 0.5*m+0.1*(M-m), ['MSE: ', num2str(MSE_V0)])
corr_V0 = corr(pyparamsOut_mattjj.model.Q0(:), pyparamsOut.model.Q0(:));
text(0.3*M, 0.5*m+0.2*(M-m), ['corr: ', num2str(corr_V0)])

%% checking C, d, R
%--------------------------------------------------------------------------
figure('Units', 'normalized','Position', [0.3,0.3,0.6,0.5]);  
clrs = copper(yDim);
m = min([min(pyparamsIn.model.C(:)), ...
         min(pyparamsOut_mattjj.model.C(:)), ...
         min(pyparamsOut.model.C(:))]);
M = max([max(pyparamsIn.model.C(:)), ...
         max(pyparamsOut_mattjj.model.C(:)), ...
         max(pyparamsOut.model.C(:))]);
subplot(3,4,1)
imagesc(squeeze(pyparamsIn.model.C))
caxis([m,M])
set(gca,'TickDir', 'out')
title('C (initialization)')
subplot(3,4,2)
imagesc(squeeze(pyparamsOut_mattjj.model.C))
caxis([m,M])
set(gca,'TickDir', 'out')
title('C (pyLDS)')
subplot(3,4,3)
imagesc(squeeze(pyparamsOut.model.C))
caxis([m,M])
set(gca,'TickDir', 'out')
title('C (pyRRHLDS)')
subplot(3,4,4)
M = max([max(pyparamsOut_mattjj.model.C(:)), max(pyparamsOut.model.C(:))]);
m = max([min(pyparamsOut_mattjj.model.C(:)), min(pyparamsOut.model.C(:))]);
plot(-1000,-1000, '.', 'color', clrs(end,:), 'markerSize', 10)
hold on
plot(-1000,-1000, 'o', 'color', clrs(1,:))
line([1.1*m,1.1*M], ...
     [1.1*m,1.1*M],'color','c')
axis(1.1*[m,M+0.001,m,M+0.001])
plot(pyparamsOut.model.C(:), ...
     pyparamsOut_mattjj.model.C(:), '.', 'color', clrs(end,:))
set(gca,'TickDir', 'out')
box off
xlabel('pyRRHLDS')
ylabel('pyLDS')
legend('C_{ij}', 'location', 'NorthWest')
legend boxoff
MSE_C = mean( (pyparamsOut_mattjj.model.C(:) - pyparamsOut.model.C(:)).^2 );
text(0.3*M, 0.5*m+0.1*(M-m), ['MSE: ', num2str(MSE_C)])
corr_C = corr(pyparamsOut_mattjj.model.C(:), pyparamsOut.model.C(:));
text(0.3*M, 0.5*m+0.2*(M-m), ['corr: ', num2str(corr_C)])


m = min([min(pyparamsIn.model.d(:)), ...
         min(pyparamsOut_mattjj.model.d(:)), ...
         min(pyparamsOut.model.d(:))]);
M = max([max(pyparamsIn.model.d(:)), ...
         max(pyparamsOut_mattjj.model.d(:)), ...
         max(pyparamsOut.model.d(:))]);
subplot(3,4,5)
imagesc(squeeze(pyparamsIn.model.d))
caxis([m,M])
set(gca,'TickDir', 'out')
title('d (initialization)')
subplot(3,4,6)
imagesc(squeeze(pyparamsOut_mattjj.model.d))
caxis([m,M])
set(gca,'TickDir', 'out')
title('d (pyLDS)')
subplot(3,4,7)
imagesc(squeeze(pyparamsOut.model.d))
caxis([m,M])
set(gca,'TickDir', 'out')
title('d (pyRRHLDS)')
subplot(3,4,8)
M = max([max(pyparamsOut_mattjj.model.d(:)), max(pyparamsOut.model.d(:))]);
m = max([min(pyparamsOut_mattjj.model.d(:)), min(pyparamsOut.model.d(:))]);
plot(-1000,-1000, '.', 'color', clrs(end,:), 'markerSize', 10)
hold on
plot(-1000,-1000, 'o', 'color', clrs(1,:))
line([1.1*m,1.1*M], ...
     [1.1*m,1.1*M],'color','c')
axis(1.1*[m,M+0.001,m,M+0.001])
plot(pyparamsOut.model.d(:), ...
     pyparamsOut_mattjj.model.d(:), '.', 'color', clrs(end,:))
set(gca,'TickDir', 'out')
box off
xlabel('pyRRHLDS')
ylabel('pyLDS')
legend('d_{ij}', 'location', 'NorthWest')
legend boxoff
MSE_d = mean( (pyparamsOut_mattjj.model.d(:) - pyparamsOut.model.d(:)).^2 );
text(0.3*M, 0.5*m+0.1*(M-m), ['MSE: ', num2str(MSE_d)])
corr_d = corr(pyparamsOut_mattjj.model.d(:), pyparamsOut.model.d(:));
text(0.3*M, 0.5*m+0.2*(M-m), ['corr: ', num2str(corr_d)])

m = min([min(pyparamsIn.model.R(:)), ...
         min(pyparamsOut_mattjj.model.R(:)), ...
         min(pyparamsOut.model.R(:))]);
M = max([max(pyparamsIn.model.R(:)), ...
         max(pyparamsOut_mattjj.model.R(:)), ...
         max(pyparamsOut.model.R(:))]);     
subplot(3,4,9)
imagesc(squeeze(pyparamsIn.model.R))
caxis([m,M])
set(gca,'TickDir', 'out')
title('R (initialization)')
subplot(3,4,10)
imagesc(squeeze(pyparamsOut_mattjj.model.R))
caxis([m,M])
set(gca,'TickDir', 'out')
title('R (pyLDS)')
subplot(3,4,11)
if min(size(pyparamsOut.model.R))==1
    imagesc(diag(squeeze(pyparamsOut.model.R)))
else   
    imagesc(squeeze(pyparamsOut.model.R))
end
caxis([m,M])
set(gca,'TickDir', 'out')
title('R (pyRRHLDS)')
subplot(3,4,12)
M = max([max(pyparamsOut_mattjj.model.R(:)), max(pyparamsOut.model.R(:))]);
m = max([min(pyparamsOut_mattjj.model.R(:)), min(pyparamsOut.model.R(:))]);
plot(-1000,-1000, '.', 'color', clrs(end,:), 'markerSize', 10)
hold on
plot(-1000,-1000, 'o', 'color', clrs(1,:))
line([1.1*m,1.1*M], ...
     [1.1*m,1.1*M],'color','c')
axis(1.1*[m,M+0.001,m,M+0.001])
if min(size(pyparamsOut.model.R))==1
  plot(pyparamsOut.model.R(:), ...
       diag(pyparamsOut_mattjj.model.R), '.', 'color', clrs(end,:))    
else
  plot(pyparamsOut.model.R(:), ...
       pyparamsOut_mattjj.model.R(:), '.', 'color', clrs(end,:))
  plot(eig(squeeze(pyparamsOut.model.R)), ...
       eig(squeeze(pyparamsOut_mattjj.model.R)), 'o', 'color', clrs(1,:))
end
set(gca,'TickDir', 'out')
box off
xlabel('pyRRHLDS')
ylabel('pyLDS')
legend('R_{ij}', '[eig(R)]_i', 'location', 'NorthWest')
legend boxoff
if min(size(pyparamsOut.model.R))==1
  MSE_R = mean( (diag(pyparamsOut_mattjj.model.R) - pyparamsOut.model.R(:)).^2 );
  MSE_Reig = MSE_R;
  corr_R = corr(diag(pyparamsOut_mattjj.model.R), pyparamsOut.model.R(:));

else
  MSE_R = mean( (pyparamsOut_mattjj.model.R(:) - pyparamsOut.model.R(:)).^2 );
  MSE_Reig = mean( (eig(squeeze(pyparamsOut_mattjj.model.R))-...
                    eig(squeeze(pyparamsOut.model.R))).^2 );
  corr_R = corr(pyparamsOut_mattjj.model.R(:), pyparamsOut.model.R(:));
                
end
text(0.3*M, 0.5*m, ['MSE (eig): ', num2str(MSE_Reig)])
text(0.3*M, 0.5*m+0.1*(M-m), ['MSE: ', num2str(MSE_R)])
text(0.3*M, 0.5*m+0.2*(M-m), ['corr: ', num2str(corr_R)])

%%

if min(size(truePars.R)) == 1
 truePars.R = diag(truePars.R);    
end
if min(size(estPars.R)) == 1
 estPars.R = diag(estPars.R);    
end

figure;
covyyt = cov([pySeq_mattjj.y(:,2:end);pySeq_mattjj.y(:,1:end-1)]');
covy_true = truePars.C * Pi * truePars.C' + truePars.R;
covy_est = estPars.C * Pi_h * estPars.C' + estPars.R;
covy_emp = covyyt(1:yDim, 1:yDim);

m = min([covy_true(:); covy_est(:); covy_emp(:)]);
M = max([covy_true(:); covy_est(:); covy_emp(:)]);
subplot(2,3,1)
imagesc(covy_true)
set(gca, 'clim', [m,M])
title('$C \Pi C^T + R$', 'Interpreter', 'latex')
subplot(2,3,2)
imagesc(covy_est)
set(gca, 'clim', [m,M])
title('$\hat{C} \hat{\Pi} \hat{C}^T + \hat{R}$', 'Interpreter', 'latex')
subplot(2,3,3)
imagesc(covy_emp)
set(gca, 'clim', [m,M])
title('$cov(y_t,y_t)$', 'Interpreter', 'latex')

covy_t_true = truePars.C * (Pi_t) * truePars.C';
covy_t_est = estPars.C * (Pi_t_h) * estPars.C';
covy_t_emp = covyyt(1:yDim, yDim+(1:yDim));

m = min([covy_true(:); covy_est(:); covy_t_emp(:)]);
M = max([covy_true(:); covy_est(:); covy_t_emp(:)]);
subplot(2,3,4)
imagesc(covy_t_true)
set(gca, 'clim', [m,M])
title('$C A \Pi C^T$', 'Interpreter', 'latex')
subplot(2,3,5)
imagesc(covy_t_est)
set(gca, 'clim', [m,M])
title('$\hat{C} \hat{A} \hat{\Pi} \hat{C}$', 'Interpreter', 'latex')
subplot(2,3,6)
imagesc(covy_t_emp)
set(gca, 'clim', [m,M])
title('$cov(y_{t-1}, y_{t})$', 'Interpreter', 'latex')


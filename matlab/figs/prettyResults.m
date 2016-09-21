%% Visualises results of the EM stitching implementation
% Stitching in space and time.
% Slightly more pretty than the standard version. 

clear all
close all
clc

%% load, translate, compute

addpath( ...
   genpath('/home/mackelab/Desktop/Projects/Stitching/code/pop_spike_dyn'))

%dataSet = 'LDS_save_p200_good_fit_bad_stitch';
%dataSet  = 'LDS_save_9x3_trulyNoDynamics';
dataSet = 'LDS_save_9x3_withDynamicsLowOverlap';
%dataSet = 'LDS_save_9x3_withDynamicsLowOverlap';
load(['/home/mackelab/Desktop/Projects/Stitching/results/test_problems',...
      '/', dataSet, '.mat'])
  T    = size(y,2); % simulation length
  Trial= size(y,3); % number of trials
  xDim = size(A_h,1); % q
  yDim = size(y,1);   % p 
  uDim = size(u,1);   % r
    
%% Visualise observation protocol

if ~ iscell(obsScheme.subpops)
  numPops = size(obsScheme.subpops,1);
  subpops = cell(numPops,1);
  for i = 1:numPops
      subpops{i} = double(obsScheme.subpops(i,:))+1;
  end
else
  numPops = length(obsScheme.subpops);
  subpops = obsScheme.subpops;
  for i = 1:numPops
      subpops{i} = double(subpops{i})+1;
  end  
  numPops = length(subpops);
end

idxStitched = true(yDim,yDim); % (i,j) for each pair of variables ...  
for k = 1:numPops
    idxStitched(subpops{k},subpops{k}) = 0;
end

protocol = false(yDim, T);
idx = subpops{obsScheme.obsPops(1)+1};
protocol(idx,1:obsScheme.obsTime(1)) = true;
for i = 2:length(obsScheme.obsTime)
    idx = subpops{obsScheme.obsPops(i)+1};
    protocol(idx,obsScheme.obsTime(i-1)+1:obsScheme.obsTime(i)) = true;
end

figure('Units', 'normalized','Position', [0.1,0.3,0.8,0.6]);
imagesc(protocol)
title('stitching protocol (probably too dense to properly visualize')
xlabel('time')
ylabel('neurons')

%% Compare parameters with ground truth
groundTruthKnown = 1;
if groundTruthKnown
    figure('Units', 'normalized','Position', [0.5,0.5,0.4,0.5]);
    subplot(2,2,1)
    [eigA_hs, idxS] = sort(real(eig(A_h)));
    if xDim > 1
        plot(1:xDim, sort(real(eig(A))), 'g', 'linewidth', 2)
        hold on
        plot(1:xDim, eigA_hs, 'b', 'linewidth', 2)
        xlabel('(sorted) eigenvalue')
    else
        plot(1:xDim, sort(eig(A)), 'go', 'linewidth', 2, 'markerSize', 8)
        hold on
        plot(1:xDim, eigA_hs, 'bo', 'linewidth', 2, 'markerSize', 8)
        set(gca, 'XTick', [])
        xlabel('A_{11} (A is scalar)')
    end   
    box off
    axis([0.5,xDim+0.5,min(real([eig(A);eig(A_h)]))-0.1, ...
                       max(real([eig(A);eig(A_h)]))+0.1])  
    title('spectrum of A')
    legend('true A', 'est. A', 'Location', 'Northwest')
    legend boxoff
    set(gca, 'TickDir', 'out')

    subplot(2,2,2)
    if xDim > 1
        imagesc(C(:,idxS))
        title('C_{est}')
        
    else    
        bar(1:yDim, C, 0.5, 'b', 'faceColor', 'b', 'edgeColor', 'none')
        hold on
        if corr(C, C_h) < 0
            bar((1:yDim)-0.5, -C_h, 0.5, 'g', 'faceColor', 'g', 'edgeColor', 'none')
        else
            bar((1:yDim)-0.5, C_h, 0.5, 'g', 'faceColor', 'g', 'edgeColor', 'none')
        end
        bar(1:yDim, C/std(C)*std(C_h), 0.5, 'm', 'faceColor', 'm', 'edgeColor', 'none')
        legend('C true', 'C est.', 'C true resc.', 'Location', 'Northeast')
        legend boxoff
        box off
        xlabel('i')
        title('C')
        set(gca, 'TickDir', 'out')
        axis([0, yDim+1, 0, 1])
        axis autoy
    end

    subplot(2,2,3)
    bar(1:yDim, d, 0.5, 'b', 'faceColor', 'b', 'edgeColor', 'none')
    hold on
    if exist('d_h', 'var')
     bar((1:yDim)-0.5, d_h, 0.5, 'g', 'faceColor', 'g', 'edgeColor', 'none')
    end
    box off
    xlabel('i')
    title('d_i')
    set(gca, 'TickDir', 'out')
    axis([0, yDim+1, 0, 1])
    axis autoy

    subplot(2,2,4)
    bar(1:yDim, diag(R), 0.5, 'b', 'faceColor', 'b', 'edgeColor', 'none')
    hold on
    bar((1:yDim)-0.5, R_h, 0.5, 'g', 'faceColor', 'g', 'edgeColor', 'none')
    box off
    xlabel('i')
    ylabel('R_{ii}')
    title('diagonal of R')
    set(gca, 'TickDir', 'out')
    axis([0, yDim+1, 0, 1])
    axis autoy
end

%% Compare instantaneous covariances of model with truth
figure('Units', 'normalized','Position', [0.5,0.0,0.4,0.5]);

subplot(2,2,2)
LDScovy_h = C_h * (Pi_h) * C_h' + diag(R_h);
imagesc(LDScovy_h)
for ip = 1:numPops
    if unique(diff(sort(subpops{ip}))) == [1]
        mP = min(subpops{ip})-0.5;
        MP = max(subpops{ip})+0.5;
        line([mP,MP], [mP,mP], 'color', 'g', 'linewidth', 1.2)
        line([mP,MP], [MP,MP], 'color', 'g', 'linewidth', 1.2)
        line([mP,mP], [mP,MP], 'color', 'g', 'linewidth', 1.2)
        line([MP,MP], [mP,MP], 'color', 'g', 'linewidth', 1.2)
    end
end
set(gca, 'YTick', [1, yDim])
set(gca, 'XTick', [])
box off
ylabel('# neuron')
title('$\hat{C} \hat{\Pi} \hat{C}^T + \hat{R}$' ,'interpreter','latex')
colormap('gray')

covy = cov(y');
subplot(2,2,1)
imagesc(covy)
for ip = 1:numPops
    if unique(diff(sort(subpops{ip}))) == [1]
        mP = min(subpops{ip})-0.5;
        MP = max(subpops{ip})+0.5;
        line([mP,MP], [mP,mP], 'color', 'g', 'linewidth', 1.2)
        line([mP,MP], [MP,MP], 'color', 'g', 'linewidth', 1.2)
        line([mP,mP], [mP,MP], 'color', 'g', 'linewidth', 1.2)
        line([MP,MP], [mP,MP], 'color', 'g', 'linewidth', 1.2)
    end
end
set(gca, 'XTick', [])
set(gca, 'YTick', [1, yDim])
box off
title(['Instantaneous covariances cov(y_n, y_n)'])
ylabel('# neuron')


subplot(2,2,4)
mc = min([vec(covy(idxStitched));LDScovy_h(idxStitched)]);
Mc = max([vec(covy(idxStitched));LDScovy_h(idxStitched)]);
mc = min([mc;vec(covy(~idxStitched));LDScovy_h(~idxStitched)]);
Mc = max([Mc;vec(covy(~idxStitched));LDScovy_h(~idxStitched)]);
hold off
line(1.1*[mc, Mc], 1.1*[mc, Mc], 'color', 'k', 'linewidth', 1.5)
hold on
plot(covy(idxStitched), LDScovy_h(idxStitched), 'r.', 'markerSize',15)
hold on
if sum(idxStitched(:)>0)
    text(mc,0.9*Mc, ...
        ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
        num2str(corr(covy(idxStitched),LDScovy_h(idxStitched)))])
end
line(1.1*[mc, Mc], 1.1*[mc, Mc], 'color', 'k', 'linewidth', 1.5)
plot(covy(~idxStitched), LDScovy_h(~idxStitched), 'g.', 'markerSize',15)
text(mc,0.7*Mc, ...
    ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
    num2str(corr(covy(~idxStitched),LDScovy_h(~idxStitched)))])
axis(1.1*[mc, Mc, mc, Mc])
xlabel('true cov(y)')
ylabel('est. cov(y)')
box off



%% Compare time-lag covariances of model with truth
figure('Units', 'normalized','Position', [0.1,0.0,0.4,0.5]);
if numel(u)>1 && stcmp(inputType,'pwdconst')
    numPieces = double(floor(T/constantInputLength));
    m = min(u(:));
    M = max(u(:));
    spwidth = 17;
    sphight = 17;    
    subplot(sphight,spwidth,vec(bsxfun(@plus,(14:17)',(6:10)*spwidth)))
    LDScovy_h = C_h * (A_h * Pi_h) * C_h';
    imagesc(LDScovy_h)
    for ip = 1:numPops
        if unique(diff(sort(subpops{ip}))) == [1]
            mP = min(subpops{ip})-0.5;
            MP = max(subpops{ip})+0.5;
            line([mP,MP], [mP,mP], 'color', 'k', 'linewidth', 1.2)
            line([mP,MP], [MP,MP], 'color', 'k', 'linewidth', 1.2)
            line([mP,mP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
            line([MP,MP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
        end
    end
    set(gca, 'YTick', [1, yDim])
    set(gca, 'XTick', [])
    box off
    ylabel('# neuron')
    title('$\hat{C} \hat{A} \hat{\Pi} \hat{C}^T$' ,'interpreter','latex')
    subplot(sphight,spwidth,vec(bsxfun(@plus,(14:17)',(12:16)*spwidth)))
    LDScovy = C * (A* Pi) * C';
    imagesc(LDScovy)
    for ip = 1:numPops
        if unique(diff(sort(subpops{ip}))) == [1]
            mP = min(subpops{ip})-0.5;
            MP = max(subpops{ip})+0.5;
            line([mP,MP], [mP,mP], 'color', 'k', 'linewidth', 1.2)
            line([mP,MP], [MP,MP], 'color', 'k', 'linewidth', 1.2)
            line([mP,mP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
            line([MP,MP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
        end
    end
    set(gca, 'XTick', [1, yDim])
    set(gca, 'YTick', [1, yDim])
    title('C A \Pi C^T') 
    ylabel('# neuron')
    box off
    for i = 1:numPieces
        covy = cov([y(:, (i-1)*constantInputLength + ...
                     (round(constantInputLength/2):constantInputLength));
                    y(:, (i-1)*constantInputLength -1 +  ...
                     (round(constantInputLength/2):constantInputLength))]');
        covy = covy(yDim+1:end, 1:yDim); % take lover left corner                 
        subplot(sphight,spwidth,vec(bsxfun(@plus,(14:17)',(0:4)*spwidth)))
        imagesc(covy)
        for ip = 1:numPops
            if unique(diff(sort(subpops{ip}))) == [1]
                mP = min(subpops{ip})-0.5;
                MP = max(subpops{ip})+0.5;
                line([mP,MP], [mP,mP], 'color', 'k', 'linewidth', 1.2)
                line([mP,MP], [MP,MP], 'color', 'k', 'linewidth', 1.2)
                line([mP,mP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
                line([MP,MP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
            end
        end
        set(gca, 'XTick', [])
        set(gca, 'YTick', [1, yDim])
        box off
        title(['empirical cov(y)']) 
        ylabel('# neuron')
        subplot(sphight,spwidth,vec(bsxfun(@plus,(1:11)',(0:5)*spwidth)))
        hold off
        plot(u')
        hold on
        line([1;1]*(i+0.1)*constantInputLength,  ...
             1.1*[m;M], 'color', 'k', 'linewidth', 2) 
        line([1;1]*(i-1.1)*constantInputLength,  ...
             1.1*[m;M], 'color', 'k', 'linewidth', 2) 
        box off
        title(['Time-lagged covariances cov(y_n, y_{n-1})']) 
        axis([0, T+1, 1.2 * m, 1.2 * M])        
        set(gca, 'TickDir', 'out')
        xlabel('time n')
        ylabel('u[n]')
        subplot(sphight,spwidth,vec(bsxfun(@plus,(1:11)',(9:16)*spwidth)))
        hold off
        mc = min([vec(covy(idxStitched));LDScovy_h(idxStitched)]);
        Mc = max([vec(covy(idxStitched));LDScovy_h(idxStitched)]);
        mc = min([mc;vec(covy(~idxStitched));LDScovy_h(~idxStitched)]);
        Mc = max([Mc;vec(covy(~idxStitched));LDScovy_h(~idxStitched)]);
        hold off
        plot(0,0)
        line(1.1*[mc, Mc], 1.1*[mc, Mc], 'color', 'k', 'linewidth', 1.5)
        hold on
        plot(covy(idxStitched), LDScovy_h(idxStitched), 'b.', 'markerSize',7)
        text(mc,0.9*Mc, ...
             ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
              num2str(corr(covy(idxStitched),LDScovy_h(idxStitched)))])
        plot(covy(~idxStitched), LDScovy_h(~idxStitched), 'g.', 'markerSize',7)           
        text(mc,0.7*Mc, ...
             ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
              num2str(corr(covy(~idxStitched),LDScovy_h(~idxStitched)))])
        axis(1.1*[mc, Mc, mc, Mc])
        xlabel('true cov(y)')
        ylabel('est. cov(y)')
        box off
        
        
        pause
    end
else
    subplot(2,2,2)
    LDScovy_h = C_h * (A_h *Pi_h) * C_h';
    imagesc(LDScovy_h)
    for ip = 1:numPops
        if unique(diff(sort(subpops{ip}))) == [1]
            mP = min(subpops{ip})-0.5;
            MP = max(subpops{ip})+0.5;
            line([mP,MP], [mP,mP], 'color', 'k', 'linewidth', 1.2)
            line([mP,MP], [MP,MP], 'color', 'k', 'linewidth', 1.2)
            line([mP,mP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
            line([MP,MP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
        end
    end
    set(gca, 'YTick', [1, yDim])
    set(gca, 'XTick', [])
    box off
    ylabel('# neuron')
    title('$\hat{C}  \hat{A}  \hat{\Pi} \hat{C}^T$' ,'interpreter','latex')
    subplot(2,2,3)
    LDScovy = C * (A* Pi) * C';
    imagesc(LDScovy)
    for ip = 1:numPops
        if unique(diff(sort(subpops{ip}))) == [1]
            mP = min(subpops{ip})-0.5;
            MP = max(subpops{ip})+0.5;
            line([mP,MP], [mP,mP], 'color', 'k', 'linewidth', 1.2)
            line([mP,MP], [MP,MP], 'color', 'k', 'linewidth', 1.2)
            line([mP,mP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
            line([MP,MP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
        end
    end
    set(gca, 'XTick', [1, yDim])
    set(gca, 'YTick', [1, yDim])
    title('C A \Pi C^T') 
    ylabel('# neuron')
    box off
        
    covy = cov([y(:,2:end);y(:,1:end-1)]');
    covy = covy(yDim+1:end, 1:yDim);
    subplot(2,2,1)
    imagesc(covy)
    for ip = 1:numPops
        if unique(diff(sort(subpops{ip}))) == [1]
            mP = min(subpops{ip})-0.5;
            MP = max(subpops{ip})+0.5;
            line([mP,MP], [mP,mP], 'color', 'k', 'linewidth', 1.2)
            line([mP,MP], [MP,MP], 'color', 'k', 'linewidth', 1.2)
            line([mP,mP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
            line([MP,MP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
        end
    end
    set(gca, 'XTick', [])
    set(gca, 'YTick', [1, yDim])
    box off
    title(['Time-lagged covariances cov(y_n, y_{n-1})'])
    ylabel('# neuron')
    subplot(2,2,4)
    mc = min([vec(covy(idxStitched));LDScovy_h(idxStitched)]);
    Mc = max([vec(covy(idxStitched));LDScovy_h(idxStitched)]);
    mc = min([mc;vec(covy(~idxStitched));LDScovy_h(~idxStitched)]);
    Mc = max([Mc;vec(covy(~idxStitched));LDScovy_h(~idxStitched)]);
    hold off
    line(1.1*[mc, Mc], 1.1*[mc, Mc], 'color', 'k', 'linewidth', 1.5)
    hold on
    plot(covy(idxStitched), LDScovy_h(idxStitched), 'b.', 'markerSize',7)
    if sum(idxStitched(:))>0
      text(mc,0.9*Mc, ...
        ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
        num2str(corr(covy(idxStitched),LDScovy_h(idxStitched)))])
    end
    plot(covy(~idxStitched), LDScovy_h(~idxStitched), 'g.', 'markerSize',7)
    if sum(~idxStitched(:))>0
      text(mc,0.7*Mc, ...
        ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
        num2str(corr(covy(~idxStitched),LDScovy_h(~idxStitched)))])
    end
    axis(1.1*[mc, Mc, mc, Mc])
    xlabel('true cov(y)')
    ylabel('est. cov(y)')
    box off    
end

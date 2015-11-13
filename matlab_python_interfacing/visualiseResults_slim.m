%% Visualises results of the EM stitching implementation
% Stitching in space and time.

clear all
%close all
clc

%% load, translate, compute

addpath( ...
   genpath('/home/mackelab/Desktop/Projects/Stitching/code/pop_spike_dyn'))

%dataSet = '200x100_stitchingInSpace_slim';
%dataSet = 'stitchingInTime_SparseSparseSparse_slim';
dataSet = 'stitchingInTime_SparseFullFull_slim';

load(['/home/mackelab/Desktop/Projects/Stitching/results/test_problems',...
      '/', dataSet, '.mat'])
  
% gives the following variables:
% observed data traces              y
% input data traces                 u
% true sampled latent traces        z
% true parameters                   {A,  B,  Q,  mu0,  V0,  C,  d,  R  }
% initial parameters                {A_0,B_0,Q_0,mu0_0,V0_0,C_0,d_0,R_0}
% fitted parameters                 {A_h,B_h,Q_h,mu0_h,V0_h,C_h,d_h,R_h}
% derived statistics                 Pi, Pi_h, Pi_t, Pi_t_h
% E-step results, fitted params.    Ext_h, Extxt_h, Extxtm1_h
% E-step results, true params.      Ext,   Extxt,   Extxtm1
% description of input data         ifDataGeneratedWithInput,
%                                   ifInputPiecewiseConstant
%                                   constantInputLength

%constantInputLength = double(constantInputLength);

if max(abs([A(:);C(:)])) > 0
    groundTruthKnown = 1;
else
    groundTruthKnown = 0;    
    A  = double(A);
    C  = double(C);
    R  = double(R);
    Pi = double(Pi);    
    R  = double(d);
end

%  T    = size(y,2); % simulation length
%  Trial= size(y,3); % number of trials
  Trial = 1;
  xDim = size(A_h,1); % q
  yDim = size(C_h,1);   % p 
 % uDim = size(u,1);   % r
    
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

%for i = 1:numPops:
%    idxtPop = find(obsScheme.obsPops==i-1);
%    for j = 1:length(idxtPop):
%      protocol(subpops{i},obsScheme.obsTime(idxtPop(j)-1)+1:obsScheme.obsTime(idxtPop(j)))=1;
%    end
%end

% %% Display parameter fits
% figure('Units', 'normalized','Position', [0.0,0.5,0.4,0.5]);
% subplot(2,3,1)
% imagesc(A_h)
% box off
% title('$\hat{A}$', 'interpreter', 'latex')
% set(gca, 'XTick', unique([1, xDim]))
% set(gca, 'YTick', unique([1, xDim]))
% set(gca, 'TickDir', 'out')
% 
% subplot(2,3,2)
% if ifUseB
%     imagesc(B_h)
%     title('$\hat{B}$', 'interpreter', 'latex')
%     set(gca, 'XTick', unique([1, uDim]))
%     set(gca, 'YTick', unique([1, xDim]))
%     set(gca, 'TickDir', 'out')
%     box off    
% else
%     text(0,1,'Did not fit parameter B') 
%     axis([0,5,0,2])
%     axis off
%     box off
%     title('$\hat{B}$', 'interpreter', 'latex')
% end
% 
% subplot(2,3,3)
% imagesc(Q_h)
% title('$\hat{Q}$', 'interpreter', 'latex')
% set(gca, 'XTick', unique([1, xDim]))
% set(gca, 'YTick', unique([1, xDim]))
% set(gca, 'TickDir', 'out')
% box off
% 
% subplot(2,3,4)
% imagesc(C_h)
% title('$\hat{C}$', 'interpreter', 'latex')
% set(gca, 'XTick', unique([1, xDim]))
% set(gca, 'YTick', unique([1, yDim]))
% set(gca, 'TickDir', 'out')
% box off
% 
% subplot(2,3,5)
% bar(d_h)
% title('$\hat{d}$', 'interpreter', 'latex')
% set(gca, 'XTick', unique([1, yDim]))
% set(gca, 'TickDir', 'out')
% axis([0, yDim+1, 0, 1])
% axis autoy
% box off
% 
% subplot(2,3,6)
% imagesc(diag(R_h))
% title('$\hat{R}$', 'interpreter', 'latex')
% set(gca, 'XTick', unique([1, yDim]))
% set(gca, 'YTick', unique([1, yDim]))
% set(gca, 'TickDir', 'out')
% box off


%% Show distance from initialisation

figure('Units', 'normalized','Position', [0.0,0.5,0.55,0.4]);

m = min([min(A_0(:)),min(A_h(:))]);
M = max([max(A_0(:)),max(A_h(:))]);
subplot(2,4,1)
imagesc(A_0)
axis off
title('A_0')
subplot(2,4,5)
caxis([m,M])
imagesc(A_h)
axis off
title('$\hat{A}$', 'interpreter', 'latex')
caxis([m,M])

m = min([min(Q_0(:)),min(Q_h(:))]);
M = max([max(Q_0(:)),max(Q_h(:))]);
subplot(2,4,2)
imagesc(Q_0)
caxis([m,M])
axis off
title('Q_0')
subplot(2,4,6)
imagesc(Q_h)
caxis([m,M])
axis off
title('$\hat{Q}$', 'interpreter', 'latex')

m = min([min(C_0(:)),min(C_h(:))]);
M = max([max(C_0(:)),max(C_h(:))]);
subplot(2,4,3)
imagesc(C_0)
caxis([m,M])
axis off
title('C_0')
subplot(2,4,7)
imagesc(C_h)
caxis([m,M])
axis off
title('$\hat{C}$', 'interpreter', 'latex')

subplot(2,4,4)
plot(d_h, 'g', 'linewidth', 2)
hold on
plot(d_0, '--', 'linewidth', 2)
legend({'$\hat{d}$', '$d_0$'}, 'interpreter', 'latex')
box off
set(gca, 'TickDir', 'out')
set(gca, 'XTick', [0, yDim])
axis([0,yDim+1,0,1])
axis autoy
title('d')
subplot(2,4,8)
plot(R_h, 'g', 'linewidth', 2)
hold on
plot(R_0, '--', 'linewidth', 2)
legend({'diag $\hat{R}$', 'diag $R_0$'}, 'interpreter', 'latex')
box off
set(gca, 'TickDir', 'out')
set(gca, 'XTick', [0, yDim])
axis([0,yDim+1,0,1])
axis autoy
title('diag(R)')

%% Compare parameters with ground truth
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
% if numel(u)>1 && stcmp(inputType,'pwdconst')
%     numPieces = double(floor(T/constantInputLength));
%     m = min(u(:));
%     M = max(u(:));
%     spwidth = 17;
%     sphight = 17;    
%     subplot(sphight,spwidth,vec(bsxfun(@plus,(14:17)',(6:10)*spwidth)))
%     LDScovy_h = C_h * (Pi_h) * C_h' + diag(R_h);
%     imagesc(LDScovy_h)
%     for ip = 1:numPops
%         if unique(diff(sort(subpops{ip}))) == [1]
%             mP = min(subpops{ip})-0.5;
%             MP = max(subpops{ip})+0.5;
%             line([mP,MP], [mP,mP], 'color', 'k', 'linewidth', 1.2)
%             line([mP,MP], [MP,MP], 'color', 'k', 'linewidth', 1.2)
%             line([mP,mP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
%             line([MP,MP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
%         end
%     end
%     set(gca, 'YTick', [1, yDim])
%     set(gca, 'XTick', [])
%     box off
%     ylabel('# neuron')
%     title('$\hat{C} \hat{\Pi} \hat{C}^T + \hat{R}$' ,'interpreter','latex')
%     subplot(sphight,spwidth,vec(bsxfun(@plus,(14:17)',(12:16)*spwidth)))
%     LDScovy = C * (Pi) * C' + R;
%     imagesc(LDScovy)
%     for ip = 1:numPops
%         if unique(diff(sort(subpops{ip}))) == [1]
%             mP = min(subpops{ip})-0.5;
%             MP = max(subpops{ip})+0.5;
%             line([mP,MP], [mP,mP], 'color', 'k', 'linewidth', 1.2)
%             line([mP,MP], [MP,MP], 'color', 'k', 'linewidth', 1.2)
%             line([mP,mP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
%             line([MP,MP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
%         end
%     end
%     set(gca, 'XTick', [1, yDim])
%     set(gca, 'YTick', [1, yDim])
%     title('C \Pi C^T + R') 
%     ylabel('# neuron')
%     box off
%     for i = 1:numPieces
%         
%         covy = cov(y(:, (i-1)*constantInputLength + ...
%                      (round(constantInputLength/2):constantInputLength))');
%         subplot(sphight,spwidth,vec(bsxfun(@plus,(14:17)',(0:4)*spwidth)))
%         imagesc(covy)
%         for ip = 1:numPops
%             if unique(diff(sort(subpops{ip}))) == [1]
%                 mP = min(subpops{ip})-0.5;
%                 MP = max(subpops{ip})+0.5;
%                 line([mP,MP], [mP,mP], 'color', 'k', 'linewidth', 1.2)
%                 line([mP,MP], [MP,MP], 'color', 'k', 'linewidth', 1.2)
%                 line([mP,mP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
%                 line([MP,MP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
%             end
%         end
%         set(gca, 'XTick', [])
%         set(gca, 'YTick', [1, yDim])
%         box off
%         title(['empirical cov(y)']) 
%         ylabel('# neuron')
%         subplot(sphight,spwidth,vec(bsxfun(@plus,(1:11)',(0:5)*spwidth)))
%         hold off
%         plot(u')
%         hold on
%         line([1;1]*(i+0.1)*constantInputLength,  ...
%              1.1*[m;M], 'color', 'k', 'linewidth', 2) 
%         line([1;1]*(i-1.1)*constantInputLength,  ...
%              1.1*[m;M], 'color', 'k', 'linewidth', 2) 
%         box off
%         title(['Instantaneous covariances cov(y_n, y_n)']) 
%         axis([0, T+1, 1.2 * m, 1.2 * M])        
%         set(gca, 'TickDir', 'out')
%         xlabel('time n')
%         ylabel('u[n]')
%         subplot(sphight,spwidth,vec(bsxfun(@plus,(1:11)',(9:16)*spwidth)))
%         mc = min([vec(covy(idxStitched));LDScovy_h(idxStitched)]);
%         Mc = max([vec(covy(idxStitched));LDScovy_h(idxStitched)]);
%         mc = min([mc;vec(covy(~idxStitched));LDScovy_h(~idxStitched)]);
%         Mc = max([Mc;vec(covy(~idxStitched));LDScovy_h(~idxStitched)]);
%         hold off
%         plot(0,0)
%         line(1.1*[mc, Mc], 1.1*[mc, Mc], 'color', 'k', 'linewidth', 1.5)
%         hold on
%         plot(covy(idxStitched), LDScovy_h(idxStitched), 'b.', 'markerSize',7)
%         text(mc,0.9*Mc, ...
%              ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
%               num2str(corr(covy(idxStitched),LDScovy_h(idxStitched)))])
%         plot(covy(~idxStitched), LDScovy_h(~idxStitched), 'g.', 'markerSize',7)           
%         text(mc,0.7*Mc, ...
%              ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
%               num2str(corr(covy(~idxStitched),LDScovy_h(~idxStitched)))])
%         axis(1.1*[mc, Mc, mc, Mc])
%         xlabel('true cov(y)')
%         ylabel('est. cov(y)')
%         box off
%         
%         
%         pause
%     end
% else
    subplot(2,2,2)
    LDScovy_h = C_h * (Pi_h) * C_h' + diag(R_h);
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
    title('$\hat{C} \hat{\Pi} \hat{C}^T + \hat{R}$' ,'interpreter','latex')
    subplot(2,2,3)
    LDScovy = C * (Pi) * C' + R;
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
    title('C \Pi C^T + R') 
    ylabel('# neuron')
    box off
        
    %covy = cov(y');
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
    plot(covy(idxStitched), LDScovy_h(idxStitched), 'b.', 'markerSize',7)
    hold on
    if sum(idxStitched(:)>0)
     text(mc,0.9*Mc, ...
        ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
        num2str(corr(covy(idxStitched),LDScovy_h(idxStitched)))])
    end
    plot(covy(~idxStitched), LDScovy_h(~idxStitched), 'g.', 'markerSize',7)
    text(mc,0.7*Mc, ...
        ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
        num2str(corr(covy(~idxStitched),LDScovy_h(~idxStitched)))])
    line(1.1*[mc, Mc], 1.1*[mc, Mc], 'color', 'k', 'linewidth', 1.5)
    axis(1.1*[mc, Mc, mc, Mc])
    xlabel('true cov(y)')
    ylabel('est. cov(y)')
    box off
% end


%% Compare time-lag covariances of model with truth
figure('Units', 'normalized','Position', [0.1,0.0,0.4,0.5]);
% if numel(u)>1 && stcmp(inputType,'pwdconst')
%     numPieces = double(floor(T/constantInputLength));
%     m = min(u(:));
%     M = max(u(:));
%     spwidth = 17;
%     sphight = 17;    
%     subplot(sphight,spwidth,vec(bsxfun(@plus,(14:17)',(6:10)*spwidth)))
%     LDScovy_h = C_h * (A_h * Pi_h) * C_h';
%     imagesc(LDScovy_h)
%     for ip = 1:numPops
%         if unique(diff(sort(subpops{ip}))) == [1]
%             mP = min(subpops{ip})-0.5;
%             MP = max(subpops{ip})+0.5;
%             line([mP,MP], [mP,mP], 'color', 'k', 'linewidth', 1.2)
%             line([mP,MP], [MP,MP], 'color', 'k', 'linewidth', 1.2)
%             line([mP,mP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
%             line([MP,MP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
%         end
%     end
%     set(gca, 'YTick', [1, yDim])
%     set(gca, 'XTick', [])
%     box off
%     ylabel('# neuron')
%     title('$\hat{C} \hat{A} \hat{\Pi} \hat{C}^T$' ,'interpreter','latex')
%     subplot(sphight,spwidth,vec(bsxfun(@plus,(14:17)',(12:16)*spwidth)))
%     LDScovy = C * (A* Pi) * C';
%     imagesc(LDScovy)
%     for ip = 1:numPops
%         if unique(diff(sort(subpops{ip}))) == [1]
%             mP = min(subpops{ip})-0.5;
%             MP = max(subpops{ip})+0.5;
%             line([mP,MP], [mP,mP], 'color', 'k', 'linewidth', 1.2)
%             line([mP,MP], [MP,MP], 'color', 'k', 'linewidth', 1.2)
%             line([mP,mP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
%             line([MP,MP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
%         end
%     end
%     set(gca, 'XTick', [1, yDim])
%     set(gca, 'YTick', [1, yDim])
%     title('C A \Pi C^T') 
%     ylabel('# neuron')
%     box off
%     for i = 1:numPieces
%         covy = cov([y(:, (i-1)*constantInputLength + ...
%                      (round(constantInputLength/2):constantInputLength));
%                     y(:, (i-1)*constantInputLength -1 +  ...
%                      (round(constantInputLength/2):constantInputLength))]');
%         covy = covy(yDim+1:end, 1:yDim); % take lover left corner                 
%         subplot(sphight,spwidth,vec(bsxfun(@plus,(14:17)',(0:4)*spwidth)))
%         imagesc(covy)
%         for ip = 1:numPops
%             if unique(diff(sort(subpops{ip}))) == [1]
%                 mP = min(subpops{ip})-0.5;
%                 MP = max(subpops{ip})+0.5;
%                 line([mP,MP], [mP,mP], 'color', 'k', 'linewidth', 1.2)
%                 line([mP,MP], [MP,MP], 'color', 'k', 'linewidth', 1.2)
%                 line([mP,mP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
%                 line([MP,MP], [mP,MP], 'color', 'k', 'linewidth', 1.2)
%             end
%         end
%         set(gca, 'XTick', [])
%         set(gca, 'YTick', [1, yDim])
%         box off
%         title(['empirical cov(y)']) 
%         ylabel('# neuron')
%         subplot(sphight,spwidth,vec(bsxfun(@plus,(1:11)',(0:5)*spwidth)))
%         hold off
%         plot(u')
%         hold on
%         line([1;1]*(i+0.1)*constantInputLength,  ...
%              1.1*[m;M], 'color', 'k', 'linewidth', 2) 
%         line([1;1]*(i-1.1)*constantInputLength,  ...
%              1.1*[m;M], 'color', 'k', 'linewidth', 2) 
%         box off
%         title(['Time-lagged covariances cov(y_n, y_{n-1})']) 
%         axis([0, T+1, 1.2 * m, 1.2 * M])        
%         set(gca, 'TickDir', 'out')
%         xlabel('time n')
%         ylabel('u[n]')
%         subplot(sphight,spwidth,vec(bsxfun(@plus,(1:11)',(9:16)*spwidth)))
%         hold off
%         mc = min([vec(covy(idxStitched));LDScovy_h(idxStitched)]);
%         Mc = max([vec(covy(idxStitched));LDScovy_h(idxStitched)]);
%         mc = min([mc;vec(covy(~idxStitched));LDScovy_h(~idxStitched)]);
%         Mc = max([Mc;vec(covy(~idxStitched));LDScovy_h(~idxStitched)]);
%         hold off
%         plot(0,0)
%         line(1.1*[mc, Mc], 1.1*[mc, Mc], 'color', 'k', 'linewidth', 1.5)
%         hold on
%         plot(covy(idxStitched), LDScovy_h(idxStitched), 'b.', 'markerSize',7)
%         text(mc,0.9*Mc, ...
%              ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
%               num2str(corr(covy(idxStitched),LDScovy_h(idxStitched)))])
%         plot(covy(~idxStitched), LDScovy_h(~idxStitched), 'g.', 'markerSize',7)           
%         text(mc,0.7*Mc, ...
%              ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
%               num2str(corr(covy(~idxStitched),LDScovy_h(~idxStitched)))])
%         axis(1.1*[mc, Mc, mc, Mc])
%         xlabel('true cov(y)')
%         ylabel('est. cov(y)')
%         box off
%         
%         
%         pause
%     end
% else
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
        
    %covy = cov([y(:,2:end);y(:,1:end-1)]');
    %covy = covy(yDim+1:end, 1:yDim);
    
    subplot(2,2,1)
    imagesc(covy_tl)
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
    mc = min([vec(covy_tl(idxStitched));LDScovy_h(idxStitched)]);
    Mc = max([vec(covy_tl(idxStitched));LDScovy_h(idxStitched)]);
    mc = min([mc;vec(covy_tl(~idxStitched));LDScovy_h(~idxStitched)]);
    Mc = max([Mc;vec(covy_tl(~idxStitched));LDScovy_h(~idxStitched)]);
    hold off
    line(1.1*[mc, Mc], 1.1*[mc, Mc], 'color', 'k', 'linewidth', 1.5)
    hold on
    plot(covy_tl(idxStitched), LDScovy_h(idxStitched), 'b.', 'markerSize',7)
    if sum(idxStitched(:))>0
      text(mc,0.9*Mc, ...
        ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
        num2str(corr(covy_tl(idxStitched),LDScovy_h(idxStitched)))])
    end
    plot(covy_tl(~idxStitched), LDScovy_h(~idxStitched), 'g.', 'markerSize',7)
    if sum(~idxStitched(:))>0
      text(mc,0.7*Mc, ...
        ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
        num2str(corr(covy_tl(~idxStitched),LDScovy_h(~idxStitched)))])
    end
    axis(1.1*[mc, Mc, mc, Mc])
    xlabel('true cov(y)')
    ylabel('est. cov(y)')
    box off    
% end

% %% Check for twists in latent space
% figure;
% subplot(2,3,1), imagesc(Pi), title('\Pi = A \Pi A^T + Q')
% subplot(2,3,2), 
% plot(squeeze(Extxt_true(1,1,:))' - squeeze(Ext_true(1,:).*Ext_true(1,:)))
% title('cov(x_1,x_1) over time')
% subplot(2,3,3), 
% if xDim > 1
%     plot(squeeze(Extxt_true(xDim-1,xDim,:))' - squeeze(Ext_true(xDim-1,:).*Ext_true(xDim,:)))
% end
% title(['cov(x_',num2str(xDim-1), ',x_',num2str(xDim),') over time'])
% subplot(2,3,4), 
% t = 0.9 * obsScheme.obsTime(1);
% imagesc(squeeze(Extxt_true(:,:,t)- squeeze(Ext_true(:,t))*squeeze(Ext_true(:,t))'))
% title(['cov(x) at t = ', num2str(t)])
% if length(obsScheme.obsTime)>1
%     subplot(2,3,5), 
%     t = 0.9 * obsScheme.obsTime(2);
%     imagesc(squeeze(Extxt_true(:,:,t)- squeeze(Ext_true(:,t))*squeeze(Ext_true(:,t))'))
%     title(['cov(x) at t = ', num2str(t)])
% end
% if length(obsScheme.obsTime)>2
%     subplot(2,3,6), 
%     t = 0.9 * obsScheme.obsTime(3);
%     imagesc(squeeze(Extxt_true(:,:,t)- squeeze(Ext_true(:,t))*squeeze(Ext_true(:,t))'))
%     title(['cov(x) at t = ', num2str(t)])
% end
% 
% %% Check for twists in latent space
% figure;
% subplot(2,3,1), imagesc(Pi), title('\Pi = A \Pi A^T + Q')
% subplot(2,3,2), 
% plot(squeeze(Extxt_h(1,1,:))' - squeeze(Ext_h(1,:).*Ext_h(1,:)))
% title('cov(x_1,x_1) over time')
% subplot(2,3,3), 
% if xDim > 1
%     plot(squeeze(Extxt_h(xDim-1,xDim,:))' - squeeze(Ext_h(xDim-1,:).*Ext_h(xDim,:)))
% end
% title(['cov(x_',num2str(xDim-1), ',x_',num2str(xDim),') over time'])
% subplot(2,3,4), 
% t = 0.9 * obsScheme.obsTime(1);
% imagesc(squeeze(Extxt_h(:,:,t)- squeeze(Ext_h(:,t))*squeeze(Ext_h(:,t))'))
% title(['cov(x) at t = ', num2str(t)])
% if length(obsScheme.obsTime)>1
%     subplot(2,3,5), 
%     t = 0.9 * obsScheme.obsTime(2);
%     imagesc(squeeze(Extxt_h(:,:,t)- squeeze(Ext_h(:,t))*squeeze(Ext_h(:,t))'))
%     title(['cov(x) at t = ', num2str(t)])
% end
% if length(obsScheme.obsTime)>2
%     subplot(2,3,6), 
%     t = 0.9 * obsScheme.obsTime(3);
%     imagesc(squeeze(Extxt_h(:,:,t)- squeeze(Ext_h(:,t))*squeeze(Ext_h(:,t))'))
%     title(['cov(x) at t = ', num2str(t)])
% end

%%
% dt = 10;
% figure; 
% crosscov = A_h*Pi_h;
% crosscorr_emp = zeros(2*dt+1,1);
% for i = 1:xDim
%     for j = i:xDim
%         for t = -dt:dt, 
%             crosscorr_emp(t+dt+1) = corr(x(i,(dt+1:end-dt))', x(j,(dt+1:end-dt)+t)'); 
%         end
%         plot(-dt:dt, crosscorr_emp)
%         hold on
%         plot(1, A_h(i,j), 'ro');
%         hold off
%         title(['i = ', num2str(i), ', j = ', num2str(j)])
%         pause;
%     end
% end


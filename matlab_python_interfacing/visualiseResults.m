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

load(['/home/mackelab/Desktop/Projects/Stitching/code/le_stitch/python',...
      '/LDS_data_to_visualise.mat'])
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

constantInputLength = double(constantInputLength);

  T    = size(y,2); % simulation length
  Trial= size(y,3); % number of trials
  xDim = size(x,1); % q
  yDim = size(y,1); % p 
  uDim = size(u,1); % r
    

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
for i = 1:yDim     % brute force
    for j = 1:yDim % search ...
        for k = 1:numPops
            if ismember(i, subpops{k}) && ismember(j, subpops{k})
                idxStitched(i,j) = 0;
            end
        end
    end
end
%% bla

% figure;
% subplot(1,2,1)
% plot(x', 'k')
% hold on
% plot(Ext', 'b')
% plot(Ext_h', 'r')

%% Compare parameters
figure('Units', 'normalized','Position', [0.2,0.1,0.4,0.5]);
subplot(2,2,1)
if xDim > 1
    plot(1:xDim, sort(real(eig(A))), 'g', 'linewidth', 2)
    hold on
    plot(1:xDim, sort(real(eig(A_h))), 'b', 'linewidth', 2)
    xlabel('(sorted) eigenvalue')
else
    plot(1:xDim, sort(eig(A)), 'go', 'linewidth', 2, 'markerSize', 8)
    hold on
    plot(1:xDim, sort(eig(A_h)), 'bo', 'linewidth', 2, 'markerSize', 8)
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
    text(0,0,'Here be something with C and maybe Q')
    axis([0,5,-1,1])
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
bar((1:yDim)-0.5, d_h, 0.5, 'g', 'faceColor', 'g', 'edgeColor', 'none')
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

%% Compare instantaneous covariances of model with truth

if ifDataGeneratedWithInput && ifInputPiecewiseConstant
    figure('Units', 'normalized','Position', [0.2,0.1,0.4,0.5]);
    numPieces = double(floor(T/constantInputLength));
    m = min(u(:));
    M = max(u(:));
    spwidth = 17;
    sphight = 17;    
    subplot(sphight,spwidth,vec(bsxfun(@plus,(14:17)',(6:10)*spwidth)))
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
    subplot(sphight,spwidth,vec(bsxfun(@plus,(14:17)',(12:16)*spwidth)))
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
    for i = 1:numPieces
        
        covy = cov(y(:, (i-1)*constantInputLength + ...
                     (round(constantInputLength/2):constantInputLength))');
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
        title(['Instantaneous covariances cov(y_n, y_n)']) 
        axis([0, T+1, 1.2 * m, 1.2 * M])        
        set(gca, 'TickDir', 'out')
        xlabel('time n')
        ylabel('u[n]')
        subplot(sphight,spwidth,vec(bsxfun(@plus,(1:11)',(9:16)*spwidth)))
        mc = min([vec(covy(idxStitched));LDScovy_h(idxStitched)]);
        Mc = max([vec(covy(idxStitched));LDScovy_h(idxStitched)]);
        hold off
        plot(covy(idxStitched), LDScovy_h(idxStitched), 'b.', 'markerSize',7)
        hold on
        mc = min([mc;vec(covy(~idxStitched));LDScovy_h(~idxStitched)]);
        Mc = max([Mc;vec(covy(~idxStitched));LDScovy_h(~idxStitched)]);
        text(mc,0.9*Mc, ...
             ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
              num2str(corr(covy(idxStitched),LDScovy_h(idxStitched)))])
        plot(covy(~idxStitched), LDScovy_h(~idxStitched), 'g.', 'markerSize',7)           
        text(mc,0.7*Mc, ...
             ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
              num2str(corr(covy(~idxStitched),LDScovy_h(~idxStitched)))])
        line(1.1*[mc, Mc], 1.1*[mc, Mc], 'color', 'k', 'linewidth', 1.5)
        axis(1.1*[mc, Mc, mc, Mc])
        xlabel('true cov(y)')
        ylabel('est. cov(y)')
        box off
        
        
        pause
    end
else
    figure('Units', 'normalized','Position', [0.2,0.1,0.4,0.5]);
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
        
    covy = cov(y');
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
    title(['empirical cov(y)'])
    ylabel('# neuron')
    subplot(2,2,4)
    mc = min([vec(covy(idxStitched));LDScovy_h(idxStitched)]);
    Mc = max([vec(covy(idxStitched));LDScovy_h(idxStitched)]);
    hold off
    plot(covy(idxStitched), LDScovy_h(idxStitched), 'b.', 'markerSize',7)
    hold on
    mc = min([mc;vec(covy(~idxStitched));LDScovy_h(~idxStitched)]);
    Mc = max([Mc;vec(covy(~idxStitched));LDScovy_h(~idxStitched)]);
    text(mc,0.9*Mc, ...
        ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
        num2str(corr(covy(idxStitched),LDScovy_h(idxStitched)))])
    plot(covy(~idxStitched), LDScovy_h(~idxStitched), 'g.', 'markerSize',7)
    text(mc,0.7*Mc, ...
        ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
        num2str(corr(covy(~idxStitched),LDScovy_h(~idxStitched)))])
    line(1.1*[mc, Mc], 1.1*[mc, Mc], 'color', 'k', 'linewidth', 1.5)
    axis(1.1*[mc, Mc, mc, Mc])
    xlabel('true cov(y)')
    ylabel('est. cov(y)')
    box off
end


%% Compare time-lag covariances of model with truth

if ifDataGeneratedWithInput && ifInputPiecewiseConstant
    figure('Units', 'normalized','Position', [0.2,0.1,0.4,0.5]);
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
    title('$\hat{C} \hat{\Pi} \hat{C}^T + \hat{R}$' ,'interpreter','latex')
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
    title('C \Pi C^T + R') 
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
        title(['Instantaneous covariances cov(y_n, y_n)']) 
        axis([0, T+1, 1.2 * m, 1.2 * M])        
        set(gca, 'TickDir', 'out')
        xlabel('time n')
        ylabel('u[n]')
        subplot(sphight,spwidth,vec(bsxfun(@plus,(1:11)',(9:16)*spwidth)))
        mc = min([vec(covy(idxStitched));LDScovy_h(idxStitched)]);
        Mc = max([vec(covy(idxStitched));LDScovy_h(idxStitched)]);
        hold off
        plot(covy(idxStitched), LDScovy_h(idxStitched), 'b.', 'markerSize',7)
        hold on
        mc = min([mc;vec(covy(~idxStitched));LDScovy_h(~idxStitched)]);
        Mc = max([Mc;vec(covy(~idxStitched));LDScovy_h(~idxStitched)]);
        text(mc,0.9*Mc, ...
             ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
              num2str(corr(covy(idxStitched),LDScovy_h(idxStitched)))])
        plot(covy(~idxStitched), LDScovy_h(~idxStitched), 'g.', 'markerSize',7)           
        text(mc,0.7*Mc, ...
             ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
              num2str(corr(covy(~idxStitched),LDScovy_h(~idxStitched)))])
        line(1.1*[mc, Mc], 1.1*[mc, Mc], 'color', 'k', 'linewidth', 1.5)
        axis(1.1*[mc, Mc, mc, Mc])
        xlabel('true cov(y)')
        ylabel('est. cov(y)')
        box off
        
        
        pause
    end
else
    figure('Units', 'normalized','Position', [0.2,0.1,0.4,0.5]);
    subplot(2,2,2)
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
    title('$\hat{C} \hat{\Pi} \hat{C}^T + \hat{R}$' ,'interpreter','latex')
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
    title('C \Pi C^T + R') 
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
    title(['empirical cov(y)'])
    ylabel('# neuron')
    subplot(2,2,4)
    mc = min([vec(covy(idxStitched));LDScovy_h(idxStitched)]);
    Mc = max([vec(covy(idxStitched));LDScovy_h(idxStitched)]);
    hold off
    plot(covy(idxStitched), LDScovy_h(idxStitched), 'b.', 'markerSize',7)
    hold on
    mc = min([mc;vec(covy(~idxStitched));LDScovy_h(~idxStitched)]);
    Mc = max([Mc;vec(covy(~idxStitched));LDScovy_h(~idxStitched)]);
    text(mc,0.9*Mc, ...
        ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
        num2str(corr(covy(idxStitched),LDScovy_h(idxStitched)))])
    plot(covy(~idxStitched), LDScovy_h(~idxStitched), 'g.', 'markerSize',7)
    text(mc,0.7*Mc, ...
        ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
        num2str(corr(covy(~idxStitched),LDScovy_h(~idxStitched)))])
    line(1.1*[mc, Mc], 1.1*[mc, Mc], 'color', 'k', 'linewidth', 1.5)
    axis(1.1*[mc, Mc, mc, Mc])
    xlabel('true cov(y)')
    ylabel('est. cov(y)')
    box off    
end

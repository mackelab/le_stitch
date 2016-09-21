
function make_fitting_result_overview_plots(y,x,u,truePars,estPars,initPars,Pi,Pi_h,obs_scheme)
%% Visualises results of the EM stitching implementation
% Stitching in space and time.



%% load, translate, compute
  
% gives the following variables:
% observed data traces              y
% input data traces                 u
% true sampled latent traces        z
% true parameters                   {A,  B,  Q,  mu0,  V0,  C,  d,  R  }
% initial parameters                {A_0,B_0,Q_0,mu0_0,V0_0,C_0,d_0,R_0}
% fitted parameters                 {A_h,B_h,Q_h,mu0_h,V0_h,C_h,d_h,R_h}
% derived statistics                 Pi, Pi_h, Pi_t, Pi_t_h

y = y';
x = x';
u = u';

[p,T] = size(y);
n = size(x,1);

A = truePars.A;
B = 0;
Q = truePars.Q;
mu0 = truePars.mu0;
V0 = truePars.V0;
C = truePars.C;
d = truePars.d;
R = diag(truePars.R);

A_h = estPars.A;
B_h = 0;
Q_h = estPars.Q;
mu0_h = estPars.mu0;
V0_h = estPars.V0;
C_h = estPars.C;
d_h = estPars.d;
R_h = diag(estPars.R);

A_0 =  initPars.A;
B_0 = 0;
Q_0 = initPars.Q;
mu0_0 = initPars.mu0;
V0_0 = initPars.V0;
C_0 = initPars.C;
d_0 = initPars.d;
R_0 = initPars.R;

% 
% obs_scheme.sub_pops = {0:4, 4:8};
% obs_scheme.obs_pops = [0, 1];
% obs_scheme.obs_time = [fix(T/2)-1, T-1];

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

  T    = size(y,2); % simulation length
  Trial= size(y,3); % number of trials
  xDim = size(A_h,1); % q
  yDim = size(y,1);   % p 
  uDim = size(u,1);   % r
    
%% Visualise observation protocol

if ~ iscell(obs_scheme.sub_pops)
  numPops = size(obs_scheme.sub_pops,1);
  sub_pops = cell(numPops,1);
  for i = 1:numPops
      sub_pops{i} = double(obs_scheme.sub_pops(i,:))+1;
  end
else
  numPops = length(obs_scheme.sub_pops);
  sub_pops = obs_scheme.sub_pops;
  for i = 1:numPops
      sub_pops{i} = double(sub_pops{i})+1;
  end  
  numPops = length(sub_pops);
end

idxStitched = true(yDim,yDim); % (i,j) for each pair of variables ...  
for k = 1:numPops
    idxStitched(sub_pops{k},sub_pops{k}) = 0;
end

protocol = false(yDim, T);
idx = sub_pops{obs_scheme.obs_pops(1)+1};
protocol(idx,1:obs_scheme.obs_time(1)) = true;
for i = 2:length(obs_scheme.obs_time)
    idx = sub_pops{obs_scheme.obs_pops(i)+1};
    protocol(idx,obs_scheme.obs_time(i-1)+1:obs_scheme.obs_time(i)) = true;
end

figure('Units', 'normalized','Position', [0.1,0.3,0.8,0.6]);
imagesc(protocol)
title('stitching protocol (probably too dense to properly visualize')
xlabel('time')
ylabel('neurons')


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
    [eigA_hs, idxS] = sort(abs(eig(A_h)));
    if xDim > 1
        plot(1:xDim, sort(abs(eig(A))), 'g', 'linewidth', 2)
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
    axis([0.5,xDim+0.5,min(abs([eig(A);eig(A_h)]))-0.1, ...
                       max(abs([eig(A);eig(A_h)]))+0.1])  
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
if numel(u)>1 && stcmp(inputType,'pwdconst')
    numPieces = double(floor(T/constantInputLength));
    m = min(u(:));
    M = max(u(:));
    spwidth = 17;
    sphight = 17;    
    subplot(sphight,spwidth,vec(bsxfun(@plus,(14:17)',(6:10)*spwidth)))
    LDScovy_h = C_h * (Pi_h) * C_h' + diag(R_h);
    imagesc(LDScovy_h)
    for ip = 1:numPops
        if unique(diff(sort(sub_pops{ip}))) == [1]
            mP = min(sub_pops{ip})-0.5;
            MP = max(sub_pops{ip})+0.5;
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
        if unique(diff(sort(sub_pops{ip}))) == [1]
            mP = min(sub_pops{ip})-0.5;
            MP = max(sub_pops{ip})+0.5;
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
            if unique(diff(sort(sub_pops{ip}))) == [1]
                mP = min(sub_pops{ip})-0.5;
                MP = max(sub_pops{ip})+0.5;
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
        mc = min([mc;vec(covy(~idxStitched));LDScovy_h(~idxStitched)]);
        Mc = max([Mc;vec(covy(~idxStitched));LDScovy_h(~idxStitched)]);
        hold off
        plot(0,0)
        line(1.1*[mc, Mc], 1.1*[mc, Mc], 'color', 'k', 'linewidth', 1.5)
        hold on
        plot(covy(idxStitched), LDScovy_h(idxStitched), 'r.', 'markerSize',21)
        text(mc,0.9*Mc, ...
             ['stitch. corr(cov(y)_{est},cov(y)_{true})  = ', ...
              num2str(corr(covy(idxStitched),LDScovy_h(idxStitched)))])
        plot(covy(~idxStitched), LDScovy_h(~idxStitched), 'b.', 'markerSize',7)           
        text(mc,0.7*Mc, ...
             ['obs. corr(cov(y)_{est},cov(y)_{true})  = ', ...
              num2str(corr(covy(~idxStitched),LDScovy_h(~idxStitched)))])
        axis(1.1*[mc, Mc, mc, Mc])
        xlabel('true cov(y)')
        ylabel('est. cov(y)')
        box off
        
        
        pause
    end
else
    subplot(2,2,2)
    LDScovy_h = C_h * (Pi_h) * C_h' + diag(R_h);
    imagesc(LDScovy_h)
    for ip = 1:numPops
        if unique(diff(sort(sub_pops{ip}))) == [1]
            mP = min(sub_pops{ip})-0.5;
            MP = max(sub_pops{ip})+0.5;
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
        if unique(diff(sort(sub_pops{ip}))) == [1]
            mP = min(sub_pops{ip})-0.5;
            MP = max(sub_pops{ip})+0.5;
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
        if unique(diff(sort(sub_pops{ip}))) == [1]
            mP = min(sub_pops{ip})-0.5;
            MP = max(sub_pops{ip})+0.5;
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
    plot(covy(idxStitched), LDScovy_h(idxStitched), 'r.', 'markerSize',21)
    hold on
    if sum(idxStitched(:)>0)
     text(mc,0.9*Mc, ...
        ['corr(cov(y)_{est},cov(y)_{true})  = ', ...
        num2str(corr(covy(idxStitched),LDScovy_h(idxStitched)))])
    end
    plot(covy(~idxStitched), LDScovy_h(~idxStitched), 'b.', 'markerSize',7)
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
figure('Units', 'normalized','Position', [0.1,0.0,0.4,0.5]);
if numel(u)>1 && stcmp(inputType,'pwdconst')
    numPieces = double(floor(T/constantInputLength));
    m = min(u(:));
    M = max(u(:));
    spwidth = 17;
    sphight = 17;    
    subplot(sphight,spwidth,vec(bsxfun(@plus,(14:17)',(6:10)*spwidth)))
    LDScovy_tl_h = C_h * (A_h * Pi_h) * C_h';
    imagesc(LDScovy_tl_h)
    for ip = 1:numPops
        if unique(diff(sort(sub_pops{ip}))) == [1]
            mP = min(sub_pops{ip})-0.5;
            MP = max(sub_pops{ip})+0.5;
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
    LDScovy_tl = C * (A* Pi) * C';
    imagesc(LDScovy_tl)
    for ip = 1:numPops
        if unique(diff(sort(sub_pops{ip}))) == [1]
            mP = min(sub_pops{ip})-0.5;
            MP = max(sub_pops{ip})+0.5;
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
        covy_tl = cov([y(:, (i-1)*constantInputLength + ...
                     (round(constantInputLength/2):constantInputLength));
                    y(:, (i-1)*constantInputLength -1 +  ...
                     (round(constantInputLength/2):constantInputLength))]');
        covy_tl = covy_tl(yDim+1:end, 1:yDim); % take lover left corner                 
        subplot(sphight,spwidth,vec(bsxfun(@plus,(14:17)',(0:4)*spwidth)))
        imagesc(covy_tl)
        for ip = 1:numPops
            if unique(diff(sort(sub_pops{ip}))) == [1]
                mP = min(sub_pops{ip})-0.5;
                MP = max(sub_pops{ip})+0.5;
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
        mc = min([vec(covy_tl(idxStitched));LDScovy_tl_h(idxStitched)]);
        Mc = max([vec(covy_tl(idxStitched));LDScovy_tl_h(idxStitched)]);
        mc = min([mc;vec(covy_tl(~idxStitched));LDScovy_tl_h(~idxStitched)]);
        Mc = max([Mc;vec(covy_tl(~idxStitched));LDScovy_tl_h(~idxStitched)]);
        hold off
        plot(0,0)
        line(1.1*[mc, Mc], 1.1*[mc, Mc], 'color', 'k', 'linewidth', 1.5)
        hold on
        plot(covy_tl(idxStitched), LDScovy_tl_h(idxStitched), 'r.', 'markerSize',7)
        text(mc,0.9*Mc, ...
             ['stitch. corr(cov(y)_{est},cov(y)_{true})  = ', ...
              num2str(corr(covy_tl(idxStitched),LDScovy_tl_h(idxStitched)))])
        plot(covy_tl(~idxStitched), LDScovy_tl_h(~idxStitched), 'b.', 'markerSize',7)           
        text(mc,0.7*Mc, ...
             ['obs. corr(cov(y)_{est},cov(y)_{true})  = ', ...
              num2str(corr(covy_tl(~idxStitched),LDScovy_tl_h(~idxStitched)))])
        axis(1.1*[mc, Mc, mc, Mc])
        xlabel('true cov(y)')
        ylabel('est. cov(y)')
        box off
        
        
        pause
    end
else
    subplot(2,2,2)
    LDScovy_tl_h = C_h * (A_h *Pi_h) * C_h';
    imagesc(LDScovy_tl_h)
    for ip = 1:numPops
        if unique(diff(sort(sub_pops{ip}))) == [1]
            mP = min(sub_pops{ip})-0.5;
            MP = max(sub_pops{ip})+0.5;
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
    LDScovy_tl = C * (A* Pi) * C';
    imagesc(LDScovy_tl)
    for ip = 1:numPops
        if unique(diff(sort(sub_pops{ip}))) == [1]
            mP = min(sub_pops{ip})-0.5;
            MP = max(sub_pops{ip})+0.5;
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
        
    covy_tl = cov([y(:,1:end-1);y(:,2:end)]');
    covy_tl = covy_tl(yDim+1:end, 1:yDim);
    subplot(2,2,1)
    imagesc(covy_tl)
    for ip = 1:numPops
        if unique(diff(sort(sub_pops{ip}))) == [1]
            mP = min(sub_pops{ip})-0.5;
            MP = max(sub_pops{ip})+0.5;
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
    mc = min([vec(covy_tl(idxStitched));LDScovy_tl_h(idxStitched)]);
    Mc = max([vec(covy_tl(idxStitched));LDScovy_tl_h(idxStitched)]);
    mc = min([mc;vec(covy_tl(~idxStitched));LDScovy_tl_h(~idxStitched)]);
    Mc = max([Mc;vec(covy_tl(~idxStitched));LDScovy_tl_h(~idxStitched)]);
    hold off
    line(1.1*[mc, Mc], 1.1*[mc, Mc], 'color', 'k', 'linewidth', 1.5)
    hold on
    plot(covy_tl(idxStitched), LDScovy_tl_h(idxStitched), 'r.', 'markerSize',7)
    if sum(idxStitched(:))>0
      text(mc,0.9*Mc, ...
        ['stitch. corr(cov(y)_{est},cov(y)_{true})  = ', ...
        num2str(corr(covy_tl(idxStitched),LDScovy_tl_h(idxStitched)))])
    end
    plot(covy_tl(~idxStitched), LDScovy_tl_h(~idxStitched), 'b.', 'markerSize',7)
    if sum(~idxStitched(:))>0
      text(mc,0.7*Mc, ...
        ['obs. corr(cov(y)_{est},cov(y)_{true})  = ', ...
        num2str(corr(covy_tl(~idxStitched),LDScovy_tl_h(~idxStitched)))])
    end
    axis(1.1*[mc, Mc, mc, Mc])
    xlabel('true cov(y)')
    ylabel('est. cov(y)')
    box off    
end

end

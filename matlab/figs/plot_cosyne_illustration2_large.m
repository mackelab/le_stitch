%% set up

clear all
addpath(genpath('/home/mackelab/Desktop/Projects/Stitching/code/code_le_stitch/figure_generation'))

abs_path_to_data = '/home/mackelab/Desktop/Projects/Stitching/results/cosyne_poster/';
exps = 'illustration_2';

cd([abs_path_to_data, exps])

p = 90;
overlap = 0;
%% gather data
idx_datasets={'test_problemsLDS_save_tempStitching_sparse', 'test_problemsLDS_save_tempStitching_mixed'};

performances_init = zeros(2,1);
performances_fit  = zeros(2,1);
llikes_fit = zeros(2,1);

covys_est   = cell(length(idx_datasets),1);        
covys_true  = cell(length(idx_datasets),1);  
covys_init  = cell(length(idx_datasets),1);  

crosscorrs = cell(length(idx_datasets),p,p);
crossc_est = cell(length(idx_datasets),p,p);

obs_scheme.sub_pops = {[], 1:90, 1:45, 46:90};

idx_stitched12 = true(p,p);
for j = 1:length(obs_scheme.sub_pops)
    idx_stitched12(obs_scheme.sub_pops{j}+1,obs_scheme.sub_pops{j}+1) = false;
end

for idx_set = 1:length(idx_datasets)    
   
    load(['data/', idx_datasets{idx_set}, '.mat'])    
    initPars.A = A_0; initPars.B = B_0; initPars.Q = Q_0; initPars.C = C_0; initPars.d = d_0; initPars.R = diag(R_0);
    clear A_0 B_0 Q_0 C_0 d_0 R_0
    estPars.A = A_h; estPars.B = B_h; estPars.Q = Q_h; estPars.C = C_h; estPars.d = d_h; estPars.R = diag(R_h);
    clear A_h B_h Q_h C_h d_h R_h
    truePars.A = A; truePars.B = B; truePars.Q = Q; truePars.C = C; truePars.d = d; truePars.R = diag(R);
    clear A B Q C d R
    y=y';
    [T,p] = size(y);
    
    Pi_est = direct_dlyap( estPars.A, estPars.Q);     
    covys_true{idx_set} = cov(y);
    covys_est{idx_set} =  estPars.C * Pi_est *  estPars.C' + estPars.R;
    covys_init{idx_set} = initPars.C * direct_dlyap( initPars.A, initPars.Q) * initPars.C' + initPars.R;    
    
    obs_scheme.obs_time = obsScheme.obsTime;
    obs_scheme.obs_pops = obsScheme.obsPops;
    obs_scheme.sub_pops = obsScheme.subpops;
    clear obsScheme

    if sum(vec(idx_stitched12)) > 0
        tmp = corrcoef(covys_init{idx_set}(idx_stitched12), covys_true{idx_set}(idx_stitched12));
        performances_init(idx_set) = tmp(1,2);
        tmp = corrcoef(covys_est{idx_set}(idx_stitched12), covys_true{idx_set}(idx_stitched12));
        performances_fit(idx_set) = tmp(1,2);
        llikes_fit(idx_set) = LL(end);
    else % if nothing to stitch, report overall goodness
        tmp = corrcoef(vec(covys_init), vec(covys_true));
        performances_init(idx_set) = tmp(1,2);
        tmp = corrcoef(vec(covys_est), vec(covys_true));
        performances_fit(idx_set) = tmp(1,2);
        llikes_fit(idx_set) = LL(end);
    end

 
 offset_length=5;
 window_size = 2*offset_length+1;
 crosscorr_ts = -offset_length:offset_length;

 for idxi = 1:p
  for idxj = idxi:p        
    crosscorrs{idx_set,idxi,idxj} = zeros(window_size,1);
    crossc_est{idx_set,idxi,idxj} = zeros(window_size,1);    
  end
 end
 for t= crosscorr_ts 
  if t > 0
    tmpcove = estPars.C * (estPars.A^t) * Pi_est * estPars.C';
  elseif t < 0
    tmpcove = estPars.C *  Pi_est * ((estPars.A')^(-t)) * estPars.C';
  else
    tmpcove = covys_est{idx_set};
  end
  norm = diag(1./sqrt(diag(covys_est{idx_set})));                
  tmpcove = norm*tmpcove*norm;
  for idxi = 1:p
    for idxj = idxi:p
        tmp = 1+offset_length:T-offset_length;
        tmpcov = corrcoef(y(tmp,idxi),y(tmp+t,idxj));
        crosscorrs{idx_set,idxi,idxj}(t+offset_length+1) = tmpcov(1,2);
        crossc_est{idx_set,idxi,idxj}(t+offset_length+1) = tmpcove(idxi,idxj);
    end

  end
 end
end

%% plot 'params' covariances

figure;
for idx_set = 1:length(idx_datasets)
    subplot(length(idx_datasets),2, (idx_set-1)*2+1)
    imagesc(covys_true{idx_set})
    title(['data set #', num2str(idx_set), ' true covs'])
    subplot(length(idx_datasets),2, 2*idx_set)
    imagesc(covys_est{idx_set})    
    title(['data set #', num2str(idx_set), ' est. covs'])
end

%% plot cross-correlograms
sub_set_i  = [5,11,46,81];
num_rows = length(sub_set_i);
clrs = [0,100,200;200,50,10]/256;
%while true
    %sub_set_i = sort(randsample(p,num_rows));
    %figure
     
    for idx_set = 1:length(idx_datasets)
     for idxi = 1:length(sub_set_i)
      for idxj = idxi:length(sub_set_i)
       subplot(num_rows,num_rows,(idxi-1)*num_rows+idxj)
       
       if (idxi-1)*num_rows+idxj == 4
           plot(-10,-10, 'color', clrs(1,:),'linewidth', 1.5)
           hold on
           plot(-10,-10, 'color', clrs(2,:),'linewidth', 1.5)           
           legend('single time scale', 'mixed time scale')
       end
    
       i = sub_set_i(idxi);    j = sub_set_i(idxj);

       if idx_set==1
           plot(crosscorr_ts, crosscorrs{idx_set,i,j}, '-', 'color', [0,0,0]/256, 'linewidth', 3)
           hold on
           plot(crosscorr_ts, crossc_est{idx_set,i,j}, '-', 'color', clrs(1,:), 'linewidth', 1.5)
       else
           plot(crosscorr_ts, crossc_est{idx_set,i,j}, '-', 'color', clrs(2,:), 'linewidth', 1.5)       
           hold off 
       end
       title([num2str(i), '/', num2str(j)])
       axis([min(crosscorr_ts),max(crosscorr_ts), -0.5, 1])
       if j == i
           xlabel('\Delta{}t')
           ylabel('cov')
       else
           set(gca, 'XTick', [])
           set(gca, 'YTick', [])
       end
       set(gca, 'TickDir', 'out') 
       box off
      end
     end
    end
%pause 
%end
idx_stitched_1 = 46:90;
idx_stitched_2 = 1:45;

perfs = zeros(6,2);
subplots = [5,9,10,13,14,15];
for t = 0:5
  subplot(num_rows,num_rows,subplots(t+1))
  for idx_set = 1:2
    if idx_set==1, 
        clr = clrs(1,:);
        disp('yep')
    else
        clr = clrs(2,:);
    end
    alltrue = []; allest = [];
    for i = idx_stitched_2
        for j = idx_stitched_1
            alltrue = [alltrue, crosscorrs{idx_set,i,j}];
            allest =  [allest, crossc_est{idx_set,i,j}];
        end
    end
    plot(alltrue(t+offset_length+1,:), allest(t+offset_length+1,:), '.', 'color', clr)
    tmp =  corrcoef(alltrue(t+offset_length+1,:),allest(t+offset_length+1,:));
    perfs(t+1,idx_set) = tmp(1,2);
    hold on
    title(['\Delta{}t +', num2str(t)])
    axis square
    set(gca, 'TickDir', 'out') 
    box off
  end
  hold off
end

%% 
figure; 
plot(0:5,perfs(:,1), 'color', clrs(1,:), 'linewidth', 2.5)
hold on
plot(0:5,perfs(:,2), 'color', clrs(2,:), 'linewidth', 2.5)
xlabel('\Delta{}t')
ylabel('$\rho(t, t+\Delta{t})$', 'interpreter', 'Latex')
box off
set(gca, 'TickDir', 'out')


%% plot 'params' covariances

figure;
covys_true = cov(y');
covys_est = C_h * direct_dlyap(A_h,Q_h) * C_h' + diag(R_h);
m = min([covys_true(:);covys_est(:)]);
M = max([covys_true(:);covys_est(:)]);
subplot(1,2,1)
imagesc(covys_true, [m,M])
axis square
axis off
title('true covs')
subplot(1,2,2)
imagesc(covys_est, [m,M])    
axis off
axis square
title('est. covs')
colormap('gray')
colorbar

%%
covy = cov([y(2:end,:),y(1:end-1,:)]);

covy_e = zeros(2*p);
covy_e(1:p,1:p) = estPars.C * direct_dlyap(estPars.A,estPars.Q) * estPars.C' + estPars.R;
covy_e(p+(1:p),p+(1:p)) = covy_e(1:p,1:p);
covy_e(1:p, p+(1:p)) = estPars.C * (estPars.A * direct_dlyap(estPars.A,estPars.Q)) * estPars.C';
covy_e(p+(1:p), 1:p) = covy_e(1:p, p+(1:p));

m = min([covy(:);covy_e(:)]);
M = max([covy(:);covy_e(:)]);
figure;
subplot(1,2,1)
imagesc(covy, [m,M])
axis square
axis off
title(['true covs'])

subplot(1,2,2)
subplot(length(idx_datasets),2, 2*idx_set)
imagesc(covy_e, [m,M])
axis off
axis square
title(['est. covs'])
colormap('gray')
colorbar


%%
figure; 

idx_datasets={'test_problemsLDS_save_tempStitching_sparse', 'test_problemsLDS_save_tempStitching_mixed'};
idx_set = 1;
load(['data/', idx_datasets{idx_set}, '.mat'])    
    
subplot(1,2,1)
h = plot(6:-1:1, log(0.5) ./ log(sort(eig(A))), 'ko', 'markerSize', 7, 'linewidth', 3);
set(h,'MarkerEdgeColor','none','MarkerFaceColor','k')
hold on

h = plot(6:-1:1, log(0.5) ./ log(sort(eig(A_h))), 'go', 'markerSize', 7, 'linewidth', 3);
set(h,'MarkerEdgeColor','none','MarkerFaceColor','g')
axis([0.5,3.5,0,1]), axis autoy
box off
set(gca, 'TickDir', 'out')
xlabel('# latent mode')
ylabel('half life (bins)')
set(gca, 'XTick', 1:3)


subplot(1,2,2)
h = plot(6:-1:1, 20 *log(0.5) ./ log(sort(eig(A))), 'ko', 'markerSize', 7, 'linewidth', 3);
set(h,'MarkerEdgeColor','none','MarkerFaceColor','k')
hold on

h = plot(6:-1:1, 20 *log(0.5) ./ log(sort(eig(A_h))), 'go', 'markerSize', 7, 'linewidth', 3);
set(h,'MarkerEdgeColor','none','MarkerFaceColor','g')
axis([0.5,3.5,0,1]), axis autoy
box off
set(gca, 'TickDir', 'out')
xlabel('# latent mode')
ylabel('half life [ms]')
set(gca, 'XTick', 1:3)
%% set up

clear all
addpath(genpath('/home/mackelab/Desktop/Projects/Stitching/code/code_le_stitch/figure_generation'))

abs_path_to_data = '/home/mackelab/Desktop/Projects/Stitching/results/cosyne_poster/';
exps = 'illustration_1';

cd([abs_path_to_data, exps])


p = 9;
overlap = 3; % data exists for overlap = [1, 3, 9]

%% gather data
idx_datasets=[7];

initialisers = {'random', 'params_naive','params'};
num_repets_per_initialiser = [10,1,1];
performances_init = {zeros(length(idx_datasets),num_repets_per_initialiser(1)), ...
                     zeros(length(idx_datasets),num_repets_per_initialiser(2)), ...
                     zeros(length(idx_datasets),num_repets_per_initialiser(3))};
performances_fit  = {zeros(length(idx_datasets),num_repets_per_initialiser(1)), ...
                     zeros(length(idx_datasets),num_repets_per_initialiser(2)), ...
                     zeros(length(idx_datasets),num_repets_per_initialiser(3))};
llikes_fit = { -Inf*zeros(length(idx_datasets),num_repets_per_initialiser(1)), ...
               -Inf*zeros(length(idx_datasets),num_repets_per_initialiser(2)), ...
               -Inf*zeros(length(idx_datasets),num_repets_per_initialiser(3))};

params_ssid = cell(length(idx_datasets),1);         
covys_est   = cell(length(idx_datasets),1);        
covys_true  = cell(length(idx_datasets),1);  

crosscorrs = cell(length(idx_datasets),p,p);
crossc_est = cell(length(idx_datasets),p,p);

if overlap==1    
    obs_scheme.sub_pops = {0:4, 4:8};
elseif overlap==3
    obs_scheme.sub_pops = {0:3, 5:8};    
elseif overlap==9
    obs_scheme.sub_pops = {0:8};
end
obs_scheme.obs_pops = [0,1];
idx_stitched12 = true(p,p);
for j = 1:length(obs_scheme.sub_pops)
    idx_stitched12(obs_scheme.sub_pops{j}+1,obs_scheme.sub_pops{j}+1) = false;
end

for idx_set = 1:length(idx_datasets)    
   
    load(['fits/overlap_', num2str(overlap), '/random_rep0_LDS_save_idx', num2str(idx_datasets(idx_set)-1), '.npz.mat'])
    [T,p] = size(y);
    covy_true = cov(y);
    
    obs_scheme.obs_time = [T/2, T];
    
 disp(['dataset ', num2str(idx_set)])
 for idx_init = 1:length(initialisers) 
         
  initialiser= initialisers{idx_init};
  disp(initialiser)    
  num_reps = num_repets_per_initialiser(idx_init);
  for rep = 1:num_reps
    loadfile = ['fits/overlap_', num2str(overlap), '/',initialiser,'_rep', num2str(rep-1), '_LDS_save_idx', num2str(idx_datasets(idx_set)-1), '.npz.mat'];
    load(loadfile)
    
     if ifBroken
        disp('run broke!')
        performances_init{idx_init}(idx_set,rep) = NaN;
        performances_fit{idx_init}(idx_set,rep) = NaN;
     else
        [T,p] = size(y);
        if strcmp(initialiser, 'random')
            covy_init = initPars.C * direct_dlyap( initPars.A, initPars.Q) * initPars.C' + diag(initPars.R);                
        else
            covy_init = initPars.C * initPars.Pi * initPars.C' + diag(initPars.R);
        end
        covy_est  =  estPars.C * direct_dlyap( estPars.A, estPars.Q) *  estPars.C' + estPars.R;

        if sum(vec(idx_stitched12)) > 0
            tmp = corrcoef(covy_init(idx_stitched12), covy_true(idx_stitched12));
            performances_init{idx_init}(idx_set,rep) = tmp(1,2);
            tmp = corrcoef(covy_est(idx_stitched12), covy_true(idx_stitched12));
            performances_fit{idx_init}(idx_set,rep) = tmp(1,2);
            llikes_fit{idx_init}(idx_set,rep) = ll(end);
        else % if nothing to stitch, report overall goodness
            tmp = corrcoef(vec(covy_init), vec(covy_true));
            performances_init{idx_init}(idx_set,rep) = tmp(1,2);
            tmp = corrcoef(vec(covy_est), vec(covy_true));
            performances_fit{idx_init}(idx_set,rep) = tmp(1,2);
            llikes_fit{idx_init}(idx_set,rep) = ll(end);
            
        end
        
        if strcmp(initialiser,'params')
            
            Pi_est = direct_dlyap( estPars.A, estPars.Q);
            
             params_ssid{idx_set} = estPars;
             covys_est{idx_set} = covy_est;
             covys_true{idx_set} = covy_true;
        
            offset_length=5;
            window_size = 2*offset_length+1;
            crosscorr_ts = -offset_length:offset_length;

            for idxi = 1:p
              for idxj = idxi:p        
                crosscorrs{idx_set,idxi,idxj} = zeros(window_size,1);
              end
            end
            for t= crosscorr_ts 
              if t > 0
                tmpcove = estPars.C * (estPars.A^t) * Pi_est * estPars.C';
              elseif t < 0
                tmpcove = estPars.C *  Pi_est * ((estPars.A')^t) * estPars.C';
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
        
     end    
  end  
 end

 
end

%% plot ll vs corr(covs) for different initialisers
figure
clrs = {'b', 'g', 'r'};
symbols = {'.', 'o', '*'};
for idx_set = 1:length(idx_datasets)    
      
 disp(['dataset ', num2str(idx_set)])
 for idx_init = 1:length(initialisers) 
         
  initialiser= initialisers{idx_init};
  disp(initialiser)    
  num_reps = num_repets_per_initialiser(idx_init);
  subplot(1,length(idx_datasets) ,idx_set)
  for rep = 1:num_reps    
    plot(llikes_fit{idx_init}(idx_set,rep), performances_fit{idx_init}(idx_set,rep), 'marker', symbols{idx_init}, 'color', clrs{idx_init})
    hold on    
  end
  box off
  xlabel('log-likelihood')
  ylabel('correlation of covariances')
  title(['data set #', num2str(idx_set)])  
 end
end

%% corr(covs) for different initialisers
figure
clrs = {'b', 'g', 'r'};
symbols = {'.', '*', 'o'};
for idx_set = 1:length(idx_datasets)    
 xs = rand(num_repets_per_initialiser(1),1);
 xs = xs - mean(xs);     
 xs(1) = 0;
 disp(['dataset ', num2str(idx_set)])
 for idx_init = [1,3]
         
  initialiser= initialisers{idx_init};
  disp(initialiser)    
  num_reps = num_repets_per_initialiser(idx_init);
  subplot(1,length(idx_datasets) ,idx_set)
  for rep = 1:num_reps    
    plot(xs(rep), performances_fit{idx_init}(idx_set,rep), 'marker', ...
         symbols{idx_init}, 'markerSize', 10, 'color', clrs{idx_init}, ...
         'MarkerFaceColor', clrs{idx_init})
    hold on    
  end
  box off
  set(gca,'XTick', [])
  ylabel('correlation of covariances')
  title(['data set #', num2str(idx_set)])
  set(gca, 'TickDir', 'out')
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
    colormap('gray')
end

%% plot cross-correlograms
figure;
for idx_set = 1:length(idx_datasets)
 for idxi = 1:p
  for idxj =idxi:p
   subplot(p,p,(idxi-1)*p+idxj)
   plot(crosscorr_ts, crosscorrs{idx_set,idxi,idxj}, 'color', 'k')
   hold on
   plot(crosscorr_ts, crossc_est{idx_set,idxi,idxj}, 'color', 'r')
   hold off 
   axis([min(crosscorr_ts),max(crosscorr_ts), -0.5, 1])
   if idxi == idxj
       xlabel('\Delta{}t')
       ylabel('cov')
   else
       set(gca, 'XTick', [])
       set(gca, 'YTick', [])
   end
   box off
  end
 end
 pause;
end
clear all
close all
clc

addpath(genpath('/home/mackelab/Desktop/Projects/Stitching/code/cosyne_poster/matlab'))
cd /home/mackelab/Desktop/Projects/Stitching/code/cosyne_poster
% load data
path = '../../results/cosyne_poster/debug/';
list = what(path);
pattern = 'p11';
idx_exps = strfind(list.mat, pattern);
idx_exps_i = ones(size(idx_exps));
for i = length(idx_exps):-1:1
    idx_exps_i(i) = isempty(idx_exps{i});
end
list.mat = list.mat(~logical(idx_exps_i));    

num_exps = length(list.mat);

% run algorithm
experiments = cell(num_exps,1);

num_iter = 500;
num_subpops = 2;
overlap = 1;

idx_exps = 1; [1:63, 65:84, 86:num_exps] %num_exps;
for i = idx_exps
    save_file = list.mat{i};
    load([path, save_file])

    xDim =  size(x,2);

    [experiments{i}.params, ...
     experiments{i}.corrs_stitched, ...
     experiments{i}.corrs_observed, ...
     experiments{i}.SIGfps, experiments{i}.SIGfps_f, ...
     experiments{i}.PSIG] = ...
        run_stitchingSSID_experiment(y', xDim, num_iter, ...
                                     num_subpops, overlap, false);   
    
    pause(0.1);
    close all;
end

%%

clrs = copper(num_exps);
perfs_yy = zeros(num_iter+1, num_exps);
perfs_yyl = zeros(num_iter+1, num_exps);
perfs_obs = zeros(num_iter+1, num_exps);

figure1 = figure('units','centimeters','position',[10,5,30,20]);
% subplot(1,2,1), 
for i = idx_exps
    perfs_yy(:,i) = experiments{i}.corrs_stitched(:,1);
    perfs_yyl(:,i) = experiments{i}.corrs_stitched(:,2);
    
    PSIG = ~isnan(experiments{i}.SIGfp);
    perfs_obs(:,i) = mean( (experiments{i}.SIGfp(PSIG) - experiments{i}.SIGfp_f(PSIG)).^2 ); 
%    plot(perfs_yy(:,i), 'color', clrs(i,:)); 
%    hold on; 
end
% plot(mean(perfs_yy(:,idx_exps),2), 'linewidth', 2, 'color', 'r')
% hold off;
%     
subplot(3,3,[1:3,4:6]), 
plot(perfs_yyl, 'k'); 
hold on; 
plot(mean(perfs_yyl(:,idx_exps),2), 'linewidth', 2, 'color', 'r')
hold off;
xlabel('iterations')
ylabel('corr(est. stitched covs, true stitched covs)')
title('correlations between true and estimated entries of covariance matrix that need stitching')

subplot(3,3,[7:9]), 
plot(mean(abs(perfs_yyl(:,idx_exps)),2), 'linewidth', 2, 'color', 'r')
title('average absolute correlation between true and estimated entries of covariance matrix that need stitching')
xlabel('iteration')

%%
tmp = zeros(num_iter+1,1);
for i = 6
    for t = 1:num_iter+1
    % check computation of Pi
      params = experiments{i}.params;
      tmp(t)= mean( vec(direct_dlyap(params{t}.A, params{t}.Q) - params{t}.Q0).^2 );
      subplot(1,2,1), plot(tmp)
      subplot(1,2,2), plot(perfs_yy(:,i))
    end
  pause
end

%%
figure; 
exp_id = 6;
for i = 165:200, 
    subplot(2,2,1), 
    imagesc(experiments{exp_id}.params{i}.A), 
    subplot(2,2,2), 
    imagesc(experiments{exp_id}.params{i}.C), 
    title(num2str(i)),
    subplot(2,2,3), 
    imagesc(experiments{exp_id}.params{i}.Q), 
    subplot(2,2,4), 
    imagesc(experiments{exp_id}.params{i}.R), 
    title(num2str(i)), 
    pause, 
end

%%
i = 3; 
figure; 
subplot(2,1,1), 
plot(experiments{i}.corrs_stitched, 'k'), 
subplot(2,1,2), 
plot(experiments{i}.corrs_observed, 'r')

%%
% make a pretty figure from our high-dimensional SSID algorithm fit to
% data with nonlinear dynamcis
clear all
close all
pre_path = '/home/mackelab/Desktop/Projects/Stitching/code/le_stitch/python//fits/test_SSID_gain_from_lags/';

%%
disp('p = 50, T = 500')
clearvars -except pre_path
ttl = 'p105n3T500snr05';
ssa = zeros(10,4);
for run = 0:9
    data_path = [pre_path, ttl, '/sgd_run', num2str(run), '/'];
    load([data_path, '_m.mat'])

    C = pars_true.C;
    ssa(run+1,:) = [subspace(C, pars_est1.C), subspace(C, pars_est5.C), ...
                    subspace(C, pars_pca.C), subspace(C, pars_fa.C)];
    clearvars -except pre_path run ttl ssa
end
disp(ssa)
ssa = ssa / (pi/2);

[~, idx] = sort(ssa(:,1));
idx = 1:size(ssa,1);
figure;
bar((1:size(ssa,1))-0.15, ssa(idx,3), 0.7, 'g')
hold on
bar(1:size(ssa,1), ssa(idx,1), 0.7)
bar((1:size(ssa,1))+0.15, ssa(idx,2), 0.7, 'r')
box off
set(gca, 'xlim', [0.5, size(ssa,1)+0.5])
set(gca, 'TickDir', 'out')
xlabel('# sim')
ylabel('norm. subspace angle C_{true} vs. C_{est}')
legend('PCA', '1 lag', '5 lags', 'location', 'NorthWest')
title(ttl)



%%
disp('p = 50, T = 500')
clearvars -except pre_path
ttl = 'p50n3T500snr05';
ssa = zeros(10,4);
for run = 0:9
    data_path = [pre_path, ttl, '/sgd_run', num2str(run), '/'];
    load([data_path, '_m.mat'])

    C = pars_true.C;
    ssa(run+1,:) = [subspace(C, pars_est1.C), subspace(C, pars_est5.C), ...
                    subspace(C, pars_pca.C), subspace(C, pars_fa.C)];
    clearvars -except pre_path run ttl ssa
end
disp(ssa)
ssa = ssa / (pi/2);

[~, idx] = sort(ssa(:,1));
%idx = 1:size(ssa,1);
figure;
bar((1:size(ssa,1))-0.15, ssa(idx,3), 0.7, 'g')
hold on
bar(1:size(ssa,1), ssa(idx,1), 0.7)
bar((1:size(ssa,1))+0.15, ssa(idx,2), 0.7, 'r')
box off
set(gca, 'xlim', [0.5, size(ssa,1)+0.5])
set(gca, 'TickDir', 'out')
xlabel('# sim')
ylabel('norm. subspace angle C_{true} vs. C_{est}')
legend('PCA', '1 lag', '5 lags', 'location', 'NorthWest')
title(ttl)

%%
disp('p = 20, T = 500')
clearvars -except pre_path
ttl = 'p20n3T500snr05';
ssa = zeros(10,4);
for run = 0:9
    data_path = [pre_path, ttl, '/sgd_run', num2str(run), '/'];
    load([data_path, '_m.mat'])

    C = pars_true.C;
    ssa(run+1,:) = [subspace(C, pars_est1.C), subspace(C, pars_est5.C), ...
                    subspace(C, pars_pca.C), subspace(C, pars_fa.C)];
    clearvars -except pre_path run ttl ssa
end
disp(ssa)
ssa = ssa / (pi/2);

[~, idx] = sort(ssa(:,1));
%idx = 1:size(ssa,1);
figure;
bar((1:size(ssa,1))-0.15, ssa(idx,3), 0.7, 'g')
hold on
bar(1:size(ssa,1), ssa(idx,1), 0.7)
bar((1:size(ssa,1))+0.15, ssa(idx,2), 0.7, 'r')
box off
set(gca, 'xlim', [0.5, size(ssa,1)+0.5])
set(gca, 'TickDir', 'out')
xlabel('# sim')
ylabel('norm. subspace angle C_{true} vs. C_{est}')
legend('PCA', '1 lag', '5 lags', 'location', 'NorthWest')
title(ttl)
%%


disp('p = 5, T = 200')
clearvars -except pre_path
ttl = 'p5n2T200snr05';
ssa = zeros(10,4);
for run = 0:19
    data_path = [pre_path, ttl, '/sgd_run', num2str(run), '/'];
    load([data_path, '_m.mat'])

    C = pars_true.C;
    ssa(run+1,:) = [subspace(C, pars_est1.C), subspace(C, pars_est5.C), ...
                    subspace(C, pars_pca.C), subspace(C, pars_fa.C)];
    clearvars -except pre_path run ttl ssa
end
disp(ssa)
ssa = ssa / (pi/2);

[~, idx] = sort(ssa(:,1));
idx = 1:size(ssa,1);
figure;
bar((1:size(ssa,1))-0.15, ssa(idx,3), 0.7, 'g')
hold on
bar(1:size(ssa,1), ssa(idx,1), 0.7)
bar((1:size(ssa,1))+0.15, ssa(idx,2), 0.7, 'r')
box off
set(gca, 'xlim', [0.5, size(ssa,1)+0.5])
set(gca, 'TickDir', 'out')
xlabel('# sim')
ylabel('norm. subspace angle C_{true} vs. C_{est}')
legend('PCA', '1 lag', '5 lags', 'location', 'NorthWest')
title(ttl)
%%

disp('p = 5, T = 50')
clearvars -except pre_path
ttl = 'p5n2T50snr05';
ssa = zeros(10,4);
for run = 0:9
    data_path = [pre_path, ttl, '/run', num2str(run), '/'];
    load([data_path, '_m.mat'])

    C = pars_true.C;
    ssa(run+1,:) = [subspace(C, pars_est1.C), subspace(C, pars_est5.C), ...
                    subspace(C, pars_pca.C), subspace(C, pars_fa.C)];
    clearvars -except pre_path run ttl ssa
end
disp(ssa)
ssa = ssa / (pi/2);

[~, idx] = sort(ssa(:,1));
figure;
bar((1:10)-0.15, ssa(idx,3), 0.7, 'g')
hold on
bar(1:10, ssa(idx,1), 0.7)
bar((1:10)+0.15, ssa(idx,2), 0.7, 'r')
box off
set(gca, 'xlim', [0.5, size(ssa,1)+0.5])
set(gca, 'TickDir', 'out')
xlabel('# sim')
ylabel('norm. subspace angle C_{true} vs. C_{est}')
legend('PCA', '1 lag', '5 lags', 'location', 'NorthWest')
title(ttl)

%%

disp('p = 50, T = 100')
clearvars -except pre_path
ttl = 'p50n2T100snr05';
ssa = zeros(5,4);
for run = 0:4
    data_path = [pre_path, ttl, '/run', num2str(run), '/'];
    load([data_path, '_m.mat'])

    C = pars_true.C;
    ssa(run+1,:) = [subspace(C, pars_est1.C), subspace(C, pars_est5.C), ...
                    subspace(C, pars_pca.C), 0];
    clearvars -except pre_path run ttl ssa
end
disp(ssa)
ssa = ssa / (pi/2);

[~, idx] = sort(ssa(:,1));
figure;
bar((1:5)-0.15, ssa(idx,3), 0.7, 'g')
hold on
bar(1:5, ssa(idx,1), 0.7)
bar((1:5)+0.15, ssa(idx,2), 0.7, 'r')
box off
set(gca, 'xlim', [0.5, size(ssa,1)+0.5])
set(gca, 'TickDir', 'out')
xlabel('# sim')
ylabel('norm. subspace angle C_{true} vs. C_{est}')
legend('PCA', '1 lag', '5 lags', 'location', 'NorthWest')
title(ttl)

%%
disp('p = 105, T = 200')
clearvars -except pre_path
ttl = 'p105n10T100snr05';
ssa = zeros(4,4);
for run = 0:3
    data_path = [pre_path, ttl, '/run', num2str(run), '/'];
    load([data_path, '_m.mat'])

    C = pars_true.C;
    ssa(run+1,:) = [subspace(C, pars_est1.C), subspace(C, pars_est5.C), ...
                    subspace(C, pars_pca.C), 0];
    clearvars -except pre_path run ttl ssa
end
disp(ssa)
ssa = ssa / (pi/2);

[~, idx] = sort(ssa(:,1));
figure;
bar((1:size(ssa,1))-0.15, ssa(idx,3), 0.7, 'g')
hold on
bar(1:size(ssa,1), ssa(idx,1), 0.7)
bar((1:size(ssa,1))+0.15, ssa(idx,2), 0.7, 'r')
box off
set(gca, 'xlim', [0.5, size(ssa,1)+0.5])
set(gca, 'TickDir', 'out')
xlabel('# sim')
ylabel('norm. subspace angle C_{true} vs. C_{est}')
legend('PCA', '1 lag', '5 lags', 'location', 'NorthWest')
title(ttl)



abs_path_to_data = '/home/mackelab/Desktop/Projects/Stitching/results/cosyne_poster/';
exps = 'illustration_1';

cd([abs_path_to_data, exps])

list = what('/fits'); list = list.mat;

for i = 1:10
    load(['fits/params_idx0_LDS_save_idx', num2str(i-1), '.npz.mat'])
    
    if ifBroken
        figure;
        text(0,0, 'run broke!')
        
    else
        T,p = size(y);
        obs_scheme.sub_pops = {0:4, 4:8};
        obs_scheme.obs_pops = [0, 1];
        obs_scheme.obs_time = [fix(T/2)-1, T-1];    

        make_fitting_result_overview_plots(y,x,u,truePars,estPars,initPars,Pi,Pi_h,obs_scheme)
        pause
    end
   close all
end

%%

for i = 1:10
    load(['fits/params_flip_idx0_LDS_save_idx', num2str(i-1), '.npz.mat'])
    
    T,p = size(y);
    obs_scheme.sub_pops = {0:4, 4:8};
    obs_scheme.obs_pops = [0, 1];
    obs_scheme.obs_time = [fix(T/2)-1, T-1];    

    make_fitting_result_overview_plots(y,x,u,truePars,estPars,initPars,Pi,Pi_h,obs_scheme)
    pause
   close all
end

%%

for i = 1:10
    for rep = 0:9
    load(['fits/random_idx', num2str(rep), '_LDS_save_idx', num2str(i-1), '.npz.mat'])
    
    if ifBroken
        figure;
        text(0,0, 'run broke!')
        
    else
        T,p = size(y);
        obs_scheme.sub_pops = {0:4, 4:8};
        obs_scheme.obs_pops = [0, 1];
        obs_scheme.obs_time = [fix(T/2)-1, T-1];    

        make_fitting_result_overview_plots(y,x,u,truePars,estPars,initPars,Pi,Pi_h,obs_scheme)
        pause
    end
    end
   close all
end


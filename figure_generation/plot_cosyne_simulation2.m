
        
%%
addpath(genpath('/home/mackelab/Desktop/Projects/Stitching/code/code_le_stitch/figure_generation'))

abs_path_to_data = '/home/mackelab/Desktop/Projects/Stitching/results/cosyne_poster/';
exps = 'simulation_2';


cd([abs_path_to_data, exps])


% teach this script about the problem setup (obs_scheme didn't safe correctly...)

fracs = zeros(36,3);
frac_obs = 0.1:0.1:0.9;
idx = 0;
for i =1:length(frac_obs)
    frac_pop1 = 0.1:0.1:frac_obs(i)-0.1;
    for j=1:length(frac_pop1)
        idx = idx + 1;
        fracs(idx,:) = [frac_pop1(j), frac_obs(i), 1];
    end
end

p = 50;
obs_scheme.sub_pops = {0:15, 14:29, [0,29:49]};
obs_scheme.obs_pops = [0,1,2];

idx_stitched = true(p,p);
for i = 1:length(obs_scheme.sub_pops)
    idx_stitched(obs_scheme.sub_pops{i}+1,obs_scheme.sub_pops{i}+1) = false;
end

idx_stitched12 = true(p,p);
for i = 1:length(obs_scheme.sub_pops)
    idx_stitched12(obs_scheme.sub_pops{i}+1,obs_scheme.sub_pops{i}+1) = false;
end
idx_stitched12(:,obs_scheme.sub_pops{end}+1) = false;
idx_stitched12(obs_scheme.sub_pops{end}+1,:) = false;

%% set up extent of simulation

list = what('/fits'); list = list.mat;

initialisers = {'random', 'params', 'params_flip', 'params_naive', ...
                'params_naive_flip'};
num_repets_per_initialiser = [5,1,1,1,1];
performances_init = {zeros(size(fracs,1),num_repets_per_initialiser(1)), ...
                     zeros(size(fracs,1),num_repets_per_initialiser(2)), ...
                     zeros(size(fracs,1),num_repets_per_initialiser(3)), ...
                     zeros(size(fracs,1),num_repets_per_initialiser(4)), ...
                     zeros(size(fracs,1),num_repets_per_initialiser(5))};
performances_fit  = {zeros(size(fracs,1),num_repets_per_initialiser(1)), ...
                     zeros(size(fracs,1),num_repets_per_initialiser(2)), ...
                     zeros(size(fracs,1),num_repets_per_initialiser(3)), ...
                     zeros(size(fracs,1),num_repets_per_initialiser(4)), ...
                     zeros(size(fracs,1),num_repets_per_initialiser(5))};

init_idx = 1;
initialiser = initialisers{init_idx};

%%
for i = [7]
 for init_idx = 1:length(initialisers)
  initialiser = initialisers{init_idx};
  disp(initialiser)
  load(['fits/', initialiser, '_rep0_repet0_LDS_save_idx', num2str(i-1), '.npz.mat'])  
  covy_true = cov(y);  
  num_repets = num_repets_per_initialiser(init_idx);
  for rep = 1:size(fracs,1)
    for repet = 1:num_repets
        loadfile=['fits/', initialiser, '_rep', num2str(rep-1), '_repet', num2str(repet-1), '_LDS_save_idx', num2str(i-1), '.npz.mat'];
        load(loadfile)
        disp(loadfile)

        if ifBroken
            disp('run broke!')
            performances_init{init_idx}(rep,repet) = NaN;
            performances_fit{init_idx}(rep,repet) = NaN;
        else
            T,p = size(y);
            obs_scheme.obs_time = round(double(T)*fracs(rep,:));
            if strcmp(initialiser, 'random')
                covy_init = initPars.C * direct_dlyap( initPars.A, initPars.Q) * initPars.C' + diag(initPars.R);                
            else
                covy_init = initPars.C * initPars.Pi * initPars.C' + diag(initPars.R);
            end
            covy_est  =  estPars.C * direct_dlyap( estPars.A, estPars.Q) *  estPars.C' + estPars.R;

            tmp = corrcoef(covy_init(idx_stitched12), covy_true(idx_stitched12));
            performances_init{init_idx}(rep,repet) = tmp(1,2);
            tmp = corrcoef(covy_est(idx_stitched12), covy_true(idx_stitched12));
            performances_fit{init_idx}(rep,repet) = tmp(1,2);

            %make_fitting_result_overview_plots(y,x,u,truePars,estPars,initPars,Pi,Pi_h,obs_scheme)
            %pause
        end
    end
  end
 end
end

%%
base = zeros(2,3);
base(:,1) = [0;1]; base(:,2) = [1;-1]/sqrt(2); base(:,3) = [-1;-1]/sqrt(2);
num_bins = 10;
clrs = jet(num_bins);
edges = linspace(-1,1,num_bins);

howToCombine = 'average';


for init_idx = 1:length(initialisers)
    figure;
    disp(['initialiser: ', initialisers{init_idx}])

%     subplot(1,2,1)
%     for i = 1:size(fracs,1)
%         switch howToCombine
%             case 'average'
%                 clr = find(histc(mean(performances_init{init_idx}(i, :),2), edges));
%             case 'max'
%                 clr = find(histc(max(performances_init{init_idx}(i, :),[],2), edges));                
%         end
%         tmp = base * [fracs(i,1),diff(fracs(i,:))]';
%         if ~isempty(clr)
%             plot(tmp(1), tmp(2), 'o', 'color', clrs(clr,:))
%             hold on
%         end
%         title('SSID fits')
%     end

    edges_color = -1:0.01:1;
    m = Inf;
    M = -Inf;

    X = zeros(size(fracs,1),1);
    Y = zeros(size(fracs,1),1);
    V_init = zeros(size(fracs,1),1);
    subplot(1,2,1)
    for i = 1:size(fracs,1)
        switch howToCombine
            case 'average'
                clr = mean(performances_init{init_idx}(i, :),2);
            case 'best'
                clr = max(performances_init{init_idx}(i, :),[],2);                
            case 'worst'
                clr = min(performances_init{init_idx}(i, :),[],2);                
        end
        rel_durs = [fracs(i,1),diff(fracs(i,:))];
        %rel_durs = rel_durs([]);
        tmp = base * rel_durs';
        X(i) = tmp(1);
        Y(i) = tmp(2);
        if ~isempty(clr)
            V_init(i) = clr;
        else
            V_init(i) = 0;
        end
    end
    [xq, yq] = meshgrid(-1:0.005:1, -1:0.005:1); 
    vq = griddata(X,Y,V_init,xq,yq); 
    surf(xq,yq,vq, 'EdgeColor', 'None', 'facecolor', 'interp'); 
    view(2); axis off; colorbar
    title('initialisers')
    m = min(m, min(vq(:)));
    M = max(M, max(vq(:)));
    V_fit = zeros(size(fracs,1),1);
    subplot(1,2,2)
    for i = 1:size(fracs,1)
        switch howToCombine
            case 'average'
                clr = mean(performances_fit{init_idx}(i, :),2);
            case 'best'
                clr = max(performances_fit{init_idx}(i, :),[],2);                
            case 'worst'
                clr = min(performances_fit{init_idx}(i, :),[],2);                
        end
        if ~isempty(clr)
            V_fit(i) = clr;
        else
            V_fit(i) = 0;
        end
    end
    [xq, yq] = meshgrid(-1:0.005:1, -1:0.005:1); 
    vq = griddata(X,Y,V_fit,xq,yq); 
    surf(xq,yq,vq, 'EdgeColor', 'None', 'facecolor', 'interp'); 
    m = min(m, min(vq(:)));
    M = max(M, max(vq(:)));    
    view(2); axis off; colormap('hot')
    colorbar
    title('EM fits')
    
    subplot(1,2,1), set(gca, 'clim', [m,M])
    subplot(1,2,2), set(gca, 'clim', [m,M])
    pause
    %close all
end

%%
figure
vq = griddata(X,Y,fracs(:,2),xq,yq); 
surf(xq,yq,vq, 'EdgeColor', 'None', 'facecolor', 'interp'); 
view(2); axis off; colorbar
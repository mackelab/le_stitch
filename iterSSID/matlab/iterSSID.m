function [params, SIGfp_new, Psig_fp] = iterSSID(seq, xDim, num_iter,SIGfp_f)

[yDim,~] = size(seq.y);
params = cell(num_iter,1);

if length(seq.subpops) == 2
    idx_overlap = intersect(seq.subpops{1},seq.subpops{2});
else
    error('oops')
end

% generate NaN-masked Hankel covariances
[SIGfp,SIGff,~] = generateCovariancesFP(seq,xDim);
dSIGyy  = diag(diag(SIGff(1:yDim,1:yDim))); % ssidSVD only needs diagonal
lPsig_fp = ~isnan(SIGfp);    % get Hankel-sized '
Psig_fp = double(lPsig_fp);  % observation' mask
lnPsig_fp = ~lPsig_fp; 
PSIGfp = SIGfp(lPsig_fp); % = the 'good' part of SIGfp

% 'zeroth' iteration: use ssidSVD on Hankel-matrix with NaN set to zero:
SIGfp_new = SIGfp; SIGfp_new(~logical(Psig_fp)) = 0; 
[params{1}.A, params{1}.C, params{1}.Q, params{1}.R, params{1}.Q0]=ssidSVD(SIGfp_new,dSIGyy,xDim);
params{1} = post_process(params{1}); 

% main iterations: fill in NaN's iteratively using missing-value-SVDs:
if nargin < 4 || isempty(SIGfp_f)
    SIGfp_old = SIGfp_new;
else
    SIGfp_old = SIGfp_f;
end
alpha = 1;
for t = 2:num_iter+1
    %SIGfp_new= soft_impute(SIGfp,xDim, 0, Psig_fp, 10e-10, 2000, SIGfp_old, true);      
    
    [params{t}.A, params{t}.C, params{t}.Q, params{t}.R, params{t}.Q0]=ssidSVD(SIGfp_old,dSIGyy,xDim,...
          'pars_old',params{t-1}, 'idx_overlap', idx_overlap);
    params{t} = post_process(params{t}); %SIGfps{t} = SIGfp_new;   
    
    [SIGfp_new, ~, ~]= construct_hankel_cov(params{t});
    pars_tmp = params{t}; pars_tmp.C(seq.subpops{2}) = -pars_tmp.C(seq.subpops{2});
    [SIGfp_tmp, ~, ~]= construct_hankel_cov(pars_tmp);
    
    if mean( vec((SIGfp_tmp(:,idx_overlap)-SIGfp(:,idx_overlap)).^2) ) < mean( vec((SIGfp_new(:,idx_overlap)-SIGfp(:,idx_overlap)).^2) ) 
        params{t} = pars_tmp;
        SIGfp_new = SIGfp_tmp;
    end
    
    SIGfp_new(lPsig_fp) = PSIGfp; % makes sure we keep the 'good' part 'good'
    SIGfp_new(lnPsig_fp) = alpha * SIGfp_new(lnPsig_fp) + (1-alpha) * SIGfp_old(lnPsig_fp);
    SIGfp_old = SIGfp_new;
end
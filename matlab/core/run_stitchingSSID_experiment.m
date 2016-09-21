function [params, corrs_stitched, corrs_observed, SIGfps, SIGyyl_f, PSIG] = run_stitchingSSID_experiment(y, xDim, num_iter, num_subpops, overlap, if_plot)

if nargin < 6
    if_plot = false;
end
if nargin < 5
    overlap = 1;
end
if nargin < 4
    num_subpops = 2;
end

[yDim,T] = size(y);
disp([yDim,T])

%% get comparison against fully-observed model
seq_f.y = y;
[SIGfp_f, SIGff_f, ~] = generateCovariancesFP(seq_f,xDim);
SIGyy_f  = SIGff_f(1:yDim,1:yDim);

[params_f.A, params_f.C, params_f.Q, params_f.R, params_f.Q0]=ssidSVD(SIGfp_f,SIGyy_f,xDim);
if any(~isreal(params_f.Q))
    params_f.Q=real(params_f.Q);
    %!!! prob want to throw an error message here
end
params_f = post_process(params_f);

%% start stitching iterative-SVD-SSID

% mask data for covariance generation
ymasked = NaN(size(y));
if num_subpops==2
    seq.subpops{1} =  1:(fix(yDim/2) + ceil(overlap/2));
    seq.subpops{2} = (fix(yDim/2)+1-floor(overlap/2)):yDim;
    seq.y = y;
    ymasked(seq.subpops{1},fix(T/2)+1:end)=seq.y(seq.subpops{1},fix(T/2)+1:end);
    ymasked(seq.subpops{2},1:fix(T/2))=seq.y(seq.subpops{2}, 1:fix(T/2));
    ymasked(:,fix(T/2)+(-xDim:xDim)) = NaN;
elseif num_subpops == 1
    seq.subpops{1} = 1:yDim;
    ymasked = seq.y;
else
    error('need to generalize script to more than 2 subpopulations')
end
seq.y = ymasked;
seq.T = size(seq.y,2);

% compute non-observation mask for covariance matrix
idx_stitched = true(yDim);
for i = 1:length(seq.subpops)
    idx_stitched(seq.subpops{i},seq.subpops{i}) = false;
end


[params, SIGfps, PSIG] = iterSSID(seq, xDim, num_iter);

% gather 'performance' information
SIGyyl_f = SIGfp_f(1:yDim,1:yDim);
corrs_stitched = zeros(num_iter+1,2);
corrs_observed = zeros(num_iter+1,2);
for t = 1:num_iter+1
    [SIGfp, SIGyy, ~]= construct_hankel_cov(params{t});    
     SIGyyl = SIGfp(1:yDim,1:yDim);
    corrs_stitched(t,1) = corr(SIGyy(idx_stitched),  SIGyy_f(idx_stitched));
    corrs_stitched(t,2) = corr(SIGyyl(idx_stitched), SIGyyl_f(idx_stitched));
    corrs_observed(t,1) = corr(SIGyy(~idx_stitched), SIGyy_f(~idx_stitched));
    corrs_observed(t,2) = corr(SIGyyl(~idx_stitched),SIGyyl_f(~idx_stitched));    
end


%%
if if_plot
    figure('units','centimeters','position',[15,10,30,20]);
    
    subplot(2,3,1)
    params_h = params{end};
    covyff = params_h.C * direct_dlyap(params_h.A,params_h.Q) * params_h.C' + params_h.R;
    covyfp = params_h.C * params_h.A * direct_dlyap(params_h.A,params_h.Q) * params_h.C';
    imagesc([[covyff,covyfp']; [covyfp,covyff]])
    
    line(yDim+[.5,.5], .5+[0, seq.subpops{1}(end)], 'color', 'r', 'linewidth', 2)
    line(seq.subpops{1}(end)+[.5,.5], .5+[0, seq.subpops{1}(end)], 'color', 'r', 'linewidth', 2)
    line(.5+[seq.subpops{1}(end),yDim], .5+[0, 0], 'color', 'r', 'linewidth', 2)
    line(.5+[seq.subpops{1}(end),yDim], .5+seq.subpops{1}(end)+[0, 0], 'color', 'r', 'linewidth', 2)

    line(.5+[0, seq.subpops{1}(end)], yDim+[.5,.5], 'color', 'r', 'linewidth', 2)
    line(.5+[0, seq.subpops{1}(end)], seq.subpops{1}(end)+[.5,.5], 'color', 'r', 'linewidth', 2)
    line(.5+[0, 0], .5+[seq.subpops{1}(end),yDim], 'color', 'r', 'linewidth', 2)
    line(.5+seq.subpops{1}(end)+[0, 0], .5+[seq.subpops{1}(end),yDim], 'color', 'r', 'linewidth', 2)

%     line(.5+[0, seq.subpops{1}(end)], seq.subpops{1}(end)+[.5,.5], 'color', 'r', 'linewidth', 2)
%     line(.5+[0, seq.subpops{1}(end)], seq.subpops{1}(end)+[.5,.5], 'color', 'r', 'linewidth', 2)
%     line(.5+[0, seq.subpops{1}(end)], yDim+[.5,.5], 'color', 'r', 'linewidth', 2)
%     line(.5+[0, seq.subpops{1}(end)], yDim+seq.subpops{1}(end)+[.5,.5], 'color', 'r', 'linewidth', 2)
%     line(.5+[0, 0], .5+[seq.subpops{1}(end),yDim], 'color', 'r', 'linewidth', 2)
%     line(.5+seq.subpops{1}(end)+[0, 0], .5+[seq.subpops{1}(end),yDim], 'color', 'r', 'linewidth', 2)
%     line(.5+[0, 0], .5+[seq.subpops{1}(end),yDim]+yDim, 'color', 'r', 'linewidth', 2)
%     line(.5+seq.subpops{1}(end)+[0, 0], .5+[seq.subpops{1}(end),yDim]+yDim, 'color', 'r', 'linewidth', 2)
    
    line(2*yDim+[.5,.5], [0, seq.subpops{1}(end)], 'color', 'r')
    title('cov(y_t, y_{t-1}) est. from partial obs.')
    subplot(2,3,2)
    covy_full_e = cov([y(:,2:end); y(:,1:end-1)]');
    covyff_e = covy_full_e(1:yDim,1:yDim);
    imagesc(covy_full_e)
    line(yDim+[.5,.5], .5+[0, seq.subpops{1}(end)], 'color', 'r', 'linewidth', 2)
    line(seq.subpops{1}(end)+[.5,.5], .5+[0, seq.subpops{1}(end)], 'color', 'r', 'linewidth', 2)
    line(.5+[seq.subpops{1}(end),yDim], .5+[0, 0], 'color', 'r', 'linewidth', 2)
    line(.5+[seq.subpops{1}(end),yDim], .5+seq.subpops{1}(end)+[0, 0], 'color', 'r', 'linewidth', 2)

    line(.5+[0, seq.subpops{1}(end)], yDim+[.5,.5], 'color', 'r', 'linewidth', 2)
    line(.5+[0, seq.subpops{1}(end)], seq.subpops{1}(end)+[.5,.5], 'color', 'r', 'linewidth', 2)
    line(.5+[0, 0], .5+[seq.subpops{1}(end),yDim], 'color', 'r', 'linewidth', 2)
    line(.5+seq.subpops{1}(end)+[0, 0], .5+[seq.subpops{1}(end),yDim], 'color', 'r', 'linewidth', 2)
    title('cov(y_t, y_{t-1}) emp. from partial obs.')
    subplot(2,3,3)
    plot(covyff(logical(idx_stitched)), covyff_e(logical(idx_stitched)),'.')
    corrcoef(covyff(logical(idx_stitched)), covyff_e(logical(idx_stitched)))
    xlabel('est.')
    ylabel('emp.')
    
    subplot(2,3,4)
    covyff = params_f.C * direct_dlyap(params_f.A,params_f.Q) * params_f.C' + params_f.R;
    covyfp = params_f.C * params_f.A * direct_dlyap(params_f.A,params_f.Q) * params_f.C';
    imagesc([[covyff,covyfp']; [covyfp,covyff]])
    line(yDim+[.5,.5], .5+[0, seq.subpops{1}(end)], 'color', 'r', 'linewidth', 2)
    line(seq.subpops{1}(end)+[.5,.5], .5+[0, seq.subpops{1}(end)], 'color', 'r', 'linewidth', 2)
    line(.5+[seq.subpops{1}(end),yDim], .5+[0, 0], 'color', 'r', 'linewidth', 2)
    line(.5+[seq.subpops{1}(end),yDim], .5+seq.subpops{1}(end)+[0, 0], 'color', 'r', 'linewidth', 2)

    line(.5+[0, seq.subpops{1}(end)], yDim+[.5,.5], 'color', 'r', 'linewidth', 2)
    line(.5+[0, seq.subpops{1}(end)], seq.subpops{1}(end)+[.5,.5], 'color', 'r', 'linewidth', 2)
    line(.5+[0, 0], .5+[seq.subpops{1}(end),yDim], 'color', 'r', 'linewidth', 2)
    line(.5+seq.subpops{1}(end)+[0, 0], .5+[seq.subpops{1}(end),yDim], 'color', 'r', 'linewidth', 2)
    title('cov(y_t, y_{t-1}) est. from full obs.')
    subplot(2,3,5)
    covy_full_e = cov([y(:,2:end); y(:,1:end-1)]');
    covyff_e = covy_full_e(1:yDim,1:yDim);
    imagesc(covy_full_e)
    line(yDim+[.5,.5], .5+[0, seq.subpops{1}(end)], 'color', 'r', 'linewidth', 2)
    line(seq.subpops{1}(end)+[.5,.5], .5+[0, seq.subpops{1}(end)], 'color', 'r', 'linewidth', 2)
    line(.5+[seq.subpops{1}(end),yDim], .5+[0, 0], 'color', 'r', 'linewidth', 2)
    line(.5+[seq.subpops{1}(end),yDim], .5+seq.subpops{1}(end)+[0, 0], 'color', 'r', 'linewidth', 2)

    line(.5+[0, seq.subpops{1}(end)], yDim+[.5,.5], 'color', 'r', 'linewidth', 2)
    line(.5+[0, seq.subpops{1}(end)], seq.subpops{1}(end)+[.5,.5], 'color', 'r', 'linewidth', 2)
    line(.5+[0, 0], .5+[seq.subpops{1}(end),yDim], 'color', 'r', 'linewidth', 2)
    line(.5+seq.subpops{1}(end)+[0, 0], .5+[seq.subpops{1}(end),yDim], 'color', 'r', 'linewidth', 2)
    title('cov(y_t, y_{t-1}) emp. from full obs.')
    subplot(2,3,6)
    plot(covyff(logical(idx_stitched)), covyff_e(logical(idx_stitched)),'.')
    corrcoef(covyff(logical(idx_stitched)), covyff_e(logical(idx_stitched)))
    xlabel('est.')
    ylabel('emp.')
    
end

end
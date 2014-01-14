function stats=get_sum_stats(seq,maxlag,maxlag_diag);


yall=[seq.y];
ydim=size(yall,1);

%mean firing rate and max firing rate for each neuron:
%
for k=1:size(yall,1);
    stats.yhist(k,:)=hist(yall(k,:),[0:20]);
    stats.yhist(k,:)=stats.yhist(k,:)/sum(stats.yhist(k,:));
end

stats.ymean=mean(yall,2);
stats.ymax=max(yall,[],2);
stats.yvar=var(yall,[],2);

%now, get population spike count statistics:
stats.poprate=sum(yall,1);
%mean spike count:
stats.poprate_mean=mean(stats.poprate);
%variance of spike counts:
stats.poprate_var=var(stats.poprate);

%histogram of spike-counts
stats.poprate_bins=[0:1:200];
[stats.poprate_hist,stats.poprate_bins]=hist(stats.poprate,stats.poprate_bins);
stats.poprate_hist=stats.poprate_hist/sum(stats.poprate_hist);
stats.poprate_hist_log=log10(max(stats.poprate_hist,1/numel(stats.poprate)));

%conditional histograms of spike-counts (condition on previous bin being
%smaller than median-- does not correct for edge-effects!!!!! 
stats.poprate_median=median(stats.poprate);
poprate_cond=stats.poprate(2:end);
poprate_cond=poprate_cond(stats.poprate(1:end-1)<stats.poprate_median);
[stats.poprate_hist_cond_smallprevious]=hist(poprate_cond,stats.poprate_bins);
stats.poprate_hist_cond_smallprevious=stats.poprate_hist_cond_smallprevious/sum(stats.poprate_hist_cond_smallprevious);
stats.poprate_hist_cond_smallprevious_log=log10(max(stats.poprate_hist_cond_smallprevious,2/numel(stats.poprate)));


poprate_cond=stats.poprate(2:end);
poprate_cond=poprate_cond(stats.poprate(1:end-1)>=stats.poprate_median);
[stats.poprate_hist_cond_bigprevious]=hist(poprate_cond,stats.poprate_bins);
stats.poprate_hist_cond_bigprevious=stats.poprate_hist_cond_bigprevious/sum(stats.poprate_hist_cond_bigprevious);
stats.poprate_hist_cond_bigprevious_log=log10(max(stats.poprate_hist_cond_bigprevious,2/numel(stats.poprate)));


%now that we are done with the mean-calculations, move on to pairwise
%correlations: 
%first, get raw covariance and correlation of the data:
stats.total_cov_instant=cov(yall');
stats.total_corr_instant=corrcoef(yall');

a=mean(yall,2);
b=std(yall,1,2);
yall_minusmean=bsxfun(@minus,yall,a);
clear yall;
yall_normed=bsxfun(@rdivide,yall_minusmean,b);
clear yall_minusmean
Tall=size(yall_normed,2);


%now go for time-lagged correlations:

%keyboard
for k=0:maxlag_diag;
    kind=k+1;
    kind_diag=k+1;
    yall_normed_lagged=circshift(yall_normed,[0,k]);
    corro=yall_normed*yall_normed_lagged'/Tall;
    if abs(k)<=maxlag
        stats.total_corr_lag(:,:,kind)=corro;
    end
        stats.total_corr_diag_lag(:,kind_diag)=diag(corro);
end
stats.total_cov_diag_lag=bsxfun(@times, stats.total_corr_diag_lag,diag(stats.total_cov_instant));
stats.total_cov_lag=bsxfun(@times, stats.total_corr_lag,diag(stats.total_cov_instant));

clear yall_normed yall_normed_lagged
%stats.total_corr_check=yall_normed*yall_normed'/Tall;
%keyboard




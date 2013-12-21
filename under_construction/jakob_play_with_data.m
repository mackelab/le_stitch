clear all
close all

warning off
%load('popsim_realparams');
%realseqtest.y=real_y;
%realseq.h=real_h';
%realseq.x=real_x;

%load('popsim_realparams_nostim');
%realnostimseq.y=nostim_y;
%realnostimseq.h=real_h'*0;
%realnostimseq.x=nostim_x;

load('popsim_real_repeatstimulus');
PSTH=mean(Y,3);
realrepnoise=bsxfun(@minus,Y,PSTH);
realrepnoise=reshape(realrepnoise,size(realrepnoise,1),size(realrepnoise,2)*size(realrepnoise,3));
realrepseq.y=realrepnoise;
realrepsignalseq.y=PSTH;
realseq.y=reshape(Y,size(realrepnoise,1),size(realrepnoise,2)*size(realrepnoise,3));


load('popsim_fit_repeatstimulus');
PSTH=mean(Y,3);
fitrepnoise=bsxfun(@minus,Y,PSTH);
fitrepnoise=reshape(fitrepnoise,size(fitrepnoise,1),size(fitrepnoise,2)*size(fitrepnoise,3));
fitrepseq.y=fitrepnoise;
fitrepsignalseq.y=PSTH;
fitseq.y=reshape(Y,size(fitrepnoise,1),size(fitrepnoise,2)*size(fitrepnoise,3));

%load('popsim_fit_nostimulus');

warning on

%realseq=load('popsim_realparams');
%fitnostimseq=
%load('popsim_fitparams_nostim');
%fitnostimseq.y=nostim_y;
[ydim,T]=size(realseq.y);
%[hdim,T]=size(realseq.h);
%fitnostimseq.h=zeros(hdim,size(nostim_y,2));

maxlag=10;
maxlag_diag=20;

%fitstats=get_sum_stats(fitseq,maxlag,maxlag_diag);
realstats=get_sum_stats(realseq,maxlag,maxlag_diag);
%realnostimstats=get_sum_stats(realnostimseq,maxlag,maxlag_diag);
realrepstats=get_sum_stats(realrepseq,maxlag,maxlag_diag);
realrepsignalstats=get_sum_stats(realrepsignalseq,maxlag,maxlag_diag);
fitstats=get_sum_stats(fitseq,maxlag,maxlag_diag);
fitrepstats=get_sum_stats(fitrepseq,maxlag,maxlag_diag);
fitrepsignalstats=get_sum_stats(fitrepsignalseq,maxlag,maxlag_diag);

%fitnostimstats=get_sum_stats(fitnostimseq,maxlag,maxlag_diag);

h(1)=figure;

subplot(3,4,1)
imagesc(realstats.total_corr_instant-eye(ydim),[-.8,.8]); colorbar
title('Total correlations')

subplot(3,4,5)
imagesc(realrepsignalstats.total_corr_instant-eye(ydim),[-.8,.8]); colorbar
title('Stimulus correlations')

subplot(3,4,9)
imagesc(realrepstats.total_corr_instant-eye(ydim),[-.8,.8]); colorbar
title('Noise correlations')

subplot(3,4,2)
plot(realstats.total_corr_diag_lag'); colorbar
title('Total correlations')

subplot(3,4,6)
plot(realrepsignalstats.total_corr_diag_lag'); colorbar
title('Stimulus correlations')

subplot(3,4,10)
plot(realrepstats.total_corr_diag_lag'); colorbar
title('Noise correlations')

%%

subplot(3,4,3)
imagesc(fitstats.total_corr_instant-eye(ydim),[-.8,.8]); colorbar
title('Total correlations')

subplot(3,4,7)
imagesc(fitrepsignalstats.total_corr_instant-eye(ydim),[-.8,.8]); colorbar
title('Stimulus correlations')

subplot(3,4,11)
imagesc(fitrepstats.total_corr_instant-eye(ydim),[-.8,.8]); colorbar
title('Noise correlations')

subplot(3,4,4)
plot(fitstats.total_corr_diag_lag'); colorbar
title('Total correlations')

subplot(3,4,8)
plot(fitrepsignalstats.total_corr_diag_lag'); colorbar
title('Stimulus correlations')

subplot(3,4,12)
plot(fitrepstats.total_corr_diag_lag'); colorbar
title('Noise correlations')

%%
h(2)=figure
subplot(1,3,1)
plot(OffDiag(fitstats.total_corr_instant),OffDiag(realstats.total_corr_instant),'.')
line([-1;1],[-1;1])
axis square; axis tight;
title('Total correlation')

subplot(1,3,2)
plot(OffDiag(fitrepsignalstats.total_corr_instant),OffDiag(realrepsignalstats.total_corr_instant),'.')
title('Stimulus correlation')
line([-1;1],[-1;1])
axis square; axis tight;

subplot(1,3,3)
plot(OffDiag(fitrepstats.total_corr_instant),OffDiag(realrepstats.total_corr_instant),'.')
line([-1;1],[-1;1])
axis square; axis tight;

%%
h(3)=figure
subplot(1,3,1); hold on
plot(OffDiag(fitstats.total_cov_instant),OffDiag(realstats.total_cov_instant),'.')
title('Total covelation')
line([-10;10],[-10;10])
axis square; axis tight;

subplot(1,3,2); hold on
plot(OffDiag(fitrepsignalstats.total_cov_instant),OffDiag(realrepsignalstats.total_cov_instant),'.')
line([-10;10],[-10;10])
title('Stimulus covelation')
axis square; axis tight;

subplot(1,3,3); hold on
plot(OffDiag(fitrepstats.total_cov_instant),2.3*OffDiag(realrepstats.total_cov_instant),'.')
line([-5;5],[-5;5])
axis square; axis tight;


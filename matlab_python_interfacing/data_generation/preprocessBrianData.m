% This script serves to load simulated spiking data, as it is produced by
% the Brian simulation software, and stored as Python .p ('Pickle') files
% of binary streams. Then we do some re-arrangement of the data that comes
% as long lists of spike times (represented in seconds) into big sparse 
% matrices. 
% The Brian simulations at hand are simulated clustered networks, i.e.
% collections of inhibitory and excitatory recurrently connected cells that
% have non-uniform connectivity probabilities - the excitatory cells  form
% clusters of more tighlty connected neuron groups that may function as
% highly correlated 'super-neurons', with the excitatory population
% starting to show fairly low-dimensional population activity for large
% clustering coefficients. 

clc
clear all
close all


%% specify data set to load
dataset = '091115_v2'; % date/month/year/_version
load(['/home/mackelab/Desktop/Projects/Stitching/data/clustered_networks/raw_spikes/',dataset,'.mat'])
% loading gives two variables, ext and inh, that got translated from
% python dictionaries into matlab structures. Let's hope that nothing 
% important broke in the process.

T  = 2000000; % length of simulation, has to be verified
dt = 20;     % 

disp('CHECK THIS TIME AND TIME AGAIN:')
fprintf('total simulation length in milliseconds is %d \n',T)
fprintf('That is %d many bins at %d ms binning \n', T/dt, dt)
disp('We will assume this binning for the remainder of this data overview')
%% get summary statistics, get first overview and kick out bad cells
Ne = length(fieldnames(exc)); % number of excitatory cells
Ni = length(fieldnames(inh)); % number of inhibitory cells
N  = Ne+Ni;

fprintf('total number of excitatory cells is %d \n',Ne)
fprintf('total number of inhibitory cells is %d \n',Ni)

st = cell(Ne+Ni,1); 

for i = 0:Ne-1
    st{i+1} = exc.(['n',num2str(i)]);
end
for i = 0:Ni-1
    st{i+Ne+1} = inh.(['n',num2str(i)]);
end

fr = zeros(N,1); % firing rates can be gotten also from spike times
for i = 1:N
    fr(i) = length(st{i})/T*1000;
end

frthrs = 0.5;
fprintf('we sort out all cells with firing rate < %d Hz \n', frthrs)
idxBad_e = fr(1:Ne)<frthrs;   nbe = sum(idxBad_e);
idxBad_i = fr(Ne+1:N)<frthrs; nbi = sum(idxBad_i);


figure('Units', 'normalized','Position', [0.2,0.2,0.45,0.5]);
subplot(2,2,1)
plot(sort(fr,1,'descend'), 'k', 'linewidth', 2)
ylabel('firing rate [Hz]')
xlabel('#neuron (sorted)')
axis([0,N+1,0, 1.05*max(fr)])
box off
set(gca, 'TickDir', 'out')
title('pruning cells by raw firing rate')
text(N-0.8*(nbe+nbi), 0.8*max(fr), 'discarded')
text(0.5*(N-(nbe+nbi)), 0.8*max(fr), 'kept')
line(N-[nbe+nbi,nbe+nbi], [0, 1.05*max(fr)],'color', 'k')

st = st(~[idxBad_e;idxBad_i]); % we just outright discard the cells instead
fr = fr(~[idxBad_e;idxBad_i]); % of keeping them somewhere in the back
Ne = Ne - nbe; % 
Ni = Ni - nbi; % update (they also double as index for the e/i boundary!)
N  = Ne + Ni;  %


if frthrs == 0
 fprintf('number of excitatory cells that apppear to be dead is %d \n',nbe)
 fprintf('number of inhibitory cells that apppear to be dead is %d \n',nbi)
else
 fprintf('number of excitatory cells just sorted out is %d \n',nbe)
 fprintf('number of inhibitory cells just sorted out is %d \n',nbi)
end    
fprintf('now working with the remaining %d cells \n', N)

disp('creating dense spiking array of dim #cells-by-#bins')
disp(['Note: the representation as lists of spike times just cries out',...
      ' for sparse matrix representation, but we will need to convolve'])
X = zeros(N, ceil(T/dt));
x = (0:dt:T)/1000;
for i = 1:N
    tmp = histc(st{i}, x);              % a slightly annoying property of 
    tmp(end-1) = tmp(end-1) + tmp(end); % histc is the last bin thingy
    X(i,:) = tmp(1:end-1);
end

subplot(2,2,3:4)
imagesc(X)
xlabel('time')
ylabel('#neuron')
box off
set(gca, 'TickDir', 'out')
title(['collapsed activity, ', num2str(dt), 'ms time bins'])
line([1,1]*1.02*floor(T/dt), [0, Ne], 'color', 'b', 'linewidth', 2)
line([1,1]*1.02*floor(T/dt), [Ne+1, N], 'color', 'r', 'linewidth', 2)
axis([0,1.03*floor(T/dt),1, N]) 
colorbar

%% check major statistics of the data set

figure('Units', 'normalized','Position', [0.2,0.2,0.45,0.5]);
subplot(2,3,1)
x = linspace(min(fr(1:Ne)), max(fr(1:Ne)), 30);
h = histc(fr(1:Ne), x);
h = h/sum(h);
width = 1;
bar(x, h/sum(h), width, 'edgeColor', 'none', 'faceColor', 'b')
axis([x(1)-width*(x(2)-x(1))/2, x(end)+(x(2)-x(1))*width/2, 0, 1.1*max(h)])
box off
xlabel('Firing rate [Hz]')
ylabel('rel. frequency')
title('firing rates exc. ')

subplot(2,3,4)
corrx = corr(X(1:Ne,:)');
idxM = logical(triu(ones(Ne),1));
x = linspace(min(corrx(idxM)), max(corrx(idxM)), 50);
h = histc(corrx(idxM), x);
h = h/sum(h);
width = 1;
bar(x,h, width, 'edgeColor', 'none', 'faceColor', 'b')
xlabel('correlation')
ylabel('rel. frequency')
title('correlation strengths exc.')
box off
set(gca, 'TickDir', 'out')
axis([x(1)-width*(x(2)-x(1))/2, x(end)+(x(2)-x(1))*width/2, 0, 1.1*max(h)])

subplot(2,3,2)
x = linspace(min(fr(Ne+1:N)), max(fr(Ne+1:N)), 30);
h = histc(fr(Ne+1:N), x);
h = h/sum(h);
width = 1;
bar(x, h/sum(h), width, 'edgeColor', 'none', 'faceColor', 'r')
axis([x(1)-width*(x(2)-x(1))/2, x(end)+(x(2)-x(1))*width/2, 0, 1.1*max(h)])
box off
xlabel('Firing rate [Hz]')
ylabel('rel. frequency')
title('firing rates inh. ')

subplot(2,3,5)
corrx = corr(X(Ne+1:N,:)');
idxM = logical(triu(ones(Ni),1));
x = linspace(min(corrx(idxM)), max(corrx(idxM)), 50);
h = histc(corrx(idxM), x);
h = h/sum(h);
width = 1;
bar(x,h, width, 'edgeColor', 'none', 'faceColor', 'r')
xlabel('correlation')
ylabel('rel. frequency')
title('correlation strengths inh.')
box off
set(gca, 'TickDir', 'out')
axis([x(1)-width*(x(2)-x(1))/2, x(end)+(x(2)-x(1))*width/2, 0, 1.1*max(h)])

subplot(2,3,3)
covx = cov(X(1:Ne,:)');
imagesc(covx)
colormap('gray')
title('full covariance (exc. & inh.)')
box off
set(gca, 'TickDir', 'out')
set(gca, 'XTick', [1, Ne, N])
set(gca, 'YTick', [1, Ne, N])
line([0, N], [Ne, Ne], 'color', 'black', 'linewidth', 2)
line([Ne, Ne], [0, N], 'color', 'black', 'linewidth', 2)
line([1,1], [1, Ne], 'color', 'b', 'linewidth', 2)
line([1,Ne], [1, 1], 'color', 'b', 'linewidth', 2)
line([N,N], [Ne+1, N], 'color', 'r', 'linewidth', 2)
line([Ne+1, N], [N,N], 'color', 'r', 'linewidth', 2)

subplot(2,3,6)
L = eig(covx(1:Ne, 1:Ne));
L = L(end:-1:1);
plot(cumsum(L)/sum(L), 'b', 'linewidth', 2)
hold on
L = eig(covx(Ne+1:N, Ne+1:N));
L = L(end:-1:1);
plot(cumsum(L)/sum(L), 'r', 'linewidth', 1.5)
L = eig(covx(1:N, 1:N));
L = L(end:-1:1);
plot(cumsum(L)/sum(L), 'k', 'linewidth', 1.5)
legend('exc', 'inh', 'all', 'Location', 'Southeast')
legend boxoff
axis([0,N+1,0,1.05])
title('cumulative var. explained')
box off
set(gca, 'TickDir', 'out')
xlabel('#dimensions')
ylabel('$(\sum_{j=1}^i \lambda_j) / (\sum_{j=1}^N \lambda_j)$', ...
        'interpreter', 'latex')
%clear exc inh % let's not work with indexing structures, please

%% get first impression of how the algorithm sees this

[W,L] = eig(covx(1:Ne, 1:Ne));
L = diag(L);
L = L(end:-1:1);
W = W(:,end:-1:1);

figure('Units', 'normalized','Position', [0.2,0.2,0.3,0.6]);

subplot(3,2,1)
imagesc(covx(1:Ne, 1:Ne))
title('exh. covariance')
box off
set(gca, 'TickDir', 'out')
axis off

subplot(3,2,2)
rnk = 10;
imagesc(W(:,1:rnk)*diag(L(1:rnk))*W(:,1:rnk)')
title(sprintf('rank-%d approx.', rnk))
box off
set(gca, 'TickDir', 'out')
axis off

subplot(3,2,3:4)
PCs = (bsxfun(@minus, X(1:Ne,:), mean(X(1:Ne,:),2))' * W(:,1:rnk))';
smoothing_width = 50;
for i = 1:rnk
  PCs(i,:) = conv(PCs(i,:),ones(1,smoothing_width)/smoothing_width,'same');
end
plot(PCs')
xlabel('time')
ylabel(sprintf('PC_{1:%d}', rnk))
title(sprintf('dynamics (time-smoothed first %d PCs)', rnk))
box off
set(gca, 'TickDir', 'out')

subplot(3,2,5)
plot(sort(diag(covx(1:Ne,1:Ne))))
xlabel('#neuron (sorted)')
ylabel('spike variance')
box off
set(gca, 'TickDir', 'out')
title('est. diag($C \Pi C^T + R$)', 'interpreter', 'latex')

subplot(3,2,6)
plot(sort(mean(X(1:Ne,:),2)))
xlabel('#neuron (sorted)')
ylabel('mean firing rate')
box off
set(gca, 'TickDir', 'out')
title('est. emission offsets d')

%% generating fluorescence traces

% idx = 4;
% 
% S = [];
% S.dur      = 200;
% S.spkTimes = st{idx};
% S.recycleSpikeTimes = 1;
% S.frameRate = 50;
% OUT = modelCalcium(S,1);
% subplot(3,1,1)
% for i = 1:length(st{idx})
%     line(st{idx}(i)*[1,1], [0,5], 'color', 'r'); 
% end

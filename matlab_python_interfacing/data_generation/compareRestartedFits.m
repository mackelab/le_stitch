%

% load first run in series to compute some indexes and data statistics
dataSet  = ['LDS_save_rerun_small_run', num2str(0)];
load(['/home/mackelab/Desktop/Projects/Stitching/results/test_problems/',...
    '/', dataSet, '.mat'])
T    = size(y,2); % simulation length
Trial= size(y,3); % number of trials
xDim = size(A_h,1); % q
yDim = size(y,1);   % p
uDim = size(u,1);   % r

if ~ iscell(obsScheme.subpops)
  numPops = size(obsScheme.subpops,1);
  subpops = cell(numPops,1);
  for i = 1:numPops
      subpops{i} = double(obsScheme.subpops(i,:))+1;
  end
else
  numPops = length(obsScheme.subpops);
  subpops = obsScheme.subpops;
  for i = 1:numPops
      subpops{i} = double(subpops{i})+1;
  end  
  numPops = length(subpops);
end
idxStitched = true(yDim,yDim); % (i,j) for each pair of variables ...  
for k = 1:numPops
    idxStitched(subpops{k},subpops{k}) = 0;
end
covy = cov(y');

% now go through all runs and collect performance in LL and stitching
LLs = zeros(20,1);
ccs = zeros(20,1);
cco = zeros(20,1);
for i = 1:20
  dataSet  = ['LDS_save_rerun_small_run', num2str(i-1)];
  clear C_h Pi_h C_h R_h
  load(['/home/mackelab/Desktop/Projects/Stitching/results/test_problems/',...
        '/', dataSet, '.mat'])
  LDScovy_h = C_h * (Pi_h) * C_h' + diag(R_h);
  cco(i) = corr(covy(~idxStitched),LDScovy_h(~idxStitched));
  ccs(i) = corr(covy( idxStitched),LDScovy_h( idxStitched));
  LLs(i) = LL(end-1);
end

[~, idxs] = sort(LLs);
%idxs = 1:length(LLs);

figure(42); 
subplot(2,2,1), 
plot(LLs(idxs)), 
subplot(2,2,2), 
plot(cco(idxs), 'g'); 
hold on, 
plot(ccs(idxs), 'b')
hold off
subplot(2,2,3),
plot(cco, ccs, '.')
subplot(2,2,4),
plot(LLs, cco, '.g')
hold on
plot(LLs, ccs, '.b')
hold off


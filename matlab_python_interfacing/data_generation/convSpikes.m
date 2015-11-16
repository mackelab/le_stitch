% mass-convolve spikes to calcium traces


dataset = '091115_v2'; % date/month/year/_version
load(['/home/mackelab/Desktop/Projects/Stitching/data/clustered_networks/raw_spikes/',dataset,'.mat'])

T = 2000;

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
idxBad_e = fr(1:Ne)<frthrs;   nbe = sum(idxBad_e);
idxBad_i = fr(Ne+1:N)<frthrs; nbi = sum(idxBad_i);

st = st(~[idxBad_e;idxBad_i]); % we just outright discard the cells instead
fr = fr(~[idxBad_e;idxBad_i]); % of keeping them somewhere in the back
Ne = Ne - nbe; % 
Ni = Ni - nbi; % update (they also double as index for the e/i boundary!)
N  = Ne + Ni;  %

clearvars -except st Ne Ni dataset

%%

addpath( ...
   genpath('/home/mackelab/Desktop/Projects/Stitching/code/le_stitch/matlab_python_interfacing'))
S = [];
S.dur      = 2000;
S.recycleSpikeTimes = 1;
S.frameRate = 50;

numClusters = Ne/80;
for idx = 2:numClusters
    traces = zeros(80,S.dur*S.frameRate);
    disp(['clu #',num2str(idx)])
    for i = 1:80
        disp(['- neu #', num2str(i)])
        S.spkTimes = st{(idx-1)*80+i};
        OUT = modelCalcium(S,0);
        traces(i,:) = OUT.data.noisyDFFlowResT;
    end
    save(['/home/mackelab/Desktop/Projects/Stitching/data/clustered_networks/calcium_traces/',...
          dataset, '_cluster_', num2str(idx), '.mat'], ...
         'traces', 'idx')
end
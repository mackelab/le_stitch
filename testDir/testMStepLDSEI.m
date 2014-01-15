clear all;
close all;

T        = 100;
Trials   = 20;
NE       = 25;
NI       = 25;
dE       = 3;
dI       = 3;
doff     = -0.5;
yDim     = NE+NI;
xDim     = dE+dI;
BernFlag = false;
Aodiag   = 0.01; %0.06;
Asodiag  = 0.03; %0.1;

NetworkParams = {'Trials', Trials,'T',T,'NE',NE,'NI',NI,'dE',dE,'dI',dI,'BernFlag',BernFlag,'doff',doff,'Aodiag',Aodiag,'Asodiag',Asodiag};

% generate network
[tp seq] = GenerateEINetwork(NetworkParams{:});

seqOrig = seq;



%%%%%%%%%%%%% try inference, rest is given

seq    = seqOrig;
params = tp;
params = setAlgorithmDefaultParams(params);
params.algorithmic.VarInf.visFlag = false;
params.algorithmic.VarInf.maxIter = 20;


% set inital value

[seq, VarInfStats] = VariationalInferenceEI(seq,params);

plotPosterior(seq,1,params)



%%%%%%%%%%%%%%%% try MStep %%%%%%%%%%%%%%%%%%%%%%


ESTparams = MStepLDSEI(params,seq);

figure
imagesc([params.A ESTparams.A])
figure
imagesc([params.Q ESTparams.Q])
figure
imagesc([params.Q0 ESTparams.Q0])
figure
plot(params.x0,ESTparams.x0,'xr')


clear all;
close all;

T        = 100;
Trials   = 20;
NE       = 25;
NI       = 25;
dE       = 2;
dI       = 1;
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



%%%%%%%%%%%%% try inference, here just over X, rest is given

seq    = seqOrig;
params = tp;
params = setAlgorithmDefaultParams(params);
params.algorithmic.VarInf.visFlag = false;
params.algorithmic.VarInf.maxIter = 20;


% set inital value

[seq, VarInfStats] = VariationalInferenceEI(seq,params);

plotPosterior(seq,1,params)



%%%%%%%%%%%%%%%% try MStep %%%%%%%%%%%%%%%%%%%%%%

params.C = zeros(size(params.C));
params.Cgen = zeros(size(params.Cgen));

params.algorithmic.MStepC.RegHandleC = @L1RegHandle;
params.algorithmic.MStepC.lamC = 1.0;

params = MStepObservationEI(params,seq);

figure()
plot(vec(tp.C),vec(params.C),'x')

figure
imagesc([tp.C params.C])






%seq(1).posterior.phi = [];
%seq = VariationalInferenceEIZ(seq,params);
%plotPosterior(seq,1,params)

%figure
%imagesc(-log((seq(1).y>0.5)*1.0+0.3)')
%colormap gray;
%figSize   = {14,10};
%figuresize(figSize{:},'centimeters')
%xlabel('t');ylabel('neuron no');


%figure
%plot(seq(1).x')
%xlabel('t');ylabel('x(t)');

%figure
%imagesc([params.Cgen tp.Cgen])
%figure
%plot(vec(params.Cgen),vec(tp.Cgen),'rx')
%xlabel('true C');ylabel('estimated C');







% $$$ %%%%%%%%%%%%%%%%%%%%% visualize %%%%%%%%%%%%%%%%%%%%%%
% $$$ figure
% $$$ plot(seqInf(trId).x')
% $$$ 
% $$$ figure
% $$$ plot(seqInf(trId).posterior.lamOpt,seqInf(trId).posterior.lamInit,'rx')
% $$$ 
% $$$ figure
% $$$ imagesc(seqInf(trId).y)
% $$$ 
% $$$ figure
% $$$ plot(vec(seqInf(trId).x),vec(x_ast),'bx')
% $$$ 
% $$$ 
% $$$ figure; hold on; title('std over time')
% $$$ for i=1:min(4,xDim)
% $$$   subplot(2,2,i); hold on
% $$$   plot(1:T,x_err(i,:),'r','linewidth',2)
% $$$ end

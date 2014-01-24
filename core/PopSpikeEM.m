function [NOWparams seq varBound EStepTimes MStepTimes] = PopSpikeEM(params,seq)
%
% [NOWparams seq varBound EStepTimes MStepTimes] = PopSpikeEM(params,seq)
%
%


Trials          = numel(seq);
maxIter         = params.opts.algorithmic.EMIterations.maxIter;
progTolvarBound = params.opts.algorithmic.EMIterations.progTolvarBound;  
maxCPUTime      = params.opts.algorithmic.EMIterations.maxCPUTime;

InferenceMethod = params.model.inferenceHandle;
MstepMethod     = params.model.MStepHandle;

EStepTimes      = nan(maxIter,1);
MStepTimes      = nan(maxIter+1,1);
varBound        = nan(maxIter,1);
PREVparams      = params;			     % params for backtracking!
NOWparams       = params;
varBoundMax     = -inf;


disp(['Starting PopSpikeEM using InferenceMethod  >>' char(InferenceMethod) '<<    and MStepMethod  >>' char(MstepMethod) '<<'])
disp('----------------------------------------------------------------------------------------------------------------------------')


Tall        = sum([seq.T]);
EMbeginTime = cputime;

%%%%%%%%%%% outer EM loop
for ii=1:maxIter

    %%%%%%% E-step: inference

    % do inference
    infTimeBegin   = cputime;
    seq = InferenceMethod(NOWparams,seq);            % !!! clever convention? I save varBound for each trials in seq.posterior... ?
    infTimeEnd     = cputime;
    EStepTimes(ii) = infTimeEnd-infTimeBegin;

    % evaluate variational lower bound  !!! generalize to other cost functions... how to store cost etc....
    varBound(ii) = 0;
    for tr=1:Trials; varBound(ii) = varBound(ii)+seq(tr).posterior.varBound; end;
    fprintf('\rIteration: %i     Elapsed time (EStep): %d     Elapsed time (MStep): %d     Variational Bound: %d',ii,EStepTimes(ii),MStepTimes(ii),varBound(ii))

    % check termination criteria
    if varBound(ii)<varBoundMax    % check if varBound is increasing!
       NOWparams = PREVparams;	   % parameter backtracking
       warning('Variational lower bound is decreasing, aborting EM & backtracking');
       break;
    end

    if (abs(varBound(ii)-varBoundMax)/Tall)<progTolvarBound
       fprintf('\nReached progTolvarBound for EM, aborting')
       break
    end	     

    if (cputime-EMbeginTime)>maxCPUTime
       fprintf('\nReached maxCPUTime for EM, aborting')
       break
    end

    varBoundMax = varBound(ii);
    PREVparams  = NOWparams;

    %%%%%%% M-step

    mstepTimeBegin = cputime;
    NOWparams = PLDSMStep(NOWparams,seq);
    mstepTimeEnd   = cputime;
    MStepTimes(ii+1) = mstepTimeEnd-mstepTimeBegin;	      

end

fprintf('\n----------------------------------------------------------------------------------------------------------------------------\n')
disp('EM iterations done')
function [NOWparams varBound] = PLDSEM(params,seq)
%
% params = PLDSEM(params,seq)
%
% to do:
%
%	- generalize to different inference methods, eg var inf, laplace etc...
%	- CPU times for E and M-steps...
%


Trials = numel(seq);


% put these into configuation file!
maxIter         = params.algorithmic.EMIterations.maxIter;
progTolvarBound = params.algorithmic.EMIterations.progTolvarBound;  
% maxCPUTime = ???

InferenceMethod = @PLDSVariationalInference; % !!! generalize here

EStepTimes  = nan(maxIter,1);
MStepTimes  = nan(maxIter+1,1);
varBound    = nan(maxIter,1);
PREVparams  = params;         % params for backtracking!
NOWparams   = params;
varBoundMax = -inf;

disp('Starting PLDS-EM')
disp('----------------')

Tall = sum([seq.T]);

%%%%%%%%%%% outer EM loop
for ii=1:maxIter

    %%%%%%% E-step: inference

    % do inference
    infTimeBegin = cputime;
    seq = InferenceMethod(NOWparams,seq);
    infTimeEnd   = cputime;
    EStepTimes(ii) = infTimeEnd-infTimeBegin;

    % evaluate variational lower bound
    varBound(ii) = 0;
    for tr=1:Trials; varBound(ii) = varBound(ii)+seq(tr).posterior.varBound;end;
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

    varBoundMax = varBound(ii);
    PREVparams  = NOWparams;

    %%%%%%% M-step

    mstepTimeBegin = cputime;
    NOWparams = PLDSMStep(NOWparams,seq);
    mstepTimeEnd   = cputime;
    MStepTimes(ii+1) = mstepTimeEnd-mstepTimeBegin;	      

end


fprintf('\n----------------\n')
disp('PLDS-EM done')
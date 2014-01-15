clear all
close all


xDim   = 10;
yDim   = 100;
T      = 100;
Trials = 50;
Iters  = 100;

params = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim);

dlyap(params.A,params.Q)

params.PiY = params.C*dlyap(params.A,params.Q)*params.C';

seq = PLDSsample(params,T,Trials);
ESTparams = params;

ESTparams.A  = eye(xDim)*0.9;
ESTparams.Q  = (1-0.9^2)*eye(xDim);
ESTparams.Q0 = dlyap(ESTparams.A,ESTparams.Q);
ESTparams.x0 = zeros(xDim,1);
ESTparams.C  = randn(yDim,xDim)*0.1/sqrt(xDim);
ESTparams.d  = -1.7*ones(yDim,1);

varBound     = -inf;
varOld       = -inf;

for ii=1:Iters
    seq = PLDSVariationalInference(seq,ESTparams);
    %plotPosterior(seq,1,params);
    % works!
    ESTparams = PLDSMStep(ESTparams,seq);
    %ESTparams = LDSMStep(ESTparams,seq);
    %ESTparams = PLDSMStepObservation(ESTparams,seq);

    ESTparams.PiY = ESTparams.C*dlyap(ESTparams.A,ESTparams.Q)*ESTparams.C';
    varOld   = varBound;
    varBound = 0;
    for tr=1:Trials;
    	varBound = varBound + seq(tr).posterior.varBound;
    end;
    fprintf('varBound = %d, Frobenius norm = %d, subspace angle %d \n',varBound,norm(params.PiY-ESTparams.PiY,'fro'),subspace(params.C,ESTparams.C));
    if varBound<varOld
       disp('DECREASING!')
       break
    end
end

%figure
%plot(params.A,ESTparams.A,'xr')

%figure
%plot(params.Q,ESTparams.Q,'xr')

%figure
%plot(params.Q0,ESTparams.Q0,'xr')

%figure
%plot(params.x0,ESTparams.x0,'xr')

%figure
%plot(params.C,ESTparams.C,'xr')



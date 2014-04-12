function seq = PLDSLaplaceInference(params,seq)
%
% function seq = PLDSlpinf(params,seq)
%

Trials      = numel(seq);
[yDim xDim] = size(params.model.C);


mps = params.model;
mps.E = [];
mps.D = [];
mps.initV = mps.Q0;
mps.initx = mps.x0;

runinfo.nStateDim = xDim;
runinfo.nObsDim   = yDim;

infparams.initparams = mps;
infparams.runinfo    = runinfo;
infparams.notes      = params.model.notes;

Tmax = max([seq.T]);
Mu   = zeros(xDim,Tmax); Mu(:,1) = params.model.x0;
for t=2:Tmax
  Mu(:,t) = params.model.A*Mu(:,t-1);
end

for tr=1:Trials
  T = size(seq(tr).y,2);
  indat = seq(tr);
  if isfield(seq(tr),'posterior') && isfield(seq(tr).posterior,'xsm')
    indat.xxInit = seq(tr).posterior.xsm;
  else
    indat.xxInit = Mu(:,1:T);
  end
  seqRet = PLDSLaplaceInferenceCore(indat, infparams);
  seq(tr).posterior.xsm      = seqRet.x;
  seq(tr).posterior.varBound = seqRet.loglik+feval(params.model.baseMeasureHandle,seq(tr).y);
  seq(tr).posterior.Vsm      = reshape(seqRet.V,xDim,xDim*T)';
  seq(tr).posterior.VVsm     = reshape(permute(seqRet.VV(:,:,2:end),[2 1 3]),xDim,xDim*(T-1))';
  seq(tr).posterior.lamOpt   = exp(vec(seqRet.Ypred));
end

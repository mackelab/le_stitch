function seqnew = rebinRaster(seq,dt)
%
% function seq = rebinRaster(seq,dt)
%
% rebin seq by a factor of dt
%


Trials   = numel(seq);
[yDim T] = size(seq(1).y);
Tnew     = floor(T/dt);   % -> throw away partially populated bins

if isfield(seq,'x')
   xDim = size(seq(1).x,1);
end

for tr=1:Trials

    yold = reshape(seq(tr).y(:,1:Tnew*dt),yDim,dt,Tnew);
    ynew = squeeze(sum(yold,2));
    seqnew(tr).y = ynew;
    seqnew(tr).T = Tnew;

    if isfield(seq,'yr')
       yrold = reshape(seq(tr).yr(:,1:Tnew*dt),yDim,dt,Tnew);
       yrnew = squeeze(sum(yrold,2));
       seqnew(tr).yr = yrnew;
    end

    if isfield(seq,'x')
       xold = reshape(seq(tr).x(:,1:Tnew*dt),xDim,dt,Tnew);
       xnew = squeeze(sum(xold,2));
       seqnew(tr).x = xnew;
    end  

end 

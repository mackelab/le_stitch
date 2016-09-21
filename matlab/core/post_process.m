function params = post_process(params)

xDim = size(params.C,2);

if any(~isreal(params.Q))
    params.Q=real(params.Q);
    %!!! prob want to throw an error message here
end
params.Q=(params.Q+params.Q')/2;

if min(eig(params.Q))<0
    [a,b]=eig(params.Q);
    params.Q=a*max(b,1d-10)*a';
    %!!! project onto pos def matrices
end
%if ~isfield(params,'Q0')
%   params.Q0 = real(dlyap(params.A,params.Q));
%end
params.x0     = zeros(xDim,1);
params.R      = diag(diag(params.R));

end
function [fu dq] = computeStimNlin(fstr, x)

% fstr.stop = 1;
% fu = fstr.f(fstr, x);

fu = zeros(fstr.outDim, size(x,2));

dp = fstr.get_params(fstr);
inprod = dp.w * x;
dq = {};
for idx = 1:size(x,2)
    [fu(:,idx), ~, dq{idx}] = fstr.f(fstr, x(:,idx), inprod(:,idx));        
end

dq = [dq{:}];
end
function Bnew = funcmaxB(params, mu, u)
  Qinv = inv(params.Q);
  g = zeros(size(mu));
  g(:,2:end) = mu(:,2:end) - params.A * mu(:,1:end-1);
  g(:,1) = mu(:,1) - params.initx;
    
% % % % %   %%
%   for idx = 1:length(params.fnlin.B)%[1:20 30:40]
%   
%   DerivCheck_Elts(@(x) maxfunc(x, g, u, Qinv, params.fnlin), idx, params.fnlin.B(:))
%   end
% %   
% %   %%
%   keyboard
  
% Try a few different things
if(1)
  options.Method = 'lbfgs';
  options.maxFunEvals = 30;
  Bnew = minFunc(@(x) maxfunc(x, g, u, Qinv, params.fnlin),  params.fnlin.B(:),options);
elseif(0)    
  opt = optimset('Display', 'iter', 'MaxIter', 20, 'GradObj', 'off', 'DerivativeCheck', 'off');
  Bnew = fminunc(@(x) maxfunc(x, g, u, Qinv, params.fnlin, false),  params.fnlin.B(:), opt);  
else
    % let's just try taking a single short step in the gradient direction
  STEPSIZE = 1e-2; 
  [f df] = maxfunc(params.fnlin.B(:), g, u, Qinv, params.fnlin);
  df = df/norm(df);
  Bnew = params.fnlin.B - STEPSIZE*df;
end

  Bnew = reshape(Bnew, size(params.fnlin.B));  
end

function [v dv] = maxfunc(B, g, u, Qinv, f) 
  f.B = reshape(B, size(f.B));
  [c dq] = computeStimNlin(f, u);
  x = g-c;
  y = Qinv*x;
  v = sum(sum(y.*x));
%   z = f.get_params(f);
%   figure(1); imagesc(reshape(z.w(1,:), 12,16));colormap gray;

  dvp = f.get_params(f);
%   dvp.w(:) = 0; dvp.a(:) = 0; dvp.b(:) = 0; dvp.c(:) = 0;
  for fn = fieldnames(dvp)',
      dvp.(fn{1}) = zeros(size(dvp.(fn{1})));
  end
  for idx = 1:(size(c,2))
      dp = dq(idx);
      for fn = fieldnames(dvp)',
          dvp.(fn{1}) = dvp.(fn{1}) + bsxfun(@times, dp.(fn{1}), y(:,idx));
      end
%       dvp.w = dvp.w + bsxfun(@times, dp.w, y(:,idx));
%       dvp.a = dvp.a + dp.a.*y(:,idx); 
%       dvp.b = dvp.b + dp.b.*y(:,idx); 
%       dvp.c = dvp.c + dp.c.*y(:,idx);
  end
  dv = f.to_params(dvp);
  dv = -2*dv(:);
  
end
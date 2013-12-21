function [fu_mean fu_cov fu_cov_T] = computeStimNlinMoments(fstr, x)

q = computeStimNlin(fstr, x);

fu_mean = sum(q,2);

fu_cov = q*q';
fu_cov_T = fu_cov - q(:,end)*q(:,end)';



% fu_mean = zeros(fstr.outDim,1);
% fu_cov = zeros(fstr.outDim);
% 
% fx1 = fstr.f(fstr, x(:,1));
% fu_mean = fx1;
% 
% for idx = 2:size(x,2)
%     fx = fstr.f(fstr, x(:,idx));
%     fu_mean = fu_mean + fx;        
%     fu_cov_T = fu_cov + fx*fx';
% end
% 
% fu_cov = fu_cov_T + fx1*fx1';

end
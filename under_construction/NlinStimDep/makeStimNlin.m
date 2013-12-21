function fstr = makeStimNlin(fname, params)
% other input parameters: 
%    - output dimensionality
%    - input dimensionality  (can have an assert)
%    - 

fstr.outDim = params.outDim;
fstr.inDim = params.inDim; 
switch lower(fname)

    case 'linear'
        fstr.type = 'linear';
        fstr.B = params.B(:);
        fstr.f = @compute_linear;
        fstr.get_params = @lin_vec2param;
        fstr.to_params = @lin_param2vec
        
    case 'quadratic'     
        fstr.type = 'quadratic';
        fstr.B = quad_param2vec(params);
        fstr.f = @compute_quadratic;
        
        fstr.get_params = @quad_vec2param;
        fstr.to_params  = @quad_param2vec;
        
end

end


function qparam = lin_vec2param(fstr)
qparam.w = reshape( fstr.B, fstr.outDim, fstr.inDim);
end

function qvec = lin_param2vec(qparam)
qvec = qparam.w(:);
end

function [q dq dp] = compute_linear(fs, x, q)
B = fs.get_params(fs);
if(nargin<3)    
    q = B.w*x;
end
dp.w = repmat(x(:)', fs.outDim, 1);
dq = lin_param2vec(dp);
end


function B = quad_param2vec(params)
B = [params.w(:); params.a; params.b; params.c];
end

function qparam = quad_vec2param(fstr)
qparam.w = reshape( fstr.B(1:fstr.outDim*fstr.inDim) , fstr.outDim, fstr.inDim);
z = reshape(fstr.B((fstr.outDim*fstr.inDim+1):end), [], 3);
qparam.a = z(:,1);
qparam.b = z(:,2);
qparam.c = z(:,3);
end

function [q dq dp] = compute_quadratic(func, x,z)
   fs = quad_vec2param(func);
   if(nargin < 3)
       z = fs.w*x; %
   end

   q = fs.a.*sum(z.*z,2)/2 + fs.b .* z + fs.c;
   
   dp = quad_vec2param(func);
   for idx = 1:size(fs.w,1)
      dp.w(idx,:) = ( (fs.a(idx)* z(idx) + fs.b(idx))*x )'; 
      dp.a(idx)   = .5*z(idx)^2;
      dp.b(idx)   = z(idx);
      dp.c(idx)   = 1;
   end
   dq = quad_param2vec(dp);
end
% 
% function [q dq dp] = compute_quadratic(func, x)
%    fs = quad_vec2param(func);
%    z = fs.w*x; 
%    q = fs.a.*sum(z.*z,2)/2 + fs.b .* z + fs.c;
%    
%    dp = quad_vec2param(func);
%    for idx = 1:size(fs.w,1)
%       dp.w(idx,:) = ( (fs.a(idx)* z(idx) + fs.b(idx))*x )'; 
%       dp.a(idx)   = .5*z(idx)^2;
%       dp.b(idx)   = z(idx);
%       dp.c(idx)   = 1;
%    end
%    dq = quad_param2vec(dp);
% end
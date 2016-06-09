function V = mat(v,n)
% V = mat(v,n)
%
% Given an m*n column vector v, returns the corresponding
% m-by-n matrix V.  If n is not given, it is assumed that
% v is an n^2 column vector.
if nargin == 1
n2 = length(v);
n = sqrt(n2);
m = n;  mn = m*n;
else
mn = length(v);
m = mn/n;
end
k = 1;
for i=1:m:mn
V(:,k) = v(i:(i+m-1));
k = k+1;
end
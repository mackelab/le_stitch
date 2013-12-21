function y = logdet(A)
% log(det(A)) where A is positive-definite.
% This is faster and more stable than using log(det(A)).

%  From Tom Minka's lightspeed toolbox

try
U = chol(A);
y = 2*sum(log(diag(U)));
catch
warning('Matrix not pd, cannot calculate logdet')
y=-inf;
end

function model = LDSRoughInit(X,model,dt)
%
% simple, rough initialization of an LDS using regression
%

xDim = size(X,1);
T  = size(X,2);
Pi = X*X'./T;
A  = X(:,2:end)/X(:,1:end-1); 

if dt>1
   A = diag(min(max((diag(abs(A))).^(1/dt),0.5),0.98));   
end

Q  = Pi-A*Pi*A;
[Uq Sq Vq] = svd(Q); 
Q  = Uq*diag(max(diag(Sq),1e-3))*Uq';
x0 = zeros(xDim,1); 
Q0 = dlyap(A,Q);

model.A  = A;
model.Pi = Pi;
model.Q  = Q;
model.Q0 = Q0;
model.x0 = x0;

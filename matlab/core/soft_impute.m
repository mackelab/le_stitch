function [Znew, Zls, lambdas, perf] = soft_impute(X, xDim, lambdas, Psig, eps, max_iter, Zold, ifPlot)

yDim = size(X,1);
if size(X,2) ~= yDim
    error('currently assuming X to be square. Little effort to change, though')
end

if nargin < 8
    ifPlot = false;
end

if nargin < 7 || isempty(Zold)
    Zold = zeros(yDim);
end
if nargin < 6 || isempty(eps)
    eps = 10e-3;
end
if nargin < 5 || isempty(max_iter)
    max_iter = 100;
end

if nargin < 4 || isempty(Psig)
    Psig = double(~isnan(X));
end
nPsig = 1 - Psig;

PsigX = Psig .* X;
PsigX(isnan(PsigX)) = 0;    

PsigXs = X(logical(Psig));
Z_RMSE = mean( PsigXs .^2 );

if nargin < 3 || isempty(lambdas)
   [~,D,~] = svd(PsigX);
   if max(diag(D)) <= 0
       disp(['min(\sigma_k) = ', num2str(min(D))])
       error('something wrong. Maximal singular value is non-positiv.')
   end
   lambdas = exp(linspace(-log(100), log(max(diag(D))), 100));
   lambdas = [lambdas(end:-1:1), 0]; 
end

if ifPlot
    figure
    subplot(2,2,1),imagesc(PsigX),
    title('presented data')
    subplot(2,2,2),imagesc(X)
    title('true data')
end

idxx = 1:xDim;
Zls = cell(length(lambdas),1);
perf = zeros(length(lambdas),1);
for i = 1:length(lambdas)
    for j = 1:max_iter
        PnsigZ = nPsig.*Zold;
        [U,D,V] = svd(PsigX + PnsigZ);
        dD = diag(D);
        dDl = max(dD(idxx)-lambdas(i),0);
        Dl = diag( dDl );
        dim_eff = sum(dDl>0);
        %disp(dDl(1:xDim)')
        Znew = U(:,idxx) * Dl * V(:,idxx)';
        err = sqrt(mean(vec(Zold-Znew).^2))/Z_RMSE;
        if err < eps
            disp(['breaking iteration after ', num2str(j), ...
                  ' updates, at norm. RSME = ', num2str(err), ...
                  ', eff. dim = ', num2str(dim_eff)])
            break
        end
        Zold = Znew;
    end
    Zls{i} = Znew;
    perf(i) = sqrt(mean( (Zls{i}(logical(Psig))-PsigXs ).^2))/Z_RMSE;    
    if ifPlot
        
      subplot(2,2,3), 
      plot(X(logical(Psig)), Zls{i}(logical(Psig)), 'r.')
      hold on      
      plot(X(~logical(Psig)), Zls{i}(~logical(Psig)), 'b.')
      hold off
      
      axis(1.1*[min(X(:)),max(X(:)),min(X(:)),max(X(:))]) % assuming min(X)<0
      title('guessed vs. provided matrix, non-obs. parts')
      
      subplot(2,2,4),
      imagesc(Zls{i})
      title(['est at. \lambda_k, k =', num2str(i), ' out of ', ...
           num2str(length(lambdas))])
      pause(0.00001)
    end
end
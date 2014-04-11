function smooth = qlds_fast_estep_gauss(data, params)
% Kalman filter/smoother, but with panache. 
% (c) Evan Archer, 2014
% 
    ri = params.runinfo;
    mps = params.initparams;

    %% Precomputations
    
    H = mps.nlin.f(mps.nlin, data.h); % compute nlin first. 
    Qinv = pinv(mps.Q);
    Q0inv = pinv(mps.initV);
    Rinv = pinv(mps.R);
    CRiC = mps.C'*(Rinv*mps.C);
    AQiA = mps.A'*Qinv*mps.A;

    QiH = zeros(size(H));
    QiH(:,2:end) = Qinv*H(:,2:end);
    QiH(:,1) = Q0inv * ( H(:,1) + mps.initx );

    T = size(data.y,2);

    % sum over s and u
    if isempty(mps.D) 
        ds = sparse(size(data.y,1), size(data.y,2));
    else
        S = reshape(data.s, ri.nObsDim, ri.nSDim, []);
        ds = squeeze(sum(bsxfun(@times,mps.D,S),2));
    end
    if isempty(mps.E)
        ds = ds + sparse(ri.nObsDim, size(data.y,2));
    else
        ds = ds + mps.E*data.u;
    end

    Yresid = data.y - bsxfun(@plus, ds, mps.d);

    %% Construct gradients and hessians (in the Gaussian case, they are evaluated at x = 0)

    % latent dynamics terms
    % (these terms will always hold, no matter the observation model)
    lat_grad = [ QiH(:,1:end-1) - mps.A'*QiH(:,2:end),   QiH(:,end)];

    II = speye(size(data.y,2)-2);
    lat_hess_diag = blkdiag(sparse(Q0inv+AQiA), kron( II, Qinv + AQiA), sparse(Qinv));

    II_c = circshift(speye(size(data.y,2)), [0 1]); II_c(end,1) = 0; 
    lat_hess_off_diag = kron(II_c, -mps.A'*Qinv); lat_hess_off_diag = lat_hess_off_diag + lat_hess_off_diag';

    % gaussian observations
    gauss_grad = mps.C'*(Rinv*( Yresid ));
    gauss_hess = kron(speye(size(data.y,2)), CRiC);

    Hess = gauss_hess + lat_hess_diag + lat_hess_off_diag;
    dd = lat_grad + gauss_grad; dd = dd(:);
    
    %% Compute posterior means
    
    smooth.x = reshape(Hess \ dd, size(lat_grad));

    %% Compute blocks of inverses
    
    AA0 = Q0inv + CRiC + AQiA;
    AAE = Qinv + CRiC;
    AA = Qinv + CRiC + AQiA;
    BB = -(mps.A'*Qinv); 
    smooth.VV = zeros(size(AA,1),size(AA,1),T);
    
    %% Compute diagonal and upper diagonal blocks of inverse
    
    AAz = zeros( [size(AA0), 3] ); AAz(:,:,1) = AA0; AAz(:,:,2) = AA; AAz(:,:,3) = AAE;    
    [smooth.V,smooth.VV(:,:,2:end)] = sym_blk_tridiag_inv_v1(AAz, BB, [1; 2*ones(T-2,1); 3], ones(T-1,1));

    %% Compute likelihood

    YY = Yresid-mps.C*smooth.x;
    XX = smooth.x(:,2:end) - mps.A*smooth.x(:,1:end-1) - H(:,2:end);
    X0 = (smooth.x(:,1) - mps.initx - H(:,1));

    % Dropping constant terms that depend on dimensionality of states and observations
    nLL = logdet(mps.initV)+ T*logdet(mps.R)+(T-1)*logdet(mps.Q) + ( sum(sum(YY.*(Rinv*YY))) + sum(sum(XX.*(Qinv*XX))) + sum(sum(X0.*(Q0inv*X0))) );

    smooth.loglik = -nLL/2;

end
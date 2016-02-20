function [SIGfp,SIGyy,SIGyy_sym] = construct_hankel_cov(params)

yDim = size(params.C,1);
xDim = size(params.C,2);

hankelSize = xDim;

SIGfp  = zeros(yDim*hankelSize,yDim*hankelSize);

Pi = direct_dlyap(params.A, params.Q);

% idx = 1:yDim;
% covy_lag1 =  params.C*params.A*Pi*params.C';
% for k = 0:hankelSize-1
%  SIGfp(idx+k*yDim,idx+k*yDim) = covy_lag1;
%  for kk = 1:hankelSize-k-1
%      covy_lag =  params.C * params.A^(kk+1)*Pi * params.C';
%      SIGfp(idx+k*yDim,idx+(kk+k)*yDim) = covy_lag';
%      SIGfp(idx+(kk+k)*yDim,idx+k*yDim) = covy_lag;
%  end
% end
SIGyy_sym = params.C*Pi*params.C';
SIGyy = SIGyy_sym + params.R;

%%  

%Yshift = Ytot;
covxx = Pi;    
  indX = 1:yDim;
  for k=1:(2*hankelSize-1)
    
    %Yshift = circshift(Yshift,[0 1]);
    %lamK   = Ytot*Yshift';
    covxx =  params.A * covxx; % * params.A';
    lamK = params.C*covxx*params.C';
        
    if k<(hankelSize+0.5)
      for kk=1:k
	SIGfp(indX+(k-kk)*yDim,indX+(kk-1)*yDim) = lamK;
      end
    else
      for kk=1:(2*hankelSize-k)
	SIGfp(indX+(hankelSize-kk)*yDim,indX+(kk+k-hankelSize-1)*yDim) = lamK;
      end
    end
  end


end
function [tvdat nlinstr X] = nlin_quadratic(fo,vdat,fot,wts)
% [tvdat nlinstr] = nlin_quadratic(fo,vdat,fot,wts)
% generate quadratic from filtered output.
% 
% Input: fo,vdat,fot,wts
%
% OUTPUT: 
%       tY - nx1 vector, predicted output
%  nlinstr - struct with feilds:
%     * name    - the string 'quadratic'
%     * wts     - regression weights used to produce tY
%     * A, B, C - regression weights as quadratic 
%            form (quadratic, linear, and constant
%            terms, respectively).
%  X - The kernelized design matrix; ie, 
%           y = X*wts 
%      is the quadratic prediction of the estimation data. 
%
    if(isempty(fo) && nargin < 4)
        fprintf('\tComputing constant model.\n')
        nfilts = 0;
    elseif(isempty(fo)&&~isempty(fot))
        nfilts = size(fot,2);
    else
        nfilts = size(fo,2);
    end

    % Grab convenient indices to cross-mult.
    [ii,jj] = find(triu(ones(nfilts))); 
    %[ii,jj] = find(diag(diag(ones(nfilts)))); 

    if( nargin < 4)
        % build design matrix
        if( ~exist('wts', 'var') )
            slen = size(vdat,1);
            X = [ones(slen,1), fo, fo(:,ii).*fo(:,jj)];
            wts = X\vdat;
        end
    end
        
    wts = wts(:); % make sure weights are a column vector

    % prepare data for prediction
    if( nfilts == 0 )
        if(exist('fot', 'var'))
           tvdat = wts*ones(size(fot,1),1); 
        else
            tvdat = wts;
        end
    elseif( exist('fot', 'var') && ~isempty(fot))
        tslen = size(fot,1);
        tX = [ones(tslen,1), fot, fot(:,ii).*fot(:,jj)];

        tvdat = tX*wts;
    else
        tvdat = [];
    end
 
    %% Pass back A,B,C (just weights reordered into a familiar quadratic
    %% format).
    if(nargout > 1)
        % return explicit quadratic form
        B = wts(2:nfilts+1);
        C = wts(1);
        
        A = zeros(nfilts);
        for idx = 1:length(ii)
        	A(ii(idx),jj(idx)) = wts(nfilts+1+idx);
        end
        
        d = diag(diag(A));
        A = ( (A-d)' + (A-d) )/2 + d;
        
        nlinstr = struct();
        nlinstr.name = 'quadratic';
        nlinstr.wts = wts;
        nlinstr.A = A;
        nlinstr.B = B;
        nlinstr.C = C;
    end
    
end

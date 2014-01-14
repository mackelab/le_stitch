
function [B nbars nkt] = make_gabor_basis(nkt, nbars)
%%
% Construct a "neuron" 
% Build some filters - 

nFilts = 3;
if(nargin < 1)
nkt = 8;
end
if(nargin < 2)
nbars = 8;
end

nt = nbars*nkt;

B = zeros(nkt,nbars, nFilts);
theSize = [nkt nbars];

% fill 'em up

theMean = [1 1];
dev = 1;
oscDir = 0;
freqSpace = 100;
phase = pi/2;
timeVsSpace = .5;

gb = gabor_fn(theSize, theMean, dev, oscDir, freqSpace, phase, timeVsSpace);
B(:,:,1) = gb;


theMean = [1 1];
dev = 2;
oscDir = 3*pi/4;
freqSpace = 20;
phase = pi/2;
timeVsSpace = 3;

gb = gabor_fn(theSize, theMean, dev, oscDir, freqSpace, phase, timeVsSpace);
B(:,:,2) = gb;



theMean = [1 1];
dev = 1.5;
oscDir = pi/2;
freqSpace = 20;
phase = 1.5;
timeVsSpace = 1;

gb = gabor_fn(theSize, theMean, dev, oscDir, freqSpace, phase, timeVsSpace);
B(:,:,3) = gb;


figure(1)
colormap gray
subplot(131)
%imagesc(B(:,:,1))
imagesc(B(:,:,1))
title('\beta _1')
set(gca, 'xtick', [0],'ytick', [0])
subplot(132)
%imagesc(B(:,:,2))
imagesc(B(:,:,2))
title('\beta _2')
set(gca, 'xtick', [0],'ytick', [0])
subplot(133)
%imagesc(B(:,:,3))
imagesc(B(:,:,3))
title('\beta _3')
set(gca, 'xtick', [0],'ytick', [0])

B = orth(reshape(B, nkt*nbars, nFilts));
% B = reshape(B_vec, nkt, nbars, nFilts);

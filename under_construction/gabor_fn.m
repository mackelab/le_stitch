function gb = gabor_fn(bnds,mn,sigma,theta,lambda,psi,gamma)
% theSize = [timeBins spaceBins];
% theMean = [0 4];
% dev =5;
% oscDir = pi/8;
% freqSpace = 10;
% phase = pi/2;
% timeVsSpace = 5;
% 
% gb = gabor_fn(theSize, theMean, dev, oscDir, freqSpace, phase, timeVsSpace);
%

sigma_x = sigma;
sigma_y = sigma/gamma;
 
% Bounding box
nstds = 3;
ymin = -(floor(bnds(1)/2) - 1 + mod(bnds(1),2));
ymax = floor(bnds(1)/2);

xmin = -(floor(bnds(2)/2) - 1 + mod(bnds(2),2));
xmax = floor(bnds(2)/2);

[x,y] = meshgrid(xmin:xmax,ymin:ymax);

 
% Rotation 
x_theta=(x-mn(1))*cos(theta)+(y-mn(2))*sin(theta);
y_theta=-(x-mn(1))*sin(theta)+(y-mn(2))*cos(theta);
 
gb= 1/(2*pi*sigma_x *sigma_y) * exp(-.5*(x_theta.^2/sigma_x^2+y_theta.^2/sigma_y^2)).*cos(2*pi/lambda*x_theta+psi);

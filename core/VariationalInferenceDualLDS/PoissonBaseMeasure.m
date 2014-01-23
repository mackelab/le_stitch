function [f] = PoissonBaseMeasure(y);
%
% log y!
%

f = -sum(log(gamma(vec(y)+1)));

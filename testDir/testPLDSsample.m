clear all
close all

xDim   = 10;
yDim   = 100;

T      = [100 50 120];
Trials = numel(T);

params = PLDSgenerateExample('xDim',xDim,'yDim',yDim);
seq    = PLDSsample(params,T,Trials);

mean(vec([seq.y]))
figure
plot(seq(1).x')
figure
imagesc(seq(1).y)
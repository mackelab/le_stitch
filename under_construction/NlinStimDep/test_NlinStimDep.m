

%% Test linear function

params.B = [1 2; 3 4];
params.outDim = size(params.B,1);
flintest = makeStimNlin('linear', params);
z = rand(2,300);
fz = computeStimNlin(flintest, z);

sum(sum(fz - params.B*z))

[mu sig] = computeStimNlinMoments(flintest, z);



%% Test quadratic function
params.outDim = 2;
params.inDim = 10;
params.w = rand(params.outDim,params.inDim);
params.a = rand(params.outDim,1);
params.b = rand(params.outDim,1);
params.c = rand(params.outDim,1);

fquadtest = makeStimNlin('quadratic', params);

z = rand(10,300);

fz = computeStimNlin(fquadtest, z);

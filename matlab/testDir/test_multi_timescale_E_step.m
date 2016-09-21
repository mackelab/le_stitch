clear all
close all
clc
%%
% Variable definitions 

q = 3;                         % dimensionality of dynamical system
N = 1000000;                    % total length of simulation
dts = [0.02, 0.05, 0.50];      % time discretization step lengths

ds = 0.0001;                   % time discretization step for numerical 
                               % solution of the integral for R given
                               % G, M and dt

dtis = [1,2,3,1];                      % set sequence of appearance of 
                                       % time discretization step lengths
dtsp = round([1, N/5, 2*N/5, 4*N/5]);  % set onsets of appearance of 
                                       % time discretization step lengths
                                       % (numbers in counts, not time!)
                                       
x0 = ones(q,1);                % initial state
                                       
% draw continuous-time linear-dynamics matrix M
if q == 2
   M = [-1, 0; 0, -3];
elseif q == 3
   M = diag([-0.9, -0.5, -0.2]);
   tmp = randn([3,3]);
   M = tmp * M * inv(tmp);
elseif q > 1
  %M = - wishrnd(eye(q), q-1);
else
  M = - 0.5 * rand(1) - 0.5;
end
% set continuous-time Brownian noise mixing matrix G
G = randn(q,q);

As  = cell(length(dts),1);          % linear-dynamics matrices A
for k = 1:length(As)
    As{k} = expm(M*dts(k));
end

Rs  = cell(length(dts),1);          % covariances for innovation noise
GG  = G*G';
for k = 1:length(As)
    Rs{k} = zeros(q,q);
    for sk = 0:floor(dts(k)/ds)
        expMsk = expm(M * (dts(k) - sk*ds));
        Rs{k} = Rs{k} + dts(k) * (expMsk * GG * expMsk');
    end
    Rs{k} = Rs{k} / sk;
end

mus = cell(length(dts),1);          % means for innovation noise (all zero)
for k = 1:length(As)
    mus{k} = zeros(q,1);
end

dtsp(end+1) = N; % so that we know when to stop...

%% start ancestral sampling procedure

x = zeros(q,N);

x(:,1) = x0;  % set initial condition

for sp = 1:length(dtsp)-1   % for each switch point ...
    dti = dtis(sp);   % p of current dt_p (current time scale index)
    dt = dts(dti);         % current time scale;
    for n = dtsp(sp)+1:dtsp(sp+1)
        x(:,n+1) = As{dti} * x(:,n) + mvnrnd(mus{dti}, Rs{dti})';     
    end    
end
x = x(:,2:end); % we know the first element ...

%% check results

covs_raw  = cell(length(dtsp)-1,1);
covs_ssf  = cell(length(dtsp)-1, 1);  % covs for same sampling frequency
means = cell(length(dtsp)-1,1);
xcorr = cell(length(dtsp)-1, 1);
xcorr_raw = cell(length(dtsp)-1, 1);

for sp = 1:length(dtsp)-1   % for each switch point ...
    dti = dtis(sp);   % p of current dt_p (current time scale index)
    dt = dts(dti);         % current time scale;

    x_raw = x(:,dtsp(sp):dtsp(sp+1)-1);
    covs_raw{sp} = cov(x_raw');
    means{sp} = mean(x_raw,2);
    nMax = 20;
    xcorr_raw{sp} = zeros(q,q,2*nMax+1);
    for i = 1:q
      for j = 1:q
        for n = -nMax:nMax       
          tmp = corrcoef(x_raw(i, (1+nMax:size(x_raw,2)-nMax)+n)', x_raw(j, (1+nMax:end-nMax))');
          xcorr_raw{sp}(i,j,n+nMax+1) = tmp(1,2);
        end
      end
    end    
    
    if max(dts)/dt - round(max(dts)/dt) < 1e-3
        x_ssf = x_raw(:, 1:max(dts)/dt:end);
        covs_ssf{sp} = cov(x_ssf');
    end
    
    xcorr{sp} = zeros(q,q,2*nMax+1);
    for i = 1:q
      for j = 1:q
        for n = -nMax:nMax       
          tmp = corrcoef(x_ssf(i, (1+nMax:size(x_ssf,2)-nMax)+n)', x_ssf(j, (1+nMax:end-nMax))');
          xcorr{sp}(i,j,n+nMax+1) = tmp(1,2);
        end
      end
    end
    
end

%% visualize results

figure(2);                                          % parameters A, R                                     
minC = Inf; maxC = -Inf;
for sp = 1:length(dts)
    minC = min([minC,min(As{sp}(:))]); 
    maxC = max([maxC,max(As{sp}(:))]); 
end
for sp = 1:length(dts)
    subplot(length(dts),2,2*sp-1)
    imagesc(As{sp})
    caxis([minC, maxC]);
    title(['A_i, i = ', num2str(sp), ', dt = ', num2str(dts(sp))])
    box off
    set(gca, 'XTick', [])
    set(gca, 'YTick', [])
    colorbar;
end
minC = Inf; maxC = -Inf;
for sp = 1:length(dts)
    minC = min([minC,min(Rs{sp}(:))]); 
    maxC = max([maxC,max(Rs{sp}(:))]); 
end
for sp = 1:length(dts)
    subplot(length(dts),2,2*sp)
    imagesc(Rs{sp})
    caxis([minC, maxC]);
    title(['R_i, i = ', num2str(sp), ', dt = ', num2str(dts(sp))])
    box off
    set(gca, 'XTick', [])
    set(gca, 'YTick', [])
    colorbar;    
end

figure(3);                                    % raw covariances and means
minC = Inf; maxC = -Inf;
for sp = 1:length(dtsp)-1
    minC = min([minC,min(covs_raw{sp}(:))]); 
    maxC = max([maxC,max(covs_raw{sp}(:))]); 
end
for sp = 1:length(dtsp)-1
    subplot(length(dtsp)-1,3,3*sp-2)
    plot(x(:,dtsp(sp)+(0:1000))')
    axis([0, 1001, 0, 1]); axis autoy
    xlabel('n')
    ylabel(['x_{i,n}, i = 1, ..., ', num2str(q)])
    box off
    set(gca, 'TickDir', 'out')
    title(['example traces, dt = ',num2str(dts(dtis(sp)))])
    subplot(length(dtsp)-1,3,3*sp-1)
    imagesc(covs_raw{sp})
    caxis([minC, maxC]);
    box off
    set(gca, 'TickDir', 'out')
    title(['Cov(X), dt = ', num2str(dts(dtis(sp)))])
    subplot(length(dtsp)-1,3,3*sp)
    plot(means{sp}, 'o')
    box off
    set(gca, 'TickDir', 'out')
    axis([0.5, q+0.5, 0, 1]); axis autoy
    title(['mean(X), dt = ', num2str(dts(dtis(sp)))])
end

figure(4);                           % covariances and means once resampled
minC = Inf; maxC = -Inf;
for sp = 1:length(dtsp)-1
    minC = min([minC,min(covs_ssf{sp}(:))]); 
    maxC = max([maxC,max(covs_ssf{sp}(:))]); 
end
for sp = 1:length(dtsp)-1
    subplot(length(dtsp)-1,3,3*sp-2)
    plot(x(:,dtsp(sp)+(0:max(dts)/dt:end)')
    axis([0, 1001, 0, 1]); axis autoy
    xlabel('n')
    ylabel(['x_{i,n}, i = 1, ..., ', num2str(q)])
    box off
    set(gca, 'TickDir', 'out')
    title(['example traces, dt = ',num2str(dts(dtis(sp)))])
    subplot(length(dtsp)-1,3,3*sp-1)
    imagesc(covs_ssf{sp})
    caxis([minC, maxC]);
    box off
    set(gca, 'TickDir', 'out')
    title(['Cov(X), dt = ', num2str(dts(dtis(sp)))])
    subplot(length(dtsp)-1,3,3*sp)
    plot(means{sp}, 'o')
    box off
    set(gca, 'TickDir', 'out')
    axis([0.5, q+0.5, 0, 1]); axis autoy
    title(['mean(X), dt = ', num2str(dts(dtis(sp)))])
end
%%
figure(5);
clrs = hsv(length(dtsp));
for sp = 1:length(dtsp)-1
    for i = 1:q
        for j = i:q
            subplot(q+2,q,(i-1)*q+j)
            plot(-nMax:nMax, squeeze(xcorr{sp}(i,j,:)), 'color', clrs(sp,:))
            hold on
            title(['i = ',num2str(i), ', j = ',num2str(j)])
            if j == 1 && i == 1
                xlabel('temp. offset n - m')
            end
            if j == round(q/2) && i == j 
              xlabel('resampled to same sampling densitiy')
            end
            box off
            set(gca, 'TickDir', 'out')
            subplot(q+2,q,(j+1)*q+i)
            plot(-nMax:nMax, squeeze(xcorr_raw{sp}(i,j,:)), 'color', clrs(sp,:))
            hold on
            if i == round(q/2) && j == i+1 
              xlabel('raw sampling densitiy')
            end
            title(['i = ',num2str(j), ', j = ',num2str(i)])
            box off
            set(gca, 'TickDir', 'out')
            
        end
    end
end
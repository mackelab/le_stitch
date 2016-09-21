figure; 

subplot(1,2,1)
h = plot(3:-1:1, log(0.5) ./ log(sort(eig(A))), 'ko', 'markerSize', 7, 'linewidth', 3);
set(h,'MarkerEdgeColor','none','MarkerFaceColor','k')
hold on

h = plot(3:-1:1, log(0.5) ./ log(sort(eig(A_h))), 'go', 'markerSize', 7, 'linewidth', 3);
set(h,'MarkerEdgeColor','none','MarkerFaceColor','g')
axis([0.5,3.5,0,1]), axis autoy
box off
set(gca, 'TickDir', 'out')
xlabel('# latent mode')
ylabel('half life (bins)')
set(gca, 'XTick', 1:3)


subplot(1,2,2)
h = plot(3:-1:1, 20 *log(0.5) ./ log(sort(eig(A))), 'ko', 'markerSize', 7, 'linewidth', 3);
set(h,'MarkerEdgeColor','none','MarkerFaceColor','k')
hold on

h = plot(3:-1:1, 20 *log(0.5) ./ log(sort(eig(A_h))), 'go', 'markerSize', 7, 'linewidth', 3);
set(h,'MarkerEdgeColor','none','MarkerFaceColor','g')
axis([0.5,3.5,0,1]), axis autoy
box off
set(gca, 'TickDir', 'out')
xlabel('# latent mode')
ylabel('half life [ms]')
set(gca, 'XTick', 1:3)
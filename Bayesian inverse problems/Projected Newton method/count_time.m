% Counting the computational time of PNT and Newton method when
% abs(||Axk-b||_{M^-1}^2-tau*m)<=1e-8.

clear, clc;
directory = pwd;
path(directory, path)
addpath(genpath('..'))
rng(2023); 

% time for shaw
t1 = [0.014, 0.057, 0.133, 0.277, 0.365];    % PNT
t2 = [0.577, 4.394, 8.196, 11.600, 55.919];   % NT

% time for heat
t3 = [0.033, 0.098, 0.143, 0.279, 0.407];    % PNT
t4 = [0.267, 1.708, 4.007, 9.027, 17.715];   % NT

% problem scale
n = [1000, 2000, 3000, 4000, 5000];


%------ plot --------
figure;
semilogy(n, t1, '-d','Color',[0.6350 0.0780 0.1840],'MarkerIndices',1:1:5,...
    'MarkerSize',6,'MarkerFaceColor',[0.6350 0.0780 0.1840],'LineWidth',1.5);
hold on;
semilogy(n, t2, '-s','Color',[0 0.4470 0.7410],'MarkerIndices',1:1:5,...
    'MarkerSize',6,'MarkerFaceColor',[0 0.4470 0.7410],'LineWidth',1.5);
xlabel('n','Fontsize',16);
legend('PNT','Newton','Fontsize',16, 'Location', 'northwest');
ylabel('Time (seconds)','Fontsize',16);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.1);
set(gca, 'MinorGridAlpha', 0.01);


figure;
semilogy(n, t3, '-d','Color',[0.6350 0.0780 0.1840],'MarkerIndices',1:1:5,...
    'MarkerSize',6,'MarkerFaceColor',[0.6350 0.0780 0.1840],'LineWidth',1.5);
hold on;
semilogy(n, t4, '-s','Color',[0 0.4470 0.7410],'MarkerIndices',1:1:5,...
    'MarkerSize',6,'MarkerFaceColor',[0 0.4470 0.7410],'LineWidth',1.5);
xlabel('n','Fontsize',16);
legend('PNT','Newton','Fontsize',16, 'Location', 'northwest');
ylabel('Time (seconds)','Fontsize',16);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.1);
set(gca, 'MinorGridAlpha', 0.01);
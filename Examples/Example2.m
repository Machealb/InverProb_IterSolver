% Test for pGKB based pure iterative methods and hybird regularization method:
%   (1). pGKB_SPR with DP or L-curve as early stopping criteria, and
%   (2). pGKBhyb with secant update or WGCV for updating regularization parameters. 
%
% Using 1D small-scale test problems, and show the effect of the algorithms
% for reconstructing solutions.
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 06, July, 2023.
%

clear, clc;
directory = pwd;
path(directory, path)
%path([directory, '/regu'], path)
addpath(genpath('..'))
rng(2023);  

% test problems
%[A,b_true,x_true] = gravity(1024);
%[A,b_true,x_true] = heat(2048);  
% [A, b_true, x_true] = deriv2(2000);  
[A, b_true, x_true] = gauss1dsig(800, 10);  

% add noise
nel = 5e-3; % Noise level
b = AddNoise(b_true, 'gauss', nel);  % noisy data

% prepare algorithms
[m, n] = size(A);
% L1 = get_l(n, 1); 
L1 = genLirn(x_true, '1dTV_0', 1e-6);
p = size(L1, 1);
C = [A; L1];
M = L1' * L1;
alpha = 1;
G = A'*A + alpha*M;
xn = norm(x_true);
eta = 1.001 * nel * norm(b_true);

% compare pGKB reguarization methods
tol1 = 1e-6;
k = 25;
er1 = zeros(k,1);
er2 = zeros(k,1);
er3 = zeros(k,1);

[X0, res0, eta0, iterstop0, info0] = pGKBSPR_LC(A, b, M, alpha, k, tol1, 1);
[X1, res1, iterstop1] = pGKBSPR_DP(A, b, M, alpha, k, tol1, 1, eta);
[X2, res2, Lam2, iterstop2] = pGKBhyb_su(A, b, M, alpha, k, tol1, 1, eta);
[X3, res3, Lam3, GCV, iterstop3] = pGKBhyb_wgcv(A, b, M, alpha, k, tol1, 1);

for i =1:k
    er1(i) = norm(x_true-X1(:,i)) / xn;
    er2(i) = norm(x_true-X2(:,i)) / xn;
    er3(i) = norm(x_true-X3(:,i)) / xn;
end

[~, k0] = min(er1);


%-------- plot ----------------------------
lw = 2; l = 1:1:n;

%---------- true and noisy data --------------
figure;
plot(l, x_true,'b-', 'LineWidth', 2.0);
title('True solution','fontsize',16);
%ylim([-0.6 0.8]);
set(gca,'fontsize',16);

figure;
plot(l, b,'m-', 'LineWidth', 2.0);
title('Noisy data','fontsize',16);
%ylim([-0.6 0.8]);
set(gca,'fontsize',16);

%---------- reconstructed solutions  ---------
figure; 
plot(l, x_true,'b--', 'LineWidth', 2.0);
% hold on
% plot(l, b,'g-.', 'LineWidth', 2.0);
hold on
plot(l, X1(:,k0),'r-', 'LineWidth', 2.0);
legend('True solution', ['Best solution, $k_{0}$=',num2str(k0)],'interpreter','latex','fontsize',20);
ylim([-0.6 0.8]);

figure; 
plot(l, x_true,'b--', 'LineWidth', 2.0);
% hold on
% plot(l, b,'g-.', 'LineWidth', 2.0);
hold on
plot(l, X0(:,iterstop0),'r-', 'LineWidth', 2.0);
legend('True solution', ['LC solution, $k$=',num2str(iterstop0)],'interpreter','latex','fontsize',20);
ylim([-0.6 0.8]);

figure; 
plot(l, x_true,'b--', 'LineWidth', 2.0);
% hold on
% plot(l, b,'g-.', 'LineWidth', 2.0);
hold on
plot(l, X1(:,iterstop1),'r-', 'LineWidth', 2.0);
legend('True solution', ['DP solution, $k$=',num2str(iterstop1)],'interpreter','latex','fontsize',20);
ylim([-0.6 0.8]);

figure; 
plot(l, x_true,'b--', 'LineWidth', 2.0);
% hold on
% plot(l, b,'g-.', 'LineWidth', 2.0);
hold on
plot(l, X2(:,iterstop2),'r-', 'LineWidth', 2.0);
legend('True solution', ['SU solution, $k$=',num2str(iterstop2)],'interpreter','latex','fontsize',20);
ylim([-0.6 0.8]);

figure; 
plot(l, x_true,'b--', 'LineWidth', 2.0);
% hold on
% plot(l, b,'g-.', 'LineWidth', 2.0);
hold on
plot(l, X3(:,iterstop3),'r-', 'LineWidth', 2.0);
legend('True solution', ['WGCV solution, $k$=',num2str(iterstop3)],'interpreter','latex','fontsize',20);
ylim([-0.6 0.8]);


%--------- convergence history -----------------
figure;
semilogy(1:k, er1, '->','Color','b','MarkerIndices',1:1:k,...
    'MarkerSize',5,'MarkerFaceColor','b','LineWidth',1.5);
hold on;
semilogy(1:k, er2, '-s','Color',[1,0.47,0.1],'MarkerIndices',1:1:k,...
    'MarkerSize',5,'MarkerFaceColor',[1.0,0.47,0.1],'LineWidth',1.5);
hold on;
semilogy(1:k, er3, '-o','Color','g','MarkerIndices',1:1:k,...
    'MarkerSize',5,'MarkerFaceColor','g','LineWidth',1.5);
hold on;
semilogy(iterstop0, er1(iterstop0),'bo', 'MarkerSize',16, 'LineWidth',2)
hold on;
semilogy(iterstop2, er2(iterstop2),'o','Color',[1,0.47,0.1]', 'MarkerSize',16, 'LineWidth',2)
hold on;
semilogy(iterstop3, er3(iterstop3),'go', 'MarkerSize',16, 'LineWidth',2)
xlabel('Iteration','fontsize',16);
legend('pGKB\_SPR', 'pGKB\_HR, SU', 'pGKB\_HR, WGCV', 'Location', 'northeast','fontsize',16);
ylabel('Relative error','fontsize',16);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.3);
set(gca, 'MinorGridAlpha', 0.01);


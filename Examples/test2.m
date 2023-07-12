% Test convergence bahavior for pGKB based pure iterative regularization 
% method pKGKB_DP/LC for different choice of accuracy of inner iterations.
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
% [A,b_true,x_true] = heat(2048);  
%[A, b_true, x_true] = deriv2(2000);  
[A, b_true, x_true] = gauss1dsig(800, 10);

% add noise
nel = 5e-3; % Noise level
b = AddNoise(b_true, 'gauss', nel);  % noisy data

% prepare algorithms
[m, n] = size(A);
%L1 = get_l(n, 1); 
L1 = genLirn(x_true, '1dTV_0', 1e-6);
p = size(L1, 1);
C = [A; L1];
M = L1' * L1;
alpha = 1;
G = A'*A + alpha*M;
xn = norm(x_true);
eta = 1.001 * nel * norm(b_true);

% compare pGKB and jbd method
tol = 0;
k = 20;  
er1 = zeros(k,1);
er2 = zeros(k,1);
er3 = zeros(k,1);


[X1, res1, iterstop1] = pGKBSPR_DP(A, b, M, 1, k, tol, 1, eta);
[X2, res2, iterstop2] = pGKBSPR_DP(A, b, M, 1, k, 1e-6, 1, eta);
[X3, res3, iterstop3] = pGKBSPR_DP(A, b, M, 1, k, 1e-4, 1, eta);

for i =1:k
    er1(i) = norm(x_true-X1(:,i)) / xn;
    er2(i) = norm(x_true-X2(:,i)) / xn;
    er3(i) = norm(x_true-X3(:,i)) / xn;
end

[~, k0] = min(er2);

%-------- plot ------------------
lw = 2; l = 1:1:n;

figure; 
plot(l, x_true,'b--', 'LineWidth', 2.0);
% hold on
% plot(l, b,'g-.', 'LineWidth', 2.0);
hold on
plot(l, X2(:,k0),'r-', 'LineWidth', 2.0);
legend('True', 'Reconstructed');
%legend('True', 'Blurred', 'Reconstructed');
%ylim([-0.6 0.8]);

figure;
semilogy(1:k, er1, '-o','Color','[0.33010 0.8450 0.2330]', 'LineWidth', 2.0);
hold on;
semilogy(1:k, er2, '-+','Color','[0.800 0.3250 0.0180]', 'LineWidth', 2.0);
hold on;
semilogy(1:k, er3, 'b^-', 'LineWidth', 2.0);
xlabel('Iteration','Fontsize',16);
legend('$\mathtt{tol}=0$', '$\mathtt{tol}=10^{-6}$', '$\mathtt{tol}=10^{-4}$',...
    'Fontsize',16, 'Location', 'southeast','Interpreter','latex');
ylabel('Relative error','Fontsize',16);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.3);
set(gca, 'MinorGridAlpha', 0.01);


%-----------------
% figure; 
% semilogy(l,s2,'-s','Color',[0.6350 0.0780 0.1840],'MarkerIndices',1:9:k0,...
% 'MarkerSize',8,'MarkerFaceColor',[0.6350 0.0780 0.1840],'LineWidth',1.5);
% hold on; 
% semilogy(l,s22,'-o','Color',[0 0 0.8],'MarkerIndices',1:9:k0,...
%     'MarkerSize',6,'MarkerFaceColor',[0 0 0.8],'LineWidth',1.5);
% handle=legend('$\|I_{k}-\widetilde{V}_{k}^{T}\widetilde{V}_{k}\|$, $\tau_{1}$',...
%     '$\|I_{k}-\widetilde{V}_{k}^{T}\widetilde{V}_{k}\|$, $\tau_{2}$','Location','southeast');
% set(handle,'Fontsize',14,'interpreter','latex');
% xlabel('Iteration','Fontsize',15);
% ylabel('Orthogonality level','Fontsize',15);
% grid on;
% %set(gca, 'GridAlpha', 0.2);
% set(gca, 'MinorGridAlpha', 0.02);

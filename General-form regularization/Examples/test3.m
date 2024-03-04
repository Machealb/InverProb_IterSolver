% Test convergence bahavior for pGKB based pure iterative regularization 
% method pKGKB_DP/LC for reconcstructing 1D piecewise constant signal,
% and comprare it with the mLSQR method.
% 
% The regularization method M is constructed from the TV regularization, where
% we use the lagged diffusivity fixed point iteration method to linearize TV at 
% the point of x_true.
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
delta = 1e-6;
M1 = M + delta*eye(n);
alpha = 1;
G = A'*A + alpha*M;
xn = norm(x_true);
eta = 1.001 * nel * norm(b_true);

% compare pGKB and jbd method
tol = 0;
k = 20;  
er0 = zeros(k,1);
er1 = zeros(k,1);
er2 = zeros(k,1);
er3 = zeros(k,1);

[X0, res0, eta0] = mLSQR(A, M1, b, k, 1);
[X1, res1, iterstop1] = pGKBSPR_DP(A, b, M, 1, k, tol, 1, eta);
[X2, res2, iterstop2] = pGKBSPR_DP(A, b, M, 0.1, k, tol, 1, eta);
[X3, res3, iterstop3] = pGKBSPR_DP(A, b, M, 10, k, tol, 1, eta);


for i =1:k
    er0(i) = norm(x_true-X0(:,i)) / xn;
    er1(i) = norm(x_true-X1(:,i)) / xn;
    er2(i) = norm(x_true-X2(:,i)) / xn;
    er3(i) = norm(x_true-X3(:,i)) / xn;
end

[~, k0] = min(er2);

%-------- plot ------------------
lw = 2; l = 1:1:n;

figure; 
plot(l, x_true,'b-', 'LineWidth', 3.0);
% hold on
% plot(l, b,'g-.', 'LineWidth', 2.0);
hold on
plot(l, X2(:,k0),'r--', 'LineWidth', 3.0);
legend('True', 'Reconstructed');
%legend('True', 'Blurred', 'Reconstructed');
ylim([-0.6 0.8]);

figure;
semilogy(1:k, er0, 'bo-', 'LineWidth', 2.0);
hold on;
semilogy(1:k, er1, 'rx-', 'LineWidth', 2.0);
hold on;
semilogy(1:k, er2, 'm>-', 'LineWidth', 2.0);
hold on;
semilogy(1:k, er3, 'gs-', 'LineWidth', 2.0);
xlabel('Iteration','Fontsize',16);
legend('MLSQR', 'pGKB\_SPR, \alpha=1', 'pGKB\_SPR, \alpha=0.1',...
    'pGKB\_SPR, \alpha=10', 'Fontsize',15, 'Location', 'southeast');
ylabel('Relative error','Fontsize',16);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.3);
set(gca, 'MinorGridAlpha', 0.01);


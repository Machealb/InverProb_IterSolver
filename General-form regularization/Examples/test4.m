% Compare reconstructed 1D solutions with LSQR 
%

clear, clc;
directory = pwd;
path(directory, path)
addpath(genpath('..'))
rng(2023);  

% test problems 
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
delta = 1e-6;
M1 = M + delta*eye(n);
alpha = 1;
G = A'*A + alpha*M;
xn = norm(x_true);

% compare pGKB and jbd method
tol = 1e-6;
k = 25;  
er0 = zeros(k,1);
er1 = zeros(k,1);
er2 = zeros(k,1);

[X0, res0, eta0] = mLSQR(A, M1, b, k, 1, tol);
[X1,rho,eta] = lsqr_b(A,b,k,1);
[X2, res2, eta2, iterstop2, info2] = pGKBSPR_LC(A, b, M, alpha, k, tol, 1);

for i =1:k
    er0(i) = norm(x_true-X0(:,i)) / xn;
    er1(i) = norm(x_true-X1(:,i)) / xn;
    er2(i) = norm(x_true-X2(:,i)) / xn;
end

[~, k0] = min(er0);
[~, k1] = min(er1);
[~, k2] = min(er2);

%-------- plot ------------------
lw = 2; l = 1:1:n;

figure; 
plot(l, x_true,'b--', 'LineWidth', 2.0);
% hold on
% plot(l, b,'g-.', 'LineWidth', 2.0);
hold on
plot(l, X0(:,k0),'r-', 'LineWidth', 2.0);
legend('True solution', 'Best solution, MLSQR','fontsize',15);
ylim([-0.6 0.8]);

figure; 
plot(l, x_true,'b--', 'LineWidth', 2.0);
% hold on
% plot(l, b,'g-.', 'LineWidth', 2.0);
hold on
plot(l, X1(:,k1),'r-', 'LineWidth', 2.0);
legend('True solution', 'Best solution, LSQR','fontsize',15);
ylim([-0.6 0.8]);


figure; 
plot(l, x_true,'b--', 'LineWidth', 2.0);
% hold on
% plot(l, b,'g-.', 'LineWidth', 2.0);
hold on
plot(l, X2(:,k2),'r-', 'LineWidth', 2.0);
legend('True solution', 'Best solution, pGKB\_SPR','fontsize',15);
ylim([-0.6 0.8]);


figure;
semilogy(1:k, er2, 'rd-', 'LineWidth', 2.0);
hold on;
semilogy(1:k, er0, 'bv-', 'LineWidth', 2.0);
hold on;
semilogy(1:k, er1, 'gs-', 'LineWidth', 2.0);
xlabel('Iteration','Fontsize',16);
legend('pGKB\_SPR','MLSQR', 'LSQR', 'Fontsize',15, 'Location', 'southeast');
ylabel('Relative error','Fontsize',16);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.3);
set(gca, 'MinorGridAlpha', 0.01);

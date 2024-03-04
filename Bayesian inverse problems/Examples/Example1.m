% Compare SPR with hybrid method, the baseline is the optimal
% Tikhonov regularized solution
%
% Haibo Li, School of Mathematics and Statistics, The University of Melbourne
% 06, Oct, 2023.

clear, clc;
directory = pwd;
path(directory, path)
addpath(genpath('..'))
rng(2023);  

% test problems
[A, b_true, x_true] = gravity(2000);  % x \in [0,1]
a1 = 0;  a2 = 1;  b1 = 0;  b2 = 1;
% [A, b_true, x_true] =shaw(2000);  % x \in [-pi/2,pi/2]
% a1 = -pi/2;  a2 = pi/2;  b1 = -pi/2;  b2 = pi/2;

% add noise
nel = 1e-3; % Noise level
[e, Sigma] = genNoise(b_true, nel, 'white');
b = b_true + e;

% prepare algorithms
[m, n] = size(A);
M = diag(Sigma);
Lm = sqrt(1./M);
Lm = diag(Lm);
N = gen_kernel1d(a1, a2, n, 'gauss', 0.1);
N = N + 1e-10*eye(n);
% N = gen_kernel1d(a1, a2, n, 'exp', 0.1, 1);
% N = eye(n);
tau = 1.01;
Ln = chol(inv(N));
reorth = 1;
tol = 0;
k = 20;  

[x_opt, ~, lambda_opt] = Tikopt(Lm*A, Ln, Lm*b, x_true);
Lam_opt = lambda_opt^2;

[X1, res1, iterstop1] = genGKBSPR_DP(A, b, M, N, k, tol, reorth, tau);
[X2, res2, eta2, iterstop2, info2] = genGKBSPR_heu(A, b, M, N, k, tol, reorth, 'Lcurve');
[X3, res3, eta3, iterstop3, gcv3] = genGKBSPR_heu(A, b, M, N, k, tol, reorth, 'GCV');
[X4, res4, Lam4, GCV4, iterstop4] = genGKBhyb_wgcv(A, b, M, N, k, tol, 1);

er1 = zeros(k,1);  % errors of SPR_DP
er2 = zeros(k,1);  % errors of SPR_LC
er3 = zeros(k,1);  % errors of SPR_GCV
er4 = zeros(k,1);  % errors of hyb_WGCV

xn = norm(x_true);
for i =1:k
    er1(i) = norm(x_true-X1(:,i)) / xn;
    er2(i) = norm(x_true-X2(:,i)) / xn;
    er3(i) = norm(x_true-X3(:,i)) / xn;
    er4(i) = norm(x_true-X4(:,i)) / xn;
end
er_opt = norm(x_true-x_opt) / xn;  % relative error

%Ni = inv(N);
% xn_w = nm_w(x_true, Ni);
% for i =1:k
%     er1(i) = nm_w(x_true-X1(:,i), Ni) / xn_w;
%     er2(i) = nm_w(x_true-X2(:,i), Ni) / xn_w;
%     er3(i) = nm_w(x_true-X3(:,i), Ni) / xn_w;
%     er4(i) = nm_w(x_true-X4(:,i), Ni) / xn_w;
% end
% er_opt = nm_w(x_true-x_opt, Ni) / xn_w;


[spr_opt, k0] = min(er1);
[~, I1] = vec2fun(x_true, a1, a2);
[~, I2] = vec2fun(b_true, b1, b2);

%-------- plot ------------------
lw = 2; l = 1:1:n;

figure; 
plot(I1, x_true,'b-', 'LineWidth', 2.0);
hold on;
plot(I1, X1(:,k0),'g--', 'LineWidth', 2.0);
legend('True','Reconstructed');
% xlim([-pi/2 pi/2]);

figure; 
plot(I2, b,'m-', 'LineWidth', 2.0);
legend('Blurred');
% xlim([-pi/2 pi/2]);


figure;
semilogy(1:k, er1, 'rx-', 'LineWidth', 2.0);
hold on;
semilogy(1:k, er2, 'm>-', 'LineWidth', 2.0);
hold on;
semilogy(1:k, er3, 'gs-', 'LineWidth', 2.0);
hold on;
semilogy(1:k, er4, 'bo-', 'LineWidth', 2.0);
hold on;
semilogy(1:k, er_opt*ones(k,1), 'k-', 'LineWidth', 3.0);
xlabel('Iteration','Fontsize',15);
legend('SPR\_DP', 'SPR\_LC', 'SPR\_GCV', 'hyb\_WGCV', ...
    'Tikh-opt', 'Fontsize',15, 'Location', 'southeast');
ylabel('Relative error','Fontsize',15);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.3);
set(gca, 'MinorGridAlpha', 0.01);


figure;
semilogy(1:k, Lam4, 'b*-', 'LineWidth', 2.0);
hold on;
semilogy(1:k, Lam_opt*ones(k,1), 'k-', 'LineWidth', 3.0);
xlabel('Iteration','Fontsize',15);
legend('hyb\_WGCV', 'Tikh-opt', 'Fontsize',15, 'Location', 'southeast');
ylabel('Regularization parameter','Fontsize',14);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.3);
set(gca, 'MinorGridAlpha', 0.01);
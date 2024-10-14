% Compare Lagrange/PNT with hybrid method
% (the baseline is the optimal Tikhonov regularized solution.)
% The baseline should be Tikhonov with DP!!!
%
% Haibo Li, School of Mathematics and Statistics, The University of Melbourne
% 29, Feb, 2024.

clear, clc;
directory = pwd;
path(directory, path)
addpath(genpath('..'))
rng(2023);  

% test problems
[A, b_true, x_true] = shaw(5000);  % x \in [-pi/2,pi/2]
a1 = -pi/2;  a2 = pi/2;  b1 = -pi/2;  b2 = pi/2;
% [A, b_true, x_true] = heat(5000);  % x \in [0,1]
% a1 = 0;  a2 = 1;  b1 = 0;  b2 = 1;


% add noise
nel = 1e-2; % Noise level
[e, Sigma] = genNoise(b_true, nel, 'white');
% [e, Sigma] = genNoise(b_true, nel, 'nonwt');
b = b_true + e;

% prepare algorithms
[m, n] = size(A);
M = diag(Sigma);
Lm = sqrt(1./M);
Lm = diag(Lm);
N = gen_kernel1d(a1, a2, n, 'gauss', 0.1);
N = N + 1e-10*eye(n);
% N = gen_kernel1d(a1, a2, n, 'exp', 0.1, 1);
Ln = chol(inv(N));
k = 40;  

[x_opt, ~, alpha_opt] = Tikopt(Lm*A, Ln, Lm*b, x_true);
[x_dp, alpha_dp] = TikDP(Lm*A, Ln, Lm*b, 1.00);
lamb_opt = 1.0 / alpha_opt^2;
lamb_dp  = 1.0 / alpha_dp^2;

x0 = zeros(n,1);
lamb0 = 0.1;
tol = 1e-30;
method = 'direct';
[X0, res0, nx0, nh0, Lamb0] = PNT(A, b, M, N, k, lamb0, tol);
[X1, res1, nx1, nh1, Lamb1] = Newton(A, b, M, N, k, x0, lamb0, tol, method);
[X2, res2, alp2, GCV2, iterstop2] = genGKBhyb_wgcv(A, b, M, N, k, 0, 1);
[X3, res3, nx3, nh3, Lamb3] = Ch_PNT(A, b, M, N, k, lamb0, tol);

k0  = size(X0,2);
k1  = size(X1,2);
k2  = size(X2,2);
k3  = size(X3,2);
kk = max([k0,k1,k2,k3]);
Lamb2 = zeros(kk);
for i = 1:k2
    Lamb2(i) = 1.0 / alp2(i);
end

xn = norm(x_true);
er_dp = norm(x_true-x_dp) / xn;  % relative error
er_opt = norm(x_true-x_opt) / xn;  % relative error
er0 = zeros(kk,1);  % errors of PNT
er1 = zeros(kk,1);  % errors of Lagrange
er2 = zeros(kk,1);  % errors of hyb
er3 = zeros(kk,1);  % errors of Ch_PNT
for i =1:k0
    er0(i) = norm(x_true-X0(:,i)) / xn;
end
for i =1:k1
    er1(i) = norm(x_true-X1(:,i)) / xn;
end
for i =1:k2
    er2(i) = norm(x_true-X2(:,i)) / xn;
end
for i =1:k3
    er3(i) = norm(x_true-X3(:,i)) / xn;
end

[~, I1] = vec2fun(x_true, a1, a2);
[~, I2] = vec2fun(b_true, b1, b2);


%%% -------- plot ----------------------------------
figure; 
plot(I1, x_true,'b-', 'LineWidth', 2.0);
legend('True solution','fontsize',15);
xlim([-pi/2 pi/2]);
xticks(-pi/2:pi/4:pi/2)
xticklabels({'-\pi/2', '-\pi/4', '0', '\pi/4', '\pi/2'});

figure; 
plot(I2, b,'r', 'LineWidth', 2.0);
legend('Noisy data','fontsize',15);
xlim([-pi/2 pi/2]);
xticks(-pi/2:pi/4:pi/2)
xticklabels({'-\pi/2', '-\pi/4', '0', '\pi/4', '\pi/2'});


figure;
semilogy(1:k0, er0(1:k0), '-d','Color',[0.8500 0.3250 0.0980],'MarkerIndices',1:1:k0,...
    'MarkerSize',5,'MarkerFaceColor',[0.8500 0.3250 0.0980],'LineWidth',1.3);
hold on;
semilogy(1:k3, er3(1:k3), '-s','Color','b','LineWidth',1.3);
hold on;
semilogy(1:k1, er1(1:k1), '-v','Color',[0.4660 0.6740 0.1880],'MarkerIndices',1:1:k1,...
    'MarkerSize',5,'MarkerFaceColor',[0.4660 0.6740 0.1880],'LineWidth',1.3);
hold on;
semilogy(1:k2, er2(1:k2), '-o','Color',[0.3010 0.7450 0.9330],'MarkerIndices',1:1:k2,...
    'MarkerSize',5,'MarkerFaceColor',[0.3010 0.7450 0.9330],'LineWidth',1.3);
xlabel('Iteration','fontsize',16);
hold on;
semilogy(1:kk, er_dp*ones(kk,1), '-','Color',[0.4940 0.1840 0.5560], 'LineWidth', 2.3);
hold on;
semilogy(1:kk, er_opt*ones(kk,1), '--','Color','k', 'LineWidth', 2.3);
legend('PNT','Ch-PNT', 'Newton', 'genHyb', 'Tikh-DP','Tikh-opt', 'Location', 'northeast','fontsize',15);
ylabel('$\|x_{k}-x_{\mathrm{true}}\|_2/\|x_{\mathrm{true}}\|_2$','interpreter','latex','fontsize',17);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.1);
set(gca, 'MinorGridAlpha', 0.01);


figure;
semilogy(1:k0, Lamb0(1:k0), '-d','Color',[0.8500 0.3250 0.0980],'MarkerIndices',1:1:k0,...
    'MarkerSize',5,'MarkerFaceColor',[0.8500 0.3250 0.0980],'LineWidth',1.3);
hold on;
semilogy(1:k3, Lamb3(1:k3), '-s','Color','b','lineWidth',1.3);
hold on;
semilogy(1:k1, Lamb1(1:k1), '-v','Color',[0.4660 0.6740 0.1880],'MarkerIndices',1:1:k1,...
    'MarkerSize',5,'MarkerFaceColor',[0.4660 0.6740 0.1880],'LineWidth',1.3);
hold on;
semilogy(1:k2, Lamb2(1:k2), '-o','Color',[0.3010 0.7450 0.9330],'MarkerIndices',1:1:k2,...
    'MarkerSize',5,'MarkerFaceColor',[0.3010 0.7450 0.9330],'LineWidth',1.3);
hold on;
semilogy(1:kk, lamb_dp*ones(kk,1), '-','Color',[0.4940 0.1840 0.5560], 'LineWidth', 2.3);
hold on;
semilogy(1:kk, lamb_opt*ones(kk,1), '--','Color','k', 'LineWidth', 2.3);
xlabel('Iteration','Fontsize',16);
legend('PNT','Ch-PNT', 'Newton','genHyb', 'Tikh-DP', 'Tikh-opt','Fontsize',14, 'Location', 'southeast');
ylabel('$\lambda_k$','interpreter','latex','Fontsize',17);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.1);
set(gca, 'MinorGridAlpha', 0.01);


figure;
semilogy(1:k0, nh0, '->','Color','b','MarkerIndices',1:1:k0,...
    'MarkerSize',5,'MarkerFaceColor','b','LineWidth',1.5);
hold on;
semilogy(1:k3, nh3, '-d','Color',[0.4660 0.6740 0.1880],'LineWidth',1.5);
hold on;
semilogy(1:k1, nh1, '-*','Color',[1,0.47,0.1],'MarkerIndices',1:1:k1,...
    'MarkerSize',5,'MarkerFaceColor',[1,0.47,0.1],'LineWidth',1.5);
xlabel('Iteration','Fontsize',16);
legend('PNT, $h(x_{k},\lambda_{k})$','Ch-PNT, $h(x_{k},\lambda_{k})$','Newton, $h_{w}(x_{k},\lambda_{k})$',...
    'interpreter','latex', 'Fontsize',16, 'Location', 'northeast');
ylabel('Function value ','Fontsize',16);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.1);
set(gca, 'MinorGridAlpha', 0.01);


%---------- reconstructed solutions  ---------
% figure; 
% plot(I1, x_true,'b-', 'LineWidth', 2.0);
% hold on
% plot(I1, X0(:,end),'m--', 'LineWidth', 2.0);
% legend('True sol.', 'PNT sol.', 'fontsize',15);
% xlim([-pi/2 pi/2]);
% xticks(-pi/2:pi/4:pi/2)
% xticklabels({'-\pi/2', '-\pi/4', '0', '\pi/4', '\pi/2'});
% 
% figure; 
% plot(I1, x_true,'b-', 'LineWidth', 2.0);
% hold on
% plot(I1, X1(:,end),'m--', 'LineWidth', 2.0);
% legend('True sol.', 'Newton sol.', 'fontsize',15);
% xlim([-pi/2 pi/2]);
% xticks(-pi/2:pi/4:pi/2)
% xticklabels({'-\pi/2', '-\pi/4', '0', '\pi/4', '\pi/2'});
% 
% figure; 
% plot(I1, x_true,'b-', 'LineWidth', 2.0);
% hold on
% plot(I1, X2(:,end),'m--', 'LineWidth', 2.0);
% legend('True sol.', 'genHyb sol.','fontsize',15);
% xlim([-pi/2 pi/2]);
% xticks(-pi/2:pi/4:pi/2)
% xticklabels({'-\pi/2', '-\pi/4', '0', '\pi/4', '\pi/2'});
% 
% figure; 
% plot(I1, x_true,'b-','LineWidth', 2.0);
% hold on
% plot(I1, x_dp,'m--', 'LineWidth', 2.0);
% legend('True sol.', 'Tikh-DP sol.','fontsize',15);
% xlim([-pi/2 pi/2]);
% xticks(-pi/2:pi/4:pi/2)
% xticklabels({'-\pi/2', '-\pi/4', '0', '\pi/4', '\pi/2'});
% 
% figure; 
% plot(I1, x_true,'b-','LineWidth', 2.0);
% hold on
% plot(I1, x_opt,'m--', 'LineWidth', 2.0);
% legend('True sol.', 'Tikh-opt sol.','fontsize',15);
% xlim([-pi/2 pi/2]);
% xticks(-pi/2:pi/4:pi/2)
% xticklabels({'-\pi/2', '-\pi/4', '0', '\pi/4', '\pi/2'});


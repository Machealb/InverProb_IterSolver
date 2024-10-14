% Compare PNT_md with PNT and Newton method

clear, clc;
directory = pwd;
path(directory, path)
addpath(genpath('..'))
rng(2023);  

% test problems
% [A, b_true, x_true] = heat(2000);  % x \in [0,1]
% a1 = 0;  a2 = 1;  b1 = 0;  b2 = 1;
[A, b_true, x_true] = shaw(3000);  % x \in [-pi/2,pi/2]
a1 = -pi/2;  a2 = pi/2;  b1 = -pi/2;  b2 = pi/2;

% add noise
nel = 1e-2; % Noise leve
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
% N = eye(n);
Ln = chol(inv(N));
k = 30;  
k0 = 10;

% [x_opt, ~, alpha_opt] = Tikopt(Lm*A, Ln, Lm*b, x_true);
[x_dp, alpha_dp] = TikDP(Lm*A, Ln, Lm*b, 1.0001);
lamb_dp = 1.0 / alpha_dp^2;

x0 = zeros(n,1);
lamb0 = 0.1;
tol = 1e-30;
method = 'direct';
[X0, res0, nx0, nh0, Lamb0, condJ] = PNT1(A, b, M, N, k, lamb0, tol);
[X3, res3, nx3, nh3, Lamb3] = PNT_md(A, b, M, N, k0, k, lamb0, tol);

k0  = size(X0,2);
k3  = size(X3,2);
kk  = max([k0,k3]);

xn = norm(x_true);
er_dp = norm(x_true-x_dp) / xn;  % relative error
er0 = zeros(kk,1);  % errors of PNT
er3 = zeros(kk,1);  % errors of PNT_md
for i =1:k0
    er0(i) = norm(x_true-X0(:,i)) / xn;
end
for i =1:k3
    er3(i) = norm(x_true-X3(:,i)) / xn;
end

[~, I1] = vec2fun(x_true, a1, a2);
[~, I2] = vec2fun(b_true, b1, b2);


%%% -------- plot ----------------------------------
% figure; 
% plot(I1, x_true,'b-', 'LineWidth', 2.0);
% legend('True solution','fontsize',15);
% xlim([-pi/2 pi/2]);
% xticks(-pi/2:pi/4:pi/2)
% xticklabels({'-\pi/2', '-\pi/4', '0', '\pi/4', '\pi/2'});
% 
% figure; 
% plot(I2, b,'r', 'LineWidth', 2.0);
% legend('Noisy data','fontsize',15);
% xlim([-pi/2 pi/2]);
% xticks(-pi/2:pi/4:pi/2)
% xticklabels({'-\pi/2', '-\pi/4', '0', '\pi/4', '\pi/2'});


figure;
semilogy(1:k0, er0(1:k0), '-d','Color',[0.8500 0.3250 0.0980],'MarkerIndices',1:1:k0,...
    'MarkerSize',5,'MarkerFaceColor',[0.8500 0.3250 0.0980],'LineWidth',1.3);
hold on;
semilogy(1:k3, er3(1:k3), '-o','Color',[0 0.4470 0.7410],'MarkerIndices',1:1:k3,...
    'MarkerSize',5,'MarkerFaceColor',[0 0.4470 0.7410],'LineWidth',1.3);
hold on;
semilogy(1:kk, er_dp*ones(kk,1), '-','Color',[0.4940 0.1840 0.5560], 'LineWidth', 2);
legend('PNT','PNT-md','Tikh-DP', 'Location', 'northeast','fontsize',15);
ylabel('$\|x_{k}-x_{\mathrm{true}}\|_2/\|x_{\mathrm{true}}\|_2$','interpreter','latex','fontsize',17);
% ylabel('Relative  error','fontsize',16);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.1);
set(gca, 'MinorGridAlpha', 0.01);


figure;
semilogy(1:k0, Lamb0(1:k0), '-d','Color',[0.8500 0.3250 0.0980],'MarkerIndices',1:1:k0,...
    'MarkerSize',5,'MarkerFaceColor',[0.8500 0.3250 0.0980],'LineWidth',1.3);
hold on;
semilogy(1:k3, Lamb3(1:k3), '-o','Color',[0 0.4470 0.7410],'MarkerIndices',1:1:k3,...
    'MarkerSize',5,'MarkerFaceColor',[0 0.4470 0.7410],'LineWidth',1.3);
xlabel('Iteration','Fontsize',16);
hold on;
semilogy(1:kk, lamb_dp*ones(kk,1), '-','Color',[0.4940 0.1840 0.5560], 'LineWidth', 2);
legend('PNT','PNT-md', 'Tikh-DP', 'Fontsize',15, 'Location', 'southeast');
ylabel('$\lambda_k$','interpreter','latex','Fontsize',16);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.1);
set(gca, 'MinorGridAlpha', 0.01);


figure;
semilogy(1:k0, nh0, '->','Color','b','MarkerIndices',1:1:k0,...
    'MarkerSize',5,'MarkerFaceColor','b','LineWidth',1.5);
hold on;
semilogy(1:k3, nh3, '-d','Color',[0.6350 0.0780 0.1840],'MarkerIndices',1:1:k3,...
    'MarkerSize',5,'MarkerFaceColor',[0.6350 0.0780 0.1840],'LineWidth',1.5);
xlabel('Iteration','Fontsize',16);
legend('PNT','PNT-md','Fontsize',15, 'Location', 'northeast');
ylabel('$h(x_{k},\lambda_{k})$','Fontsize',17,'interpreter','latex');
grid on;
grid minor;
set(gca, 'GridAlpha', 0.1);
set(gca, 'MinorGridAlpha', 0.01);


figure;
semilogy(1:k0, condJ(1:k0), '-o','Color','m','MarkerIndices',1:1:k0,...
    'MarkerSize',5,'MarkerFaceColor','m','LineWidth',1.3);
xlabel('Iteration','Fontsize',16);
ylabel('$\kappa(J^{(k)})$','interpreter','latex','fontsize',17);
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
% legend('True sol.', 'Lagrange sol.', 'fontsize',15);
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
figure; 
plot(I1, x_true,'b-', 'LineWidth', 2.0);
hold on
plot(I1, x_dp,'m--', 'LineWidth', 2.0);
legend('True sol.', 'Tikh-DP sol.','fontsize',15);
% xlim([-pi/2 pi/2]);
% xticks(-pi/2:pi/4:pi/2)
% xticklabels({'-\pi/2', '-\pi/4', '0', '\pi/4', '\pi/2'});


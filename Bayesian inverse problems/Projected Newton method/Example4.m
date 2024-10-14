% Compare PNT with hybrid method using 2D image reconstructing problems.
% See 
% [1]. S. Gazzola, P. C. Hansen, and J. G. Nagy, IR Tools: A MATLAB package of 
% iterative regularization methods and large-scale test problems, Numer. Algor., 81 (2019), 
% pp. 773–811. 
% for more details.
%
% Haibo Li, School of Mathematics and Statistics, The University of Melbourne
% 01, Mar, 2024.
%
clear, clc;
directory = pwd;
path(directory, path)
% path([directory, '/regu'], path)
% path([directory, '/IRtools-master'], path)
addpath(genpath('..'))
IRtools_setup;
rng(2023); 

% create image data
n1 = 128;
% img = imread('HSTgray.jpg');
img = imread('shepplogan.png');
img1 = imresize(img,[n1,n1]);
optblur.trueImage = img1;
optblur.BlurLevel = 'mild';
optblur.CommitCrime = 'on';
% optblur.BC = 'zero';
% [A, b_true, x_true, ProbInfo] = PRblurspeckle(n1,optblur);
[A, b_true, x_true, ProbInfo] = PRblurshake(n1,optblur);
N1 = ProbInfo.xSize(1);  N2 = ProbInfo.xSize(2); 
a1 = 1;  a2 = n1;  b1 = 1;  b2 = n1;

% add noise
nel = 1e-3; % Noise level
% [e, Sigma] = genNoise(b_true, nel, 'white');
[e, Sigma] = genNoise(b_true, nel, 'nonwt');
b = b_true + e;
[~, NoiseInfo] = PRnoise(b_true, nel);  % obtain NoiseInfo

% prepare algorithms
[m, n] = size(A);
M = diag(Sigma);
N = gen_kernel2d(a1, a2, b1, b2, n1, 'gauss', 10);
N = N + 1e-10*eye(n);

% compute
lamb0 = 0.1;
tol = 1e-30;
k = 350;
k0 = 150;

[X1, res1, nx1, nh1, Lamb1, condJ] = PNT1(A, b, M, N, k, lamb0, tol);
% [X1, res1, nx1, nh1, Lamb1] = PNT1(A, b, M, N, k, lamb0, tol);
[X2, res2, alp2, GCV2, iterstop2] = genGKBhyb_wgcv(A, b, M, N, k, 0, 1);
[X3, res3, nx3, nh3, Lamb3] = PNT_md(A, b, M, N, k0, k, lamb0, tol);

k1  = size(X1,2);
k2  = size(X2,2); 
k3  = size(X3,2);
Lamb2 = zeros(k2);

for i = 1:k2
    Lamb2(i) = 1.0 / alp2(i);
end

xn = norm(x_true);
er1 = zeros(k1);  % errors of PNT
er2 = zeros(k2);  % errors of hyb
er3 = zeros(k3);  % errors of PNT-md
for i =1:k1
    er1(i) = norm(x_true-X1(:,i)) / xn;
end
for i =1:k2
    er2(i) = norm(x_true-X2(:,i)) / xn;
end
for i =1:k3
    er3(i) = norm(x_true-X3(:,i)) / xn;
end



%----------- plot -----------------------------------
figure;
PRshowx(x_true, ProbInfo)
title('True image','interpreter','latex','fontsize',18)
set(gca,'fontsize',18)

figure;
PRshowb(b, ProbInfo)
title('Noisy data','interpreter','latex','fontsize',18)
set(gca,'fontsize',18)


figure;
semilogy(1:k1, er1(1:k1), '-d','Color',[0.8500 0.3250 0.0980],'MarkerIndices',1:9:k1,...
    'MarkerSize',5,'MarkerFaceColor',[0.8500 0.3250 0.0980],'LineWidth',1.3);
hold on;
semilogy(1:k3, er3(1:k3), '-v','Color',[0.4940 0.1840 0.5560],'MarkerIndices',1:9:k3,...
    'MarkerSize',5,'MarkerFaceColor',[0.4940 0.1840 0.5560],'LineWidth',1.3);
hold on;
semilogy(1:k2, er2(1:k2), '-o','Color',[0.3010 0.7450 0.9330],'MarkerIndices',1:9:k2,...
    'MarkerSize',5,'MarkerFaceColor',[0.3010 0.7450 0.9330],'LineWidth',1.3);
xlabel('Iteration','fontsize',16);
legend('PNT', 'PNT-md', 'genHyb', 'Location', 'northeast','fontsize',15);
ylabel('$\|x_{k}-x_{\mathrm{true}}\|_2/\|x_{\mathrm{true}}\|_2$','interpreter','latex','fontsize',17);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.1);
set(gca, 'MinorGridAlpha', 0.01);


figure;
semilogy(1:k1, Lamb1(1:k1), '-d','Color',[0.8500 0.3250 0.0980],'MarkerIndices',1:9:k1,...
    'MarkerSize',5,'MarkerFaceColor',[0.8500 0.3250 0.0980],'LineWidth',1.3);
hold on;
semilogy(1:k3, Lamb3(1:k3), '-v','Color',[0.4940 0.1840 0.5560],'MarkerIndices',1:9:k3,...
    'MarkerSize',5,'MarkerFaceColor',[0.4940 0.1840 0.5560],'LineWidth',1.3);
hold on;
semilogy(1:k2, Lamb2(1:k2), '-o','Color',[0.3010 0.7450 0.9330],'MarkerIndices',1:9:k2,...
    'MarkerSize',5,'MarkerFaceColor',[0.3010 0.7450 0.9330],'LineWidth',1.3);
xlabel('Iteration','Fontsize',16);
legend('PNT', 'PNT-md', 'genHyb', 'Fontsize',15, 'Location', 'southeast');
ylabel('$\lambda_k$','interpreter','latex','Fontsize',16);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.1);
set(gca, 'MinorGridAlpha', 0.01);


figure;
semilogy(1:k1, nh1(1:k1), '->','Color','b','MarkerIndices',1:9:k1,...
    'MarkerSize',5,'MarkerFaceColor','b','LineWidth',1.5);
hold on;
semilogy(1:k3, nh3(1:k3), '-o','Color',[0.6350 0.0780 0.1840],'MarkerIndices',1:9:k3,...
    'MarkerSize',5,'MarkerFaceColor',[0.6350 0.0780 0.1840],'LineWidth',1.5);
xlabel('Iteration','Fontsize',16);
ylabel('$h(x_{k},\lambda_{k})$', 'interpreter','latex', 'Fontsize',16);
legend('PNT', 'PNT-md','Fontsize',15, 'Location', 'southeast');
grid on;
grid minor;
set(gca, 'GridAlpha', 0.1);
set(gca, 'MinorGridAlpha', 0.01);

figure;
semilogy(1:k1, condJ(1:k1), '-o','Color','m','MarkerIndices',1:9:k1,...
    'MarkerSize',5,'MarkerFaceColor','m','LineWidth',1.3);
xlabel('Iteration','Fontsize',16);
ylabel('$\kappa(J^{(k)})$','interpreter','latex','fontsize',17);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.1);
set(gca, 'MinorGridAlpha', 0.01);


%-----------------------
figure;
PRshowx(X1(:,k1), ProbInfo)
title('PNT sol.','interpreter','latex','fontsize',18);
set(gca,'fontsize',18)

figure;
PRshowx(X2(:,k2), ProbInfo)
title('genHyb sol.','interpreter','latex','fontsize',18);
set(gca,'fontsize',18)




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
options.phantomImage = 'smooth';
options.BlurLevel = 'mild';
options.CommitCrime = 'on';
[A, b_true, x_true, ProbInfo] = PRspherical(n1, options);
a1 = 1;  a2 = n1;  b1 = 1;  b2 = n1;
N1 = ProbInfo.xSize(1);  N2 = ProbInfo.xSize(2); 

% add noise
nel = 1e-2; % Noise level
% [e, Sigma] = genNoise(b_true, nel, 'white');
[e, Sigma] = genNoise(b_true, nel, 'nonwt');
b = b_true + e;
[~, NoiseInfo] = PRnoise(b_true, nel);  % obtain NoiseInfo

% prepare algorithms
[m, n] = size(A);
M = diag(Sigma);
N = gen_kernel2d(a1, a2, b1, b2, n1, 'matern', 100, 1.5);

% compute
lamb0 = 0.1;
tol = 1e-30;
k = 150;

[X1, res1, nx1, nh1, Lamb1] = PNT(A, b, M, N, k, lamb0, tol);
[X2, res2, alp2, GCV2, iterstop2] = genGKBhyb_wgcv(A, b, M, N, k, 0, 1);

k1  = size(X1,2);
k2  = size(X2,2);
Lamb2 = zeros(k2);

for i = 1:k2
    Lamb2(i) = 1.0 / alp2(i);
end

xn = norm(x_true);
er1 = zeros(k1);  % errors of PNT
er2 = zeros(k2);  % errors of hyb
for i =1:k1
    er1(i) = norm(x_true-X1(:,i)) / xn;
end
for i =1:k2
    er2(i) = norm(x_true-X2(:,i)) / xn;
end



%----------- plot -----------------------------------
figure;
PRshowx(x_true, ProbInfo)
title('True image','interpreter','latex','fontsize',18)
set(gca,'fontsize',18)
saveas(gcf,'true.png')

figure;
PRshowb(b, ProbInfo)
title('Noisy data','interpreter','latex','fontsize',18)
set(gca,'fontsize',18)
saveas(gcf,'noisy.png')


figure;
semilogy(1:k1, er1(1:k1), '-d','Color',[0.8500 0.3250 0.0980],'MarkerIndices',1:4:k1,...
    'MarkerSize',5,'MarkerFaceColor',[0.8500 0.3250 0.0980],'LineWidth',1.3);
hold on;
semilogy(1:k2, er2(1:k2), '-o','Color',[0.3010 0.7450 0.9330],'MarkerIndices',1:4:k2,...
    'MarkerSize',5,'MarkerFaceColor',[0.3010 0.7450 0.9330],'LineWidth',1.3);
xlabel('Iteration','fontsize',16);
legend('PNT', 'genHyb', 'Location', 'northeast','fontsize',15);
ylabel('$\|x_{k}-x_{\mathrm{true}}\|_2/\|x_{\mathrm{true}}\|_2$','interpreter','latex','fontsize',16);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.1);
set(gca, 'MinorGridAlpha', 0.01);


figure;
semilogy(1:k1, Lamb1(1:k1), '-d','Color',[0.8500 0.3250 0.0980],'MarkerIndices',1:4:k1,...
    'MarkerSize',5,'MarkerFaceColor',[0.8500 0.3250 0.0980],'LineWidth',1.3);
hold on;
semilogy(1:k2, Lamb2(1:k2), '-o','Color',[0.3010 0.7450 0.9330],'MarkerIndices',1:4:k2,...
    'MarkerSize',5,'MarkerFaceColor',[0.3010 0.7450 0.9330],'LineWidth',1.3);
xlabel('Iteration','Fontsize',16);
legend('PNT', 'genHyb', 'Fontsize',14, 'Location', 'southeast');
ylabel('$\lambda_k$','interpreter','latex','Fontsize',16);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.1);
set(gca, 'MinorGridAlpha', 0.01);


figure;
semilogy(1:k1, nh1(1:k1), '->','Color','b','MarkerIndices',1:4:k1,...
    'MarkerSize',5,'MarkerFaceColor','b','LineWidth',1.5);
xlabel('Iteration','Fontsize',16);
ylabel('$h(x_{k},\lambda_{k})$', 'interpreter','latex', 'Fontsize',16);
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




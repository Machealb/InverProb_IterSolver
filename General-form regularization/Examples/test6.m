% Compare reconstructed 2D solutions with LSQR 
% Using 2D large-scale test problems, and show the effect of the algorithms
% for reconstructing solutions. 
% In this test, PRdiffusion is a 2D inverse diffusion problem, where the forward
% operator A is stored as a functional handle.
%
clear, clc;
directory = pwd;
path(directory, path)
addpath('./src/')
addpath('./IRtools-master/')
IRtools_setup;
rng(2023); 


% Define test problem
n = 128;                          % Problem size.
nel = 1e-3;                       % Relative noise level in data.
options = PRset('Tfinal', 0.005, 'Tsteps', 100);
[A, b_true, x_true, ProbInfo] = PRdiffusion(n, options);        % Get the test problem.
[b, NoiseInfo] = PRnoise(b_true, nel);  % Add Gaussian noise.

% prepare algorithms
[m, n] = sizem(A);
M = LaplacianMatrix2D(n);
delta = 0;
M1 = M + delta*speye(n);
alpha = 1;
xn = norm(x_true);

% compare pGKB reguarization methods
tol = 1e-6;
k = 150;
er0 = zeros(k,1);
er1 = zeros(k,1);
er2 = zeros(k,1);

[X0, res0, eta0] = mLSQR(A, M1, b, k, 1, tol);
[X1,rho,eta] = lsqr_b(A,b,k,1);
[X2, res2, eta2, iterstop2, info2] = pGKBSPR_LC(A, b, M, alpha, k, tol, 1);

for i =1:k
    % er0(i) = norm(x_true-X0(:,i)) / xn;
    er1(i) = norm(x_true-X1(:,i)) / xn;
    % er2(i) = norm(x_true-X2(:,i)) / xn;
end

[~, k0] = min(er0);
[~, k1] = min(er1);
[~, k2] = min(er2);


%-------- plot ------------------------------
figure;
PRshowx(x_true, ProbInfo)
title('True image','interpreter','latex','fontsize',18)
set(gca,'fontsize',18)

figure;
PRshowx(b, ProbInfo)
title('Noisy data','interpreter','latex','fontsize',18)
set(gca,'fontsize',18)

figure;
PRshowx(X0(:,k0), ProbInfo)
zticks([0 0.2 0.4 0.6 0.8 1])
zticklabels({'0','0.2','0.4','0.6','0.8','1'})
title(['Best solution, MLSQR'],...
    'interpreter','latex','fontsize',18);
set(gca,'fontsize',18)


figure;
PRshowx(X1(:,k1), ProbInfo)
zticks([0 0.2 0.4 0.6 0.8 1])
zticklabels({'0','0.2','0.4','0.6','0.8','1'})
title(['Best solution, LSQR'],...
    'interpreter','latex','fontsize',18);
set(gca,'fontsize',18)

figure;
PRshowx(X2(:,k2), ProbInfo)
zticks([0 0.2 0.4 0.6 0.8 1])
zticklabels({'0','0.2','0.4','0.6','0.8','1'})
title(['Best solution, pGKB\_SPR'],...
    'interpreter','latex','fontsize',18);
set(gca,'fontsize',18)


k=150;
figure;
semilogy(1:k, er2(1:k), '-rd', 'Color', 'r', 'MarkerIndices',1:5:k,...
    'LineWidth',1.5);
hold on;
semilogy(1:k, er0(1:k), '-bv', 'MarkerIndices',1:5:k,...
    'LineWidth',1.5);
hold on;
semilogy(1:k, er1(1:k), '-gs', 'Color', 'g', 'MarkerIndices',1:5:k,...
    'LineWidth',1.5);
xlabel('Iteration','Fontsize',16);
legend('pGKB\_SPR','MLSQR', 'LSQR', 'Fontsize',15, 'Location', 'northeast');
ylabel('Relative error','Fontsize',16);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.3);
set(gca, 'MinorGridAlpha', 0.01);
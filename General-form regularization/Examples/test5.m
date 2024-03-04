% Compare reconstructed 2D solutions with LSQR 
% Using 2D large-scale test problems, and show the effect of the algorithms
% for reconstructing solutions. 
% In this test, PRdefocus is a 2D image deburring problem, where the forward
% operator A is stored as an object.

clear, clc;
directory = pwd;
path(directory, path)
% path([directory, '/regu'], path)
% path([directory, '/IRtools-master'], path)
addpath(genpath('..'))
IRtools_setup;
rng(2023); 

% create image data
% img = imread('HSTgray.jpg');
% img1 = imresize(img,[64,64]);
% optblur.trueImage = img1;
optblur.BlurLevel = 'mild';
optblur.CommitCrime = 'on';
optblur.BC = 'zero';
[A, b_true, x_true, ProbInfo] = PRblurdefocus(optblur);
N1 = ProbInfo.xSize(1);  N2 = ProbInfo.xSize(2); 
% add noise
nel = 2e-3;  % noise level
b = PRnoise(b_true, 'gauss', nel);  % Observed Image

% prepare algorithms
[m, n] = sizem(A);
L1 = genLirn(x_true, '2dTV', 1e-6, N1, N2);
M = L1' * L1;
delta = 1e-8;
M1 = M + delta*speye(n);
alpha = 0.1;
xn = norm(x_true);

% compare pGKB reguarization methods
tol = 1e-6;
k = 400;
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
title(['Best solution, MLSQR'],...
    'interpreter','latex','fontsize',18);
set(gca,'fontsize',18)


figure;
PRshowx(X1(:,k1), ProbInfo)
title(['Best solution, LSQR'],...
    'interpreter','latex','fontsize',18);
set(gca,'fontsize',18)

figure;
PRshowx(X2(:,k2), ProbInfo)
title(['Best solution, pGKB\_SPR'],...
    'interpreter','latex','fontsize',18);
set(gca,'fontsize',18)


figure;
semilogy(1:k, er2, '-rd', 'Color', 'r', 'MarkerIndices',1:14:k,...
    'LineWidth',1.5);
hold on;
semilogy(1:k, er0, '-bv', 'MarkerIndices',1:14:k,...
    'LineWidth',1.5);
hold on;
semilogy(1:k, er1, '-gs', 'Color', 'g', 'MarkerIndices',1:14:k,...
    'LineWidth',1.5);
xlabel('Iteration','Fontsize',16);
legend('pGKB\_SPR','MLSQR', 'LSQR', 'Fontsize',15, 'Location', 'northeast');
ylabel('Relative error','Fontsize',16);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.3);
set(gca, 'MinorGridAlpha', 0.01);

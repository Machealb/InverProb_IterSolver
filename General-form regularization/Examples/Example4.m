% Test for pGKB based pure iterative methods and hybird regularization method:
%   (1). pGKB_SPR with DP or L-curve as early stopping criteria, and
%   (2). pGKBhyb with secant update or WGCV for updating regularization parameters. 
%
% Using 2D large-scale test problems, and show the effect of the algorithms
% for reconstructing solutions.
% In this test, PRdefocus is a 2D image deburring problem, where the forward
% operator A is stored as an object.
% See 
% [1]. S. Gazzola, P. C. Hansen, and J. G. Nagy, IR Tools: A MATLAB package of 
% iterative regularization methods and large-scale test problems, Numer. Algor., 81 (2019), 
% pp. 773–811. 
% for more details.
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 06, July, 2023.
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
alpha = 0.1;
%G = A'*A + alpha*M;
xn = norm(x_true);
eta = 1.001 * nel * norm(b_true);

% compare pGKB reguarization methods
tol1 = 1e-6;
k = 200;
er1 = zeros(k,1);
er2 = zeros(k,1);
er3 = zeros(k,1);

[X1, res1, eta1, iterstop1, info1] = pGKBSPR_LC(A, b, M, alpha, k, tol1, 1);
[X2, res2, Lam2, iterstop2] = pGKBhyb_su(A, b, M, alpha, k, tol1, 1, eta);
[X3, res3, Lam3, GCV, iterstop3] = pGKBhyb_wgcv(A, b, M, alpha, k, tol1, 1);

for i =1:k
    er1(i) = norm(x_true-X1(:,i)) / xn;
    er2(i) = norm(x_true-X2(:,i)) / xn;
    er3(i) = norm(x_true-X3(:,i)) / xn;
end

[~, k0] = min(er1);

iterstop_DP = 0;
flag = 1;
for i = 1:k
    if flag == 1 && res1(i) <= eta
        iterstop_DP = i;
        flag = 0;
    end
end
    

%----------- plot -----------------------------------
figure;
PRshowx(x_true, ProbInfo)
title('True image','interpreter','latex','fontsize',18)
set(gca,'fontsize',18)

figure;
PRshowx(b, ProbInfo)
title('Noisy data','interpreter','latex','fontsize',18)
set(gca,'fontsize',18)

figure;
PRshowx(X1(:,k0), ProbInfo)
title(['Best solution, $k_{0}$ = ',num2str(k0)],...
    'interpreter','latex','fontsize',18);
set(gca,'fontsize',18)

% figure;
% PRshowx(X1(:,iterstop1), ProbInfo)
% title(['LC solution, $k$ = ',num2str(iterstop1+50)],...
%     'interpreter','latex','fontsize',18);
% set(gca,'fontsize',18)

figure;
PRshowx(X1(:,iterstop_DP), ProbInfo)
title(['DP solution, $k$ = ',num2str(iterstop_DP)],...
    'interpreter','latex','fontsize',18);
set(gca,'fontsize',18)

figure;
PRshowx(X2(:,iterstop2), ProbInfo)
title(['SU solution, $k$ = ',num2str(iterstop2)],...
    'interpreter','latex','fontsize',18);
set(gca,'fontsize',18)

figure;
PRshowx(X3(:,iterstop3), ProbInfo)
title(['WGCV solution, $k$ = ',num2str(iterstop3)],...
    'interpreter','latex','fontsize',18);
set(gca,'fontsize',18)


figure;
semilogy(1:k, er1, '->','Color','b','MarkerIndices',1:9:k,...
    'MarkerSize',5,'MarkerFaceColor','b','LineWidth',1.5);
hold on;
semilogy(1:k, er2, '-s','Color',[1,0.47,0.1],'MarkerIndices',1:9:k,...
    'MarkerSize',5,'MarkerFaceColor',[1.0,0.47,0.1],'LineWidth',1.5);
hold on;
semilogy(1:k, er3, '-o','Color','g','MarkerIndices',1:9:k,...
    'MarkerSize',5,'MarkerFaceColor','g','LineWidth',1.5);
hold on;
semilogy(iterstop_DP, er1(iterstop_DP),'bo', 'MarkerSize',16, 'LineWidth',2)
hold on;
semilogy(iterstop2, er2(iterstop2),'o','Color',[1,0.47,0.1]', 'MarkerSize',16, 'LineWidth',2)
hold on;
semilogy(iterstop3, er3(iterstop3),'go', 'MarkerSize',16, 'LineWidth',2)
xlabel('Iteration','fontsize',16);
legend('pGKB\_SPR', 'pGKB\_HR, SU', 'pGKB\_HR, WGCV', 'Location', 'northeast','fontsize',16);
ylabel('Relative error','fontsize',16);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.3);
set(gca, 'MinorGridAlpha', 0.01);


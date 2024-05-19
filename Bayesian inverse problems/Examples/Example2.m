% Test for Bayesian SPR using 2D large-scale test problems, and 
% show the effect of the algorithms for reconstructing solutions.
% In this test, PRdiffusion is a 2D inverse diffusion problem, where the forward
% operator A is stored as a functional handle.
% See 
% [1]. S. Gazzola, P. C. Hansen, and J. G. Nagy, IR Tools: A MATLAB package of 
% iterative regularization methods and large-scale test problems, Numer. Algor., 81 (2019), 
% pp. 773–811. 
% for more details.
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 08, Oct, 2023.
%
clear, clc;
directory = pwd;
path(directory, path)
% path([directory, '/regu'], path)
% path([directory, '/IRtools-master'], path)
addpath(genpath('..'))
IRtools_setup;
rng(2023); 


% Define test problem
n1 = 128;                          % Problem size.
options = PRset('Tfinal', 0.005, 'Tsteps', 100);
[A, b_true, x_true, ProbInfo] = PRdiffusion(n, options);        % Get the test problem.
a1 = 0;  a2 = 1;  b1 = 0;  b2 = 1;

% add noise
nel = 1e-3; % Noise level
[e, Sigma] = genNoise(b_true, nel, 'white');
b = b_true + e;
[~, NoiseInfo] = PRnoise(b_true, nel);  % obtain NoiseInfo

% prepare algorithms
[m, n] = sizem1(A);
M = diag(Sigma);
N = gen_kernel2d(a1, a2, b1, b2, n1, 'matern', 0.05, 2.5);
tau = 1.01;
reorth = 1;
tol = 0;
k = 100; 

[X1, res1, eta1, gcv1] = genGKB_SPR(A, b, M, N, k, tol, reorth);
[X2, res2, Lam2, GCV2, iterstop2] = genGKBhyb_wgcv(A, b, M, N, k, tol, 1);

er1 = zeros(k,1);
er2 = zeros(k,1);
xn = norm(x_true);
for i =1:k
    er1(i) = norm(x_true-X1(:,i)) / xn;
    er2(i) = norm(x_true-X2(:,i)) / xn;
end

[spr_opt, k0] = min(er1);

iterstop_DP = 0;
dp = tau * sqrt(m);
flag = 1;
for i = 1:k
    if flag == 1 && res1(i) <= dp
        iterstop_DP = i;
        flag = 0;
    end
end

[iterstop_LC, info_LC] = Lcurve(res1, eta1, 1);
[~, iterstop_gcv] = min(gcv1);
    

%----------- plot -----------------------------------
figure;
PRshowx(x_true, ProbInfo)
title('True solution','interpreter','latex','fontsize',18)
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

figure;
PRshowx(X1(:,iterstop_LC), ProbInfo)
title(['LC solution, $k$ = ',num2str(iterstop_LC)],...
    'interpreter','latex','fontsize',18);
set(gca,'fontsize',18)

figure;
PRshowx(X1(:,iterstop_DP), ProbInfo)
title(['DP solution, $k$ = ',num2str(iterstop_DP)],...
    'interpreter','latex','fontsize',18);
set(gca,'fontsize',18)

figure;
PRshowx(X1(:,iterstop_gcv), ProbInfo)
title(['GCV solution, $k$ = ',num2str(iterstop_gcv)],...
    'interpreter','latex','fontsize',18);
set(gca,'fontsize',18)

figure;
PRshowx(X2(:,iterstop2), ProbInfo)
title(['genHyb\_WGCV solution, $k$ = ',num2str(iterstop2)],...
    'interpreter','latex','fontsize',18);
set(gca,'fontsize',18)


figure;
semilogy(1:k, er1, '-v','Color','b','MarkerIndices',1:9:k,...
    'MarkerSize',6,'MarkerFaceColor','b','LineWidth',1.5);
hold on;
semilogy(1:k, er2, '-o','Color',[1,0.47,0.1],'MarkerIndices',1:9:k,...
    'MarkerSize',6,'MarkerFaceColor',[1.0,0.47,0.1],'LineWidth',1.5);
xlabel('Iteration','fontsize',16);
legend('genGKB\_SPR', 'genHyb\_WGCV', 'Location', 'northeast','fontsize',15);
ylabel('Relative error','fontsize',16);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.3);
set(gca, 'MinorGridAlpha', 0.01);


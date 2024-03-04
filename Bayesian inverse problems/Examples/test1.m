% Seismic Tomography
%
clear, clc;
IRtools_setup;
rng(2023);  % 伪随机数种子

% test matrices
n = 256;
options.phantomImage = 'smooth';
[A, b_true, x_true, ProbInfo] = PRspherical(n, options); 

% add noise
nel = 1e-2; % Noise level
b = AddNoise(b_true, 'gauss', nel);  % noisy data

%--------- plot ------------------
figure;
PRshowx(x_true, ProbInfo)
title('True image','interpreter','latex','fontsize',18)
set(gca,'fontsize',18)





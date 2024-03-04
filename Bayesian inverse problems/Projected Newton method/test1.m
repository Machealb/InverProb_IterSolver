clear, clc;
directory = pwd;
path(directory, path)
% path([directory, '/regu'], path)
% path([directory, '/IRtools-master'], path)
addpath(genpath('..'))
IRtools_setup;
rng(2023); 

% create image data
n1 = 64;
img = imread('shepplogan.png');
img1 = imresize(img,[n1,n1]);
optblur.trueImage = img1;
optblur.BlurLevel = 'mild';
optblur.CommitCrime = 'on';
optblur.BC = 'zero';
[A, b_true, x_true, ProbInfo] = PRblurdefocus(optblur);
N1 = ProbInfo.xSize(1);  N2 = ProbInfo.xSize(2); 
a1 = 1;  a2 = n1;  b1 = 1;  b2 = n1;

% add noise
nel = 1e-2; % Noise level
[e, Sigma] = genNoise(b_true, nel, 'white');
% [e, Sigma] = genNoise(b_true, nel, 'nonwt');
b = b_true + e;
[~, NoiseInfo] = PRnoise(b_true, nel);  % obtain NoiseInfo

% prepare algorithms
[m, n] = size(A);
M = diag(Sigma);
% N = gen_kernel2d(a1, a2, b1, b2, n1, 'matern', 1.0, 0.5);
N = gen_kernel2d(a1, a2, b1, b2, n1, 'gauss', 0.1);
% N = N + 1e-10*eye(n);

% compute
lamb0 = 0.1;
tol = 1e-30;
k = 10;

[bbeta, B, U, Z, ~] = gen_GKB(A, b, M, N, k, 0, 1);
i = 1;
Bk = B(1:i+1, 1:i); 
Ck = eye(i);

[Ub, ss, Xb] = gsvd1(Bk, Ck);



% Test basic recursive relations of gen-GKB


clear, clc;
directory = pwd;
path(directory, path)
addpath(genpath('..'))
rng(2023);  

% test problems
[A, b_true, x_true] = gravity(1000);  % x \in [0,1]
a1 = 0;  a2 = 1;  b1 = 0;  b2 = 1;


% add noise
nel = 1e-2; % Noise level
[e, Sigma] = genNoise(b_true, nel, 'white');
b = b_true + e;

% prepare algorithms
[m, n] = size(A);
M = diag(Sigma);
M_inv = diag(1.0./M);
N = gen_kernel1d(a1, a2, n, 'gauss', 1.0);
N = N + 1e-6*eye(n);
reorth = 1;
tol = 0;
k = 50;  

[bbeta, B, U, V, Vb] = gen_GKB(A, b, M, N, k+1, tol, reorth);

er1 = zeros(k-1,1);
er2 = zeros(k-1,1);
er3 = zeros(k-1,1);
er0 = norm(b-bbeta*U(:,1));

for i = 2:k
    Ik = eye(k+1);
    E1 = N*A'*M_inv*U(:,i) - B(i,i-1)*V(:,i-1) - B(i,i)*V(:,i);
    er1(i-1) = norm(E1);
    E2 = A*V(:,i) - B(i,i)*U(:,i) - B(i+1,i)*U(:,i+1);
    er2(i-1) = norm(E2);
    E3 = N*Vb(:,i) - V(:,i);
    er3(i-1) = norm(E3);
end

%---- plot ----------------
figure; 
l = 1:1:k-1;
semilogy(l, er1,'b-', 'LineWidth', 2.0);
hold on;
semilogy(l, er2,'r-', 'LineWidth', 2.0);
hold on;
semilogy(l, er3,'m-', 'LineWidth', 2.0);
legend('er1', 'er2', 'er3');




% Test for regulartization effect of mLSQR and pLSQR using some
% one dimensional linear ill-posed problems
% Compare the semi-convergenc behavior of these two methods with JBDQR. 
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 16, June, 2023.
%

clear, clc;
directory = pwd;
path(directory, path)
path([directory, '/IRtools-master'], path)
path([directory, '/IRtools-master/regu'], path)
IRtools_setup;
rng(2022);  % 伪随机数种子

% test matrices
%[A,b_true,x_true]=shaw(1024);  
%[A,b_true,x_true]=gravity(1024); 
%[A,b_true,x_true]=baart(1024);
%[A, b_true, x_true] = heat(1024);  
[A, b_true, x_true] = deriv2(1000);  

nel = 1e-4;  % Noise level
b = PRnoise(b_true, 'gauss', nel);  % noisy data

% construct L
[m,n]=size(A);
L = get_l(n,1);  M = L' * L;
M1 = M + 1e-6*eye(n);  % priorconditioner
% L = eye(n);  M = eye(n);
M2 = 0.01 * M;  % scaling M to reduce condition number of C
C = A' * A + M2;

% k-step preconditioned Lanczos bidiagonalization 
k = 50;
reorth = 2;
tol = 1e-6;
[X1, res1, eta1] = mLSQR(A, M1, b, k, reorth, tol);
[X2, res2, eta2, ~, ~, ~] = pLSQR(A, M2, b, k, reorth, tol);  
[Z, res0, ~, ~] = jbdqr(A, L, b, k, tol, reorth);
X0 = zeros(n, k);
[Q, R] = qr([A;L], 0);
for i = 1:k
    X0(:,i) = R \ (Q'*Z(:,i)); 
    %X0(:,i) = lsqr(@(z,tflag)afun(z,A,L,tflag),Z(:,i),1e-8,n);  % solve x_k by LSQR
end

xn = norm(x_true);
er0 = zeros(k,1);
er1 = zeros(k,1);
er2 = zeros(k,1);
for i =1:k
    er0(i) = norm(x_true-X0(:,i)) / xn;
    er1(i) = norm(x_true-X1(:,i)) / xn;
    er2(i) = norm(x_true-X2(:,i)) / xn;
end
% [~,k0] = min(er0);


%%--------- plot ------------------
lw = 2; 
figure; 
semilogy(1:k, er0, 'c*-', 'LineWidth', 2.0);
hold on;
semilogy(1:k, er1, 'bv-', 'LineWidth', 2.0);
hold on;
semilogy(1:k, er2, 'ro-', 'LineWidth', 2.0);
hold on;
legend('JBDQR', 'mLSQR', 'pLSQR', 'Location', 'northeast');
xlabel('Iteration');
handle = ylabel('$RE(k)$', 'Interpreter', 'latex');
set(handle,'interpreter', 'latex');
grid on;
%set(gca, 'GridAlpha', 0.2);
set(gca, 'MinorGridAlpha', 0.02);

% plot(l, x,'b-', 'LineWidth', 2.0);
% hold on
% plot(l, b,'g-.', 'LineWidth', 2.0);
% hold on
% plot(l, X(:,k0),'r--', 'LineWidth', 2.0);
% legend('True', 'Blurred', 'Reconstructed');
% % ylim([-0.8 0.8]);
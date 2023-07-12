% Test matrix-form recursive relations for mLSQR
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 23, May, 2023.
%

clear, clc;
directory = pwd;  
path(directory, path)
path([directory, '/IRtools-master'], path)
path([directory, '/IRtools-master/regu'], path)
IRtools_setup;
rng(0);  % 伪随机数种子

% construct the true signal 
n = 500;
x = zeros(n,1);
x(1:50) = 0.0;
x(51:151) = -0.2;
x(151:300) = 0.6;
x(301:400) = 0.2;
x(401:450) = -0.4;
x(451:500) = 0.0;

% transform the signal to be a sparse one
A = GaussBlur1d(n, 10);
%[A,~,~] = phillips(n);
L1 = genLirn(x, '1dTV', 1e-10);
[~, R1] = qr([A;L1], 0);
x = R1 * x;
xn = norm(x);

% blured signal
nel = 5e-2;  % Noise level
b_true = A * x;
b = PRnoise(b_true, 'gauss', nel);  % noisy data

% construct L
L = genLirn(x, 'lp', 1e-10, 1);  % very valid!
%L = eye(n);
%L = get_l(n,1);

% run matrix reduction process
k = 100;
x0 = x;
tau = 1e-10;
[beta1, M1, H1, U1, V1, Z1, X1] = fLSQR(A, b, x0, 1, tau, k, 0);
[beta2, M2, H2, U2, V2, Z2, X2] = precHR(A, L, b, k);
[beta3, B3, U3, V3, Z3, X3] = tLSQR(A, L, b, k, 1);

% chech matrix-form recursive relations
er11 = zeros(k,1);
er12 = zeros(k,1);
er21 = zeros(k,1);
er22 = zeros(k,1);
er31 = zeros(k,1);
er32 = zeros(k,1);
for i = 1:k
    r11 = A * Z1(:,1:i) - U1(:,1:i+1) * H1(1:i+1,1:i);
    er11(i) = norm(r11);
    r12 = A' * U1(:,1:i) - V1(:,1:i) * M1(1:i,1:i);
    er12(i) = norm(r12);
    r21 = A * Z2(:,1:i) - U2(:,1:i+1) * H2(1:i+1,1:i);
    er21(i) = norm(r21);
    r22 = A' * U2(:,1:i) - V2(:,1:i) * M2(1:i,1:i);
    er22(i) = norm(r22);
    r31 = A * Z3(:,1:i) - U3(:,1:i+1) * B3(1:i+1,1:i);
    er31(i) = norm(r31);
    %r32 = A' * U3(:,1:i) - L' * V3(:,1:i) * B3(1:i,1:i)';
    r32 = inv(L)' * A' * U3(:,1:i) - V3(:,1:i) * B3(1:i,1:i)';
    er32(i) = norm(r32);
end


%%---------------- plot -------------------------------
lw = 2; l = 1:1:k;

figure; 
semilogy(l, er11,'bx-', 'LineWidth', 2.0);
hold on
semilogy(l, er12,'ro-', 'LineWidth', 2.0);
legend('1-st', '2-nd');
title('fLSQR')

figure; 
semilogy(l, er21,'bx-', 'LineWidth', 2.0);
hold on
semilogy(l, er22,'ro-', 'LineWidth', 2.0);
legend('1-st', '2-nd');
title('precHR')

figure; 
semilogy(l, er31,'bx-', 'LineWidth', 2.0);
hold on
semilogy(l, er32,'ro-', 'LineWidth', 2.0);
legend('1-st', '2-nd');
title('tLSQR')

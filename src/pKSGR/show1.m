%% 研究JBD与Qa的Lanczos双对角过程的联系：

% 数值实验发现，双对角化过程在停机误差为tol时，与Qa的Lanczos双对角过程的联系：
% 对于良态矩阵： 相当于舍入误差为O(tol)的Lanczos双对角化过程
% 但是，此现象对不适定矩阵不再成立，误差会逐渐放大？ 不会放的很大(半收敛之前）！

clear;
rng(0);  % seed of random numbers
directory = pwd;
path(directory, path)
path([directory, '/Matrices'], path)
path([directory, '/regu'], path)
path([directory, '/IRtools-master/PRcodes'], path)

%% test matrices
% A = mmread('well1850.mtx');  % 良态矩阵
% [m,n]=size(A);
% b = ones(m,1);

%[A,b_true,x_true]=shaw(1024);  % 重度不�?�定矩阵
%[A,b_true,x_true]=gravity(1024);  % 重度不�?�定矩阵
[A,b_true,x_true]=heat(1024);  % 中度不�?�定矩阵
%[A,b_true,x_true]=deriv2(2048);  % 中度不�?�定矩阵

% add noise
[m,n]=size(A);
nel = 1e-4; % Noise level
b = PRnoise(b_true, 'gauss', nel);  % noisy data

%% parameters settting

L=get_l(n,1);
p=size(L,1);
[Q,R]=qr1([A;L]);
Qa=Q(1:m,:);
Ql=Q(m+1:m+p,:);

k0 = 50;
tol = 1e-6;
reorth = 0;  
[B, B_bar, U, U_hat, V_tilde, bbeta]=JointBid(A,L,b,k0+1,tol,reorth);
P=eye(k0);
for i=1:k0
    P(i+1,i+1)=-1*P(i,i);
end
V = Q'*V_tilde;
V_hat = V*P;
B_hat = B_bar*P;

%%  check for three matrix form recurrences
s1 = zeros(k0, 1);
s2 = zeros(k0, 1);
bnd1 = zeros(k0, 1);  % bound of ||S1||

% nr = zeros(k0,1);  % B_k^-1 的范数
% er = zeros(k0,1);  % (I-Q*Q')*v_k的值
% for i = 1:k0
%     ER = (eye(m+p)-Q*Q')*V_tilde(:,i);
%     er(i) = norm(ER);
%     d = svd(B(1:i+1,1:i));
%     nr(i) = 1/d(i);
% end

for k = 1:k0
   e_k=zeros(k,1);  e_k(k)=1;
   e_k1 = zeros(k+1, 1);  e_k1(k+1) = 1; 
   % matrix form recurrences for Lanczos bidiag of Qa
   S1 = Qa*V(:,1:k)-U(:,1:k+1)*B(1:k+1,1:k);
   S2 = Qa'*U(:,1:k+1)-V(:,1:k)*B(1:k+1,1:k)'- B(k+1,k+1)*V(:,k+1)*e_k1';
   s1(k) = norm(S1);
   s2(k) = norm(S2);
 
   d = svd(B(1:k,1:k));
   bnd1(k,1) = 100 * eps/d(k);
end

%% orthogonal level of v_k and v_til_k
lv1 = zeros(k0,1);  % orth level of v_til_k
lv2 = zeros(k0,1);  % orth level of v_k
for i = 1:k0
	E1 = eye(i)-V_tilde(:,1:i)'*V_tilde(:,1:i);
 	E2 = eye(i)-V(:,1:i)'*V(:,1:i);
	lv1(i) = norm(E1);
	lv2(i) = norm(E2);
end

%% JBD with full reorth
[B1, B1_bar, U1, U1_hat, V1_tilde, bbeta1]=JointBid(A,L,b,k0+1,tol,2);
V1 = Q'*V1_tilde;
V1_hat = V1*P;
B1_hat = B1_bar*P;
% orthogonal level of v1_k and v1_til_k
lev1 = zeros(k0,1);  % orth level of v_til_k
lev2 = zeros(k0,1);  % orth level of v_k
for i = 1:k0
	E1 = eye(i)-V1_tilde(:,1:i)'*V1_tilde(:,1:i);
 	E2 = eye(i)-V1(:,1:i)'*V1(:,1:i);
	lev1(i) = norm(E1);
	lev2(i) = norm(E2);
end

%% plot
figure; l = 1:k0;
semilogy(l,s1,'bo-');
hold on;
semilogy(l,s2,'md-');
hold on;
semilogy(l,bnd1,'r*-');
handle1 = legend('$\|F_{k}\|$','$\|G_{k+1}\|$','$100\|\underline{B}_{k}^{-1}\|\epsilon$', 'Location','best');
set(handle1,'interpreter','latex');
xlabel('Iteration');
ylabel('Error');

figure;
semilogy(l,lv1,'rd-');
hold on;
semilogy(l,lv2,'g*-');
hold on;
semilogy(l,lev1,'mv-');
hold on;
semilogy(l,lev2,'bo-');
handle2 = legend('$\tilde{\nu}_{k}$ no reorth','$\nu_{k}$ no reorth',...
'$\tilde{\nu}_{k}$ full reorth','$\nu_{k}$ full reorth','Location','best');
set(handle2,'interpreter','latex');
xlabel('Iteration');
ylabel('Orthogonal level');


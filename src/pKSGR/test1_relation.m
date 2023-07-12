% Verify the two coupled matrix-form recursive relatios that
% corresponding to PLanBid and pLSQR; 
% plot the loss of orthogonality and test reorthogonalization 
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 15, June, 2023.
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
%[A, b_true, x_true] = heat(1024);  
[A, b_true, x_true] = deriv2(1024);  

nel = 1e-2;  % Noise level
b = PRnoise(b_true, 'gauss', nel);  % noisy data

% construct L
[m,n]=size(A);
L = get_l(n,1);
M = L' * L;
%+ 1e-4*eye(n);  % prioconditioner
C = A' * A + M;

% k-step preconditioned Lanczos bidiagonalization 
k = 50;
reorth = 1;
%[bbeta, B, U, Z] = pLanBid(A, M, b, k, reorth);
[~, ~, ~, B, U, Z] = pLSQR(A, M, b, k, reorth);

% verify matrix-form recursive relations
er1 = zeros(k,1);
er2 = zeros(k,1);
for i = 1:k
    E1 = A * Z(:,1:i) - U(:,1:i+1) * B(1:i+1,1:i);
    er1(i) = norm(E1);
    % UU = M \ (A'*U(:,1:i));  E2 = UU - Z(:,1:i) * B(1:i,1:i)';
    %E2 = A' * U(:,1:i) - M * Z(:,1:i) * B(1:i,1:i)';
    E2 = A' * U(:,1:i) - C * Z(:,1:i) * B(1:i,1:i)';
    er2(i) = norm(E2);
end
% check 2-orthogonality of U and M-orthogonality of Z
orth1 = zeros(k,1);
orth2 = zeros(k,1);
for i = 1:k
    E1 = eye(i) - U(:,1:i)'*U(:,1:i);
    orth1(i) = norm(E1);
    %E2 = eye(i) - Z(:,1:i)'*M*Z(:,1:i);
    E2 = eye(i) - Z(:,1:i)'*C*Z(:,1:i);
    orth2(i) = norm(E2);
end


% xn = norm(x); 
% for i =1:k
%     er(i) = norm(x-X(:,i))/xn;
% end
% [~,k0] = min(er);


%--------- plot ------------------
lw = 2; l = 1:1:k;
figure; 
semilogy(l,er1,'-b','LineWidth',2);
hold on;
semilogy(l,er2,'-r','LineWidth',2);
handle=legend('relation 1','relation 2','Location','southeast');
set(handle,'Fontsize',14,'interpreter','latex');
xlabel('Iteration','Fontsize',15);
ylabel('Error','Fontsize',15); 
grid on;
%set(gca, 'GridAlpha', 0.2);
set(gca, 'MinorGridAlpha', 0.02);

figure; 
semilogy(l,orth1,'-m','LineWidth',2);
hold on;
semilogy(l,orth2,'-c','LineWidth',2);
handle=legend('$\|I_{k}-U_{k}^{T}U_{k}\|$','$\|I_{k}-Z_{k}^{T}CZ_{k}\|$','Location','southeast');
set(handle,'Fontsize',14,'interpreter','latex');
xlabel('Iteration','Fontsize',15);
ylabel('Orthogonality','Fontsize',15); 
grid on;
%set(gca, 'GridAlpha', 0.2);
set(gca, 'MinorGridAlpha', 0.02);
    




%%---------------------------------------
% figure; 
% plot(l, x,'b-', 'LineWidth', 2.0);
% hold on
% plot(l, b,'g-.', 'LineWidth', 2.0);
% hold on
% plot(l, X(:,k0),'r--', 'LineWidth', 2.0);
% legend('True', 'Blurred', 'Reconstructed');
% % ylim([-0.8 0.8]);

% figure;
% semilogy(1:k, er, 'bv-', 'LineWidth', 2.0);
% xlabel('Iteration');
% handle = ylabel('$RE(k)$','Interpreter','latex');
% set(handle,'interpreter','latex');




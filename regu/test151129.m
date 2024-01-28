clear;
n =256;
%$$$$$test matrix:shaw, heat,phillips , shaw, deriv2,  baart, foxgood
[A,b_t,x_true] = phillips(n);
[U,s,V]=svd(A);
b = b_t + 1e-3*norm(b_t)*randn(n,1);


%[x1,resn1,sn1] = lsqr_b(A,b,k,1);
% tic;
% [x2,resn2,err2,sn2,B,r,lam] = myhybrid(A,b,k,1,x_true);
% toc;
% 
% err1 = zeros(k,1);
% err2 = zeros(k,1);
% for i = 1:k
%     err1(i) = norm(x1(:,i) - x_true)/norm(x_true);
%     err2(i) = norm(x2(:,i) - x_true)/norm(x_true);
% end

%%%%%%%%%%%%%%%%%%%%% Part 1 %%%%%%%%%%%%%%%%%%%%%%%
% figure;
% sig = diag(s);
% semilogy(1:k-1,r(1:k-1),'-r*');
% hold on
% semilogy(1:k-1,sig(2:k),'-go');
% legend('\gamma_k','\sigma_{k+1}');
% xlabel('Iteration');
% xlim([0 1024]);
%%%%%%%%%%%%%%%%%%%%% Part 2 %%%%%%%%%%%%%%%%%%%%%%%
% b1 = b_t + 1e-2*norm(b_t)*randn(n,1);
% b2 = b_t + 1e-3*norm(b_t)*randn(n,1);
% b3 = b_t + 1e-4*norm(b_t)*randn(n,1);
% tic;
% [x1,resn1,err1,sn1,B1,r1,lam1] = myhybrid(A,b1,k,1,x_true);
% [x2,resn2,err2,sn2,B2,r2,lam2] = myhybrid(A,b2,k,1,x_true);
% [x3,resn3,err3,sn3,B3,r3,lam3] = myhybrid(A,b3,k,1,x_true);
% toc;
% k=25;
% figure;
% semilogy(1:k,err1(1:k),'-r*');
% hold on
% semilogy(1:k,err2(1:k),'-bo');
% hold on
% semilogy(1:k,err3(1:k),'-kv');
% legend('\epsilon=10^{-2}','\epsilon=10^{-3}','\epsilon=10^{-4}');
% ylabel('Relative error');
% xlabel('Iteration');
%%%%%%%%%%%%%%%%%%%%% Part 3 %%%%%%%%%%%%%%%%%%%%%%%
k =12;

[x1,resn1,sn1] = lsqr_b(A,b,k,1);
[x2,resn2,err2,sn2,B,r,lam] = myhybrid(A,b,k,1,x_true);
toc;

err1 = zeros(k,1);
% err2 = zeros(k,1);
for i = 1:k
    err1(i) = norm(x1(:,i) - x_true)/norm(x_true);
    %   err2(i) = norm(x2(:,i) - x_true)/norm(x_true);
end

%k=5;
figure;
semilogy(1:k,err1(1:k),'-r*');
hold on
semilogy(1:k,err2(1:k),'-bo');
legend('LSQR','LSQR+TSVD');
ylabel('Relative error');
xlabel('Iteration');

% figure;
% plot(x_true,'-g');
% hold on
% plot(x1(:,7),':k','linewidth',2);
% legend('x_{true}','x_{reg}');
% xlim([0 1024]);

%%%%%%%%%%%%%%%%%%%%% Part 4 %%%%%%%%%%%%%%%%%%%%%%%
% tic;
% [x1,resn1,sn1] = lsqr_b(A,b,k,1);
% [x2,resn2,err2,sn2,B,r,lam] = myhybrid(A,b,k,1,x_true);
% toc;
% 
% err1 = zeros(k,1);
% % err2 = zeros(k,1);
% for i = 1:k
%     err1(i) = norm(x1(:,i) - x_true)/norm(x_true);
%     %   err2(i) = norm(x2(:,i) - x_true)/norm(x_true);
% end
% 
% k=10;
% figure;
% semilogy(1:k,err1(1:k),'-r*');
% hold on
% semilogy(1:k,err2(1:k),'-bo');
% legend('LSQR','LSQR+TSVD');
% ylabel('Relative error');
% xlabel('Iteration');
% 
% figure;
% plot(x_true,'-g','linewidth',1.5);
% hold on
% plot(x1(:,3),':k','linewidth',1);
% hold on
% plot(x2(:,6),'-.r','linewidth',2.5);
% legend('x_{true}','x_{reg}(LSQR)','x_{reg}(LSQR+TSVD)');
% xlim([0 1024]);


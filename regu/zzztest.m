clear;
[A,b_bar,x] = deriv2(100,3);
e = 1e-9*randn (size (b_bar)); b = b_bar + e;
k =100;
[X,rho,eta] = cgls(A,b,k);
tk = 1:100;
[U,s,V] = csvd(A);
[x_k,rho1,eta1] = tsvd(U,s,V,b,tk');
err = zeros(k,1);
err1 = zeros(k,1);
for i=1:k
err(i) = norm(X(:,i) - x) ;
err1(i) =norm(x_k(:,i) - x);
end
plot(1:k,log(err/norm(x)),'-*r')
hold on
plot(1:k,log(rho(1:k)),'-ob')

% [U,s,V] = csvd (A);
% subplot (2,1,1); picard (U,s,b_bar);
% subplot (2,2,2); picard (U,s,b);
% k = 10;
% [X,rho,eta] = rrgmres(A,b,k);
% %plot(log(eta),log(rho));
% loglog(eta,rho,'-*');
%  randn(32,1);
% lambda = [1,3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,3e-5];
% X_tikh = tikhonov (U,s,V,b,lambda);
% F_tikh = fil_fac (s,lambda);
% iter = 30; reorth = 0;
% [X_lsqr,rho,eta,F_lsqr] = lsqr_b (A,b,iter,reorth,s);
% subplot (2,2,1); surf (X_tikh), axis ('ij'), title ('Tikhonov solutions')
% subplot (2,2,2); surf (log10 (F_tikh)), axis ('ij'), title ('Tikh ?lter factors, log scale')
% subplot (2,2,3); surf (X_lsqr (:,1:17)), axis ('ij'), title ('LSQR solutions')
% subplot (2,2,4); surf (log10 (F_lsqr)), axis ('ij'), title ('LSQR ?lter factors, log scale')
% 
% l_curve (U,s,b); axis ([1e-3,1,1,1e3])
% 
% 
% k=50;
% [X] = lsqr_b(A,b_bar,k);
% plot(X(:,k),'b-*')
% hold on
% plot(x,'r-o')

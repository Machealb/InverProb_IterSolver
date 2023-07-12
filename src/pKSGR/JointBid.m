% 对{A,L}的联合双对角化过程， 内迭代为LSQR, 停机准则 tol 为用户自选

function [B, Bbar, U, U_hat, V_tilde, bbeta]=JointBid(A,L,b,k,tol,reorth)

[m,n] = size(A); p=size(L,1);
beta=norm(b);
bbeta=beta;
u=b/beta;
U(:,1)=u;

utilde=[u; zeros(p,1)];
x = lsqr(@(z,tflag)afun(z,A,L,tflag),utilde,tol,3*n);
ss = A*x; tt = L*x;
v = [ss;tt];
alpha=norm(v);
v = v/alpha;
B(1,1) = alpha;
V_tilde(:,1) = v;

uhat = v(m+1:m+p);
alphahat = norm(uhat);
uhat = uhat/alphahat;
Bbar(1,1) = alphahat;
U_hat(:,1) = uhat;

if (reorth == 0)
    u = v(1:m) - alpha * u;
elseif (reorth == 1)
    u = v(1:m) - alpha * u;
    u = u - U * (U' * u);
elseif (reorth == 2)
    u = v(1:m) - alpha * u; 
    u = u - U * (U' * u);
    u = u - U * (U' * u);
end
beta = norm(u);
u = u/beta;
B(2,1) = beta;
U(:,2) = u;

for i = 2:k
    utilde = [U(:,i); zeros(p,1)];
    x = lsqr(@(z,tflag)afun(z,A,L,tflag), utilde,tol,3*n);
    ss = A*x; tt = L*x;
    Qu = [ss;tt];
    if (reorth == 0)
        v = Qu - B(i, i-1)*V_tilde(:,i-1);
    elseif(reorth == 1)
        v = Qu - B(i, i-1)*V_tilde(:,i-1);
        for j=1:i-1, v = v - (V_tilde(:,j)'*v)*V_tilde(:,j); end
%         v = v - V_tilde * V_tilde' * v;
    elseif (reorth == 2)
        v = Qu - B(i, i-1)*V_tilde(:,i-1);
        for j=1:i-1, v = v - (V_tilde(:,j)'*v)*V_tilde(:,j); end
        for j=1:i-1, v = v - (V_tilde(:,j)'*v)*V_tilde(:,j); end
%         v = v - V_tilde * V_tilde' * v;
%         v = v - V_tilde * V_tilde' * v;
    end
    alpha = norm(v);
    v = v/alpha;
    B(i,i) = alpha;
    V_tilde(:,i) = v;
    
    betahat=(alpha*B(i,i-1))/alphahat;
    if(mod(i,2)==0)
        Bbar(i-1,i) = -betahat;
    else
        Bbar(i-1,i) = betahat;
    end
    
    
    if(mod(i,2)==0)
        vv = -v(m+1:m+p);
    else
        vv = v(m+1:m+p);
    end
    
    if (reorth == 0)
        uhat = vv - betahat * U_hat(:,i-1);
    elseif (reorth == 1)
        uhat = vv - betahat * U_hat(:,i-1);
        for j=1:i-1, uhat = uhat - (U_hat(:,j)'*uhat)*U_hat(:,j); end
%         uhat = uhat - U_hat * U_hat' * uhat;
    elseif (reorth == 2)
        uhat = vv - betahat * U_hat(:,i-1);
        for j=1:i-1, uhat = uhat - (U_hat(:,j)'*uhat)*U_hat(:,j); end
        for j=1:i-1, uhat = uhat - (U_hat(:,j)'*uhat)*U_hat(:,j); end
%         uhat = uhat - U_hat * U_hat' * uhat;
%         uhat = uhat - U_hat * U_hat' * uhat;
    end
    alphahat = norm(uhat);
    if(mod(i,2)==0)
        Bbar(i,i) = -alphahat;
    else
        Bbar(i,i) = alphahat;
    end
    uhat = uhat/alphahat;
    U_hat(:,i) = uhat;
    
    if (reorth == 0)
        u = v(1:m) - alpha * u;
    elseif (reorth == 1)
        u = v(1:m) - alpha * u;
        for j=1:i, u = u - (U(:,j)'*u)*U(:,j); end
%         u = u - U * U' * u;
    elseif (reorth == 2)
        u = v(1:m) - alpha * u;
        for j=1:i, u = u - (U(:,j)'*u)*U(:,j); end
        for j=1:i, u = u - (U(:,j)'*u)*U(:,j); end
%         u = u - U * U' * u;
%         u = u - U * U' * u;
    end
    beta = norm(u);
    u = u/beta;
    B(i+1,i) = beta;
    U(:,i+1) = u;
end
end



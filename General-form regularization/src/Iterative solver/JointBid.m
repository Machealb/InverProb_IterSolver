function [B, Bbar, U, U_hat, V_tilde, bbeta] = JointBid(A, b, L, k, tol, reorth)
%  Joint bidiagonalization reduction for matrix pair {A, L}, where 
%  A and L have the same number of columns
%  It is used to develope iterative regularization methods of the general-form 
%  regularization problem:
%       min{||Ax-b||_{2}^2 + lambda ||Lx||^2}
% by 
% 1. subspace projection regularization
%       min{||Lx||} s.t. min||Ax-b||_2, where x \in span(Z_k) at the k-th step;
% 2. hybrid regularization method.
%
% Inputs:
%   A: either (a) a full or sparse mxn matrix;
%             (b) a matrix object that performs the matrix*vector operation
%   b: right-hand side vector
%   L: regularization matrix
%   k: the maximum number of iterations 
%   tol: stopping tolerance of LSQR for inner iteration
%       if tol=0, then solve it directly 
%   reorth: 
%       0: no reorthogonalization
%       1: full reorthogonaliation, MGS
%       2: double reorthogonaliation, MGS
%
% Outputs:
%   B: (k+1)xk lower bidiagonal matrix
%   Bbar: kxk upper bidiagonal matrix
%   U: mx(k+1) column 2-orthornormal matrix
%   U_hat: pxk column 2-orthornormal matrix
%   V_tilde: (m+p)xk matrix, use to costruct solution subspace Z_k
%   beta: 2-norm of b
%
% Reference: [1]. M. E. Kilmer, P. C. Hansen and M. I. Espanol, A projection–based approach to general
%  form Tikhonov regularization, SIAM J. Sci. Comput., 29 (2007), pp. 315–330.
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 06, July, 2023
%
% Check for acceptable number of input arguments
if nargin < 6
    error('Not Enough Inputs')
end

[m, n] = sizem(A); 
if n ~= size(L,2)
    error('A and L should have the same number of columns')
end
p = size(L,1);
C = [A; L];
beta = norm(b);
bbeta = beta;
u =  b/ beta;
U(:,1) = u;


utilde=[u; zeros(p,1)];
if tol == 0
    x = C \ utilde;
else
    x = lsqr(@(z,tflag)afun(z,A,L,tflag),utilde,tol,2*n);
end
ss = A*x; tt = L*x;
v = [ss; tt];
alpha = norm(v);
v = v / alpha;
B(1,1) = alpha;
V_tilde(:,1) = v;

uhat = v(m+1: m+p);
alphahat = norm(uhat);
uhat = uhat / alphahat;
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
    if tol == 0
        x = C \ utilde;
    else
        x = lsqr(@(z,tflag)afun(z,A,L,tflag),utilde,tol,2*n);
    end
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


%--------------------------------------------
function y = afun(z, A, B, transp_flag)
    if strcmp(transp_flag,'transp')   % y = (A(I_n-BB^T))' * z;
        m = size(A, 1);
        p = size(B, 1);
        s = A' * z(1:m);
        t = B' * z(m+1:m+p);
        y = s + t;
    elseif strcmp(transp_flag,'notransp') % y = (A(I_n-BB^T)) * z;
        s = A * z;
        t= B * z;
        y = [s; t];
        
    end
end

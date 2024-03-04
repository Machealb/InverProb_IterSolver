function [x,res,norm_F,lambda,gamma] = ProjectedNewton(A,b,lambda0,epsilon,maxiter,tol)
%  Projected Newton method for solving the noise constrained Tiknonov
%  problem:   min ||x||^2 s.t.  ||Ax - b||^2 = epsilon^2. 
%
%  Input:     - A: m x n matrix
%             - b: m x 1 vector (noisy right hand side)
%             - lambda0: initial lagrange multiplier
%             - epsilon: value for discrepancy principle, should be
%               approxiately equal to ||b_ex - b||, where b_ex is the
%               exact data
%             - maxiter: integer giving the maximum number of iterations
%             - tol: tolerance for convergence based on the norm of
%               F(x,lambda)
%
%  Output:    - x: matrix of maximum size n x (maxiter + 1) containing all
%               intermediate iterates
%             - res: vector containing the residual values ||Ax - b||
%             - norm_F: vector containing ||F(x,lambda)|| for the different
%               iterates
%             - lambda: vector containing the lagrange multiplier iterates
%             - gamma: vector containing the stepsizes used
%
%  NOTE: THIS IMPLEMENTATION IS NOT OPTIMIZED IN TERMS OF PERFORMANCE !!!

% Initialisations
res = zeros(1, maxiter + 1); 
lambda = zeros(1, maxiter + 1);
norm_F = zeros(1, maxiter);

U = []; V = []; y = []; 

lambda(1) = lambda0; 

normb = norm(b); 

Atb = A'*b; 

c = zeros(maxiter + 1, 1); c(1) = normb; 
Btc = zeros(maxiter + 1, 1); % vector for B'*c;

gamma = zeros(maxiter, 1);

res(1) = normb; 

theta = 0.9; 

n = length(Atb); 

x = zeros(n, maxiter + 1);

U(:,1) = b/normb; 

B = zeros(maxiter + 1,maxiter); 

r = A'*U(:,1); 
B(1,1) = norm(r); 
V(:,1) = r/B(1,1);
Btc(1) = B(1, 1)*normb;

norm_Atb = Btc(1); 

% We use this function to monitor convergence of ||F(x,lambda)||.
% This could obviously be computed much more efficiently by using the 
% expressions derived in the paper. (see remark 3.8)
F = @(x,lambda)[lambda*(A'*(A*x)) + x - lambda*Atb; (norm(A*x - b)^2 - epsilon^2)/2];

norm_F(1) = sqrt((lambda(1)*norm_Atb)^2 + (0.5*normb^2 - epsilon^2/2)^2);

c2 = 10^-4; % small parameter for the sufficient decrease linesearch

for k = 1:maxiter
    
    % Bidiagonalization 
    
    p = A*V(:, k) - B(k, k)*U(:, k);
    
    for j = 1:k
        % Reorthogonalisation of U vectors:
        p = p - (U(:, j)'*p)*U(:, j);
    end    

    B(k + 1, k) = norm(p);
    
    U(:, k + 1) = p/B(k + 1, k);
        
    r = A'*U(:, k + 1) - B(k + 1, k)*V(:, k);
   
    for j = 1:k
        % Reorthogonalisation of V vectors:
        r = r - (V(:, j)'*r)*V(:, j);
    end
    
    
    B(k + 1, k + 1) = norm(r);
    
    V(:, k + 1) = r/B(k + 1, k + 1);
        
    BtB = B(1:k + 1,1:k)'*B(1:k + 1,1:k);
    
    I = eye(k);
    
    y = [y;0];
    
    
	Fk = [lambda(k)*BtB*y + y - lambda(k)*Btc(1:k); ...
                 (norm(B(1:k + 1,1:k)*y - c(1:k + 1))^2 - epsilon^2)/2];    
    
    Jk = [lambda(k)*BtB + I, B(1:k + 1,1:k)'*(B(1:k + 1,1:k)*y - c(1:k+1)) ...
                     ; (B(1:k + 1,1:k)*y - c(1:k+1))'*B(1:k + 1,1:k), 0];  
    
    % Calculate the projected Newton direction 
    delta = - Jk\Fk;
    dy = delta(1:end - 1);
    dlambda = delta(end); 

    y_start = y; gamma(k) = 1; 
   
    lambda(k + 1) = lambda(k) + gamma(k)*dlambda; 
    
    % To ensure we have a positive lagrange multiplier
    if(lambda(k + 1) < 0)
        gamma(k) = - 0.9*lambda(k)/dlambda;
        lambda(k + 1) = lambda(k) + gamma(k)*dlambda; 
    end

    y = y_start + gamma(k)*dy; 
       
    fk_previous = 0.5*norm(Fk)^2;  
    
    % fk_next = 0.5*norm(F(V(:,1:k)*y, lambda(k + 1)))^2 ; 
    fk_next = 0.5*(norm(lambda(k + 1)*(B(1:k + 1,1:k + 1)'*(B(1:k + 1,1:k + 1)*[y;0] - c(1:k+1))) ...
        + [y;0])^2 + (0.5*norm(B(1:k + 1,1:k + 1)*[y;0] - c(1:k+1))^2 - epsilon^2/2)^2) ; 
    
    % Backtracking linesearch
    while(fk_next >  (1 - 2*c2*gamma(k))*fk_previous) 
        
        gamma(k) = gamma(k)*theta; 
        lambda(k + 1) = lambda(k) + gamma(k)*dlambda;
        y = y_start + gamma(k)*dy;
        
        fk_next = 0.5*(norm(lambda(k + 1)*(B(1:k + 1,1:k + 1)'*(B(1:k + 1,1:k + 1)*[y;0] - c(1:k+1))) ...
            + [y;0])^2 + (0.5*norm(B(1:k + 1,1:k + 1)*[y;0] - c(1:k+1))^2 - epsilon^2/2)^2) ;
    
        if(gamma(k) < 10^-14), warning('Stepsize too small. Projected Newton method stopped.')
            x = x(:, 1:k);res = res(1:k);gamma = gamma(1:k);
            lambda = lambda(1:k);norm_F = norm_F(1:k);return
        end
               
    end
    
    x(:,k + 1) = V(:,1:k)*y; % you do not really need to compute this 
    res(k + 1) = norm(A*x(:,k + 1) - b); % also not needed but interesting
    
    norm_F(k + 1) = norm(F(x(:,k + 1),lambda(k + 1))); % we could use the value fk_next to monitor convergence. 


    if(norm_F(k + 1) < tol)
        disp([' '])
        disp(['-----------------------------------'])
        disp(['---- CONVERGED in iteration ',num2str(k), ' ----'])
        disp(['-----------------------------------'])        
        break
    end
    
end

if(k == maxiter)
        disp([' '])
        disp(['-----------------------------------'])
        disp(['------- PNTM NOT CONVERGED --------'])
        disp(['-----------------------------------'])
end

% Concatenate output:
x = x(:, 1:k + 1);
res = res(1:k + 1);
gamma = gamma(1:k); 
lambda = lambda(1:k + 1);
norm_F = norm_F(1:k + 1);

end
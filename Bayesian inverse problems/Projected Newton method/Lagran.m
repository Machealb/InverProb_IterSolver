function [X, res, nx, nh, lamb] = Lagran(A, b, M, N, k, x0, lamb0, tol, method)
%  Lagrange method for solving the Bayesian linear inverse problems 
%  by noise constrained Tikhonov regularization:
%  problem:   min ||x||_{N^-1}^2 s.t.  ||Ax - b||_{M^-1}^2 <= \tau*m. 
%
%  Inputs:
%   A: either (a) a full or sparse mxn matrix;
%             (b) a matrix object that performs the matrix*vector operation
%   b: right-hand side vector
%   M: covaraince matrix of noise e, e~N(0,M), symmetric positive definite
%   N: scaled covaraince matrix of prior x, x~N(0,\lambda^{-1}N), symmetric positive definite
%   k: the maximum number of iterations 
%   x0: intialized value of x
%   lamb0: intialized value of lambda
%   tol: tolerance for convergence based on the norm of h(x,lambda)
%
%  Outputs: 
%   X: store the first k regularized solutions
%   res: strore residual norm of the first k regularized solution, ||Axk-b||_{M^{-1}}
%   nx: stores the norm of regularization term, ||x_k||_{N^{-1}}
%   nh: vector containing ||h(x_k,lambda_k)||_2 for different iterates
%   lamb: vector containing the lagrange multiplier iterates
%
%  Haibo Li, School of Mathematics and Statistics, The University of Melbourne
%  29, Feb, 2024.
    
% Check for acceptable number of input arguments
if nargin < 8
     error('Not Enough Inputs')
end
    
flag = 1;
tic
[m, n] = sizem1(A); 
if n ~= size(N,1) || m~= size(b,1)
    error('The dimensions are not consistent')
end

% Intialiazation
if size(M,2) == 1
    dm = 1.0 ./ M;
    M_inv = sparse(diag(dm));
end

X   = zeros(n, k); 
res = zeros(k,1);  
nx  = zeros(k,1); 
nh  = zeros(k,1);  
lamb = zeros(k,1);   

N_inv = inv(N);      % compute directly the inverse of N
tau_m  = 1.0001 * m;   % tau*m, the tolerance of DP
c      = 1e-4;       % small parameter for the sufficient decrease linesearch
eta    = 0.9;        % ratio for gradually decreasing step-length
ep     = 1e-3;
eta_ep = min(0.5,1-ep);
x = x0;
lambda = lamb0;
Am  = A'*M_inv;   %  this can not be computed if A is only a function handle!
Ama = Am*A;

for j = 1:k 
    fprintf('Running Lagrange: the %d-th step ----\n', j);
    rj = A*x-b;
    rj_Mn = rj'*(M_inv*rj);  % square of the norm of rj
    Fj = [lambda*Am*rj+N_inv*x; 0.5*rj_Mn-0.5*tau_m];
    Jj = [lambda*Ama+N_inv, Am*rj; (Am*rj)', 0];

    if strcmp(method, 'iterative')
        % r_tol = eta_ep * norm(Fj);  % in matlab the input tol in minres is the absolute res-norm
        r_tol = eta_ep;
        delta = minres(Jj, -Fj, r_tol, 2*n);
    elseif strcmp(method, 'direct')
        delta = - Jj \ Fj;
    end

    dx      = delta(1:end-1);
    dlamb   = delta(end);

    % To ensure a positive lagrange multiplier
    if dlamb >0
        gamma_init = 1.0;
    else
        gamma_init = -eta*lambda/dlamb;
    end

    % Backtracking linesearch
    gamma = gamma_init;
    x_new = x + gamma*dx;
    lambda_new = lambda + gamma*dlamb;
    rj_new = A*x_new-b;
    rj_Mn_new = rj_new'*(M_inv*rj_new);  % square of the norm of rj
    Fj_new = [lambda_new*Am*rj_new+N_inv*x_new; 0.5*rj_Mn_new-0.5*tau_m];

    while (0.5*sum(Fj_new.^2) >=  0.5*sum(Fj.^2)+c*gamma*delta'*Fj)
        gamma = gamma * eta; 
        x_new = x + gamma*dx;
        lambda_new = lambda + gamma*dlamb;
        rj_new = A*x_new-b;
        rj_Mn_new = rj_new'*(M_inv*rj_new);  % square of the norm of rj
        Fj_new = [lambda_new*Am*rj_new+N_inv*x_new; 0.5*rj_Mn_new-0.5*tau_m];
        if (gamma < 1e-16)
            warning('Stepsize too small. Projected Newton method stopped.')
            X    = X(:,1:j-1);
            res  = res(1:j-1);
            nx   = nx(1:j-1);
            nh   = nh(1:j-1);
            lamb = lamb(1:j-1);
            return
        end       
    end

    x = x_new;
    lambda  = lambda_new;
    X(:,j)   = x;
    lamb(j) = lambda;
    res_j   = A*x-b;
    res(j)  = sqrt(res_j'*M_inv*res_j);
    nx(j)   = sqrt(x'*N_inv*x);
    nh(j)   = 0.5 * sum(Fj_new.^2);

    if abs(res(j)^2-tau_m) <= 1e-8 && flag == 1
        toc 
        flag = 0;
    end

    if(nh(j) < tol)
        disp(['-----------------------------------']);
        disp(['---- Lagrange method CONVERGED in iteration ',num2str(j), ' ----']);
        disp(['-----------------------------------'])    ;    
        break
    end

end

if(j == k)
    disp(['-----------------------------------']);
    disp(['------- Lagrange method NOT CONVERGED --------']);
    disp(['-----------------------------------']);
end

% Concatenate output
X    = X(:,1:j);
res  = res(1:j);
nx   = nx(1:j);
nh   = nh(1:j);
lamb = lamb(1:j);

end


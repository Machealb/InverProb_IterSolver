function [X, res, nx, nh, lamb] = PNT(A, b, M, N, k, lamb0, tol)
%  Projected Newton method for solving the Bayesian linear inverse problems 
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
if nargin < 7
    error('Not Enough Inputs')
end

flag = 1;
tic
[m, n] = sizem1(A); 
if n ~= size(N,1) || m~= size(b,1)
    error('The dimensions are not consistent')
end

% Intialiazation
X   = zeros(n, k); 
res = zeros(k,1);  
nx  = zeros(k,1); 
nh  = zeros(k,1);  
lamb = zeros(k,1);

tau_m  = 1.0001 * m;   % tau*m, the tolerance of DP
y      = [];
lambda = lamb0;      % set lamb0=0.1 is good
c      = 1e-4;       % small parameter for the sufficient decrease linesearch
eta    = 0.9;        % ratio for gradually decreasing step-length
tol_gGKB = 0;        % tolerance of inner iterations of genGKB for M^{-1}v   
reorth   = 2;        % apply full reorthogonaliation to genGKB

U  = []; 
V  = []; 
Ub = [];
Vb = [];
B  = [];

% compute the inverse of M directly if M is diagonal
if size(M,2) == 1
    dm = 1.0 ./ M;
    M_inv = sparse(diag(dm));
end

% start iteration of gen-GKB, compute u_1, v_1
if tol_gGKB == 0
    if size(M,2) == 1
        sb = M_inv * b;   % if M is diagonal, it should be computed element-wise
    else
        sb = M \ b;  
    end
else
    sb = pcg(M, b, tol, 2*n);
end

bbeta = sqrt(sb'*b);
u = b / bbeta;    U(:,1) = u;
ub = sb / bbeta;  Ub(:,1) = ub;

rb = A' * ub;     r = N * rb;
alpha = sqrt(rb'*r);
vb = rb / alpha;  Vb(:,1) = vb;
v  = r / alpha;   V(:,1) = v;
B(1,1) = alpha;


% start iteration of PNT (includes genGKB)
for j = 1:k
    fprintf('Running PNT: the %d-th step ----\n', j);
    % compute u in M^{-1}-inner product, u_i should be M^{-1}-orthogonal
    s = mvp(A, v) - alpha * u;  % mvp(A, v) is A*v
    if reorth == 1  % full reorthogonalization of u
        for i = 1:j 
            s = s - U(:,i)*(Ub(:,i)'*s);
        end
    elseif reorth == 2  % double reorthogonalization of u
        for i = 1:j 
            s = s - U(:,i)*(Ub(:,i)'*s);
        end
        for i = 1:j 
            s = s - U(:,i)*(Ub(:,i)'*s);
        end
    else
        % pass
    end

    if tol_gGKB == 0
        if size(M,2) == 1
            sb = M_inv * s;  % if M is diagonal, it should be computed element-wise
        else
            sb = M \ s;  
        end
    else
        sb = pcg(M, s, tol, 2*n);
    end
    beta = sqrt(sb'*s);
    u  = s / beta;   U(:,j+1) = u;
    ub = sb / beta;  Ub(:,j+1) = ub;
    B(j+1,j) = beta;

    % compute v in N^{-1}-inner product, v_i should be N^{-1}-orthogonal
    rb = mvpt(A, ub) - beta*vb;
    if reorth == 1  % full reorthogonalization of u
        for i = 1:j 
            rb = rb - Vb(:,i)*(V(:,i)'*rb);
        end
    elseif reorth == 2  % double reorthogonalization of u
        for i = 1:j 
            rb = rb - Vb(:,i)*(V(:,i)'*rb);
        end
        for i = 1:j 
            rb = rb - Vb(:,i)*(V(:,i)'*rb);
        end
    else
        % pass
    end

    r     = N * rb;
    alpha = sqrt(rb'*r);
    vb    = rb / alpha;  Vb(:,j+1) = vb;
    v     = r / alpha;   V(:,j+1) = v;  
    B(j+1,j+1) = alpha;

    % form the j-th projected F and J
    y = [y; 0];
    I = eye(j);
    Bj  = B(1:j+1, 1:j);
    Bjt = Bj' * Bj;
    e1 = [1; zeros(j,1)];
    rj = Bj*y - bbeta*e1;
    Brj = Bj' * rj;
    
    Fj = [lambda*Brj+y; 0.5*sum(rj.^2)-0.5*tau_m];
    Jj = [lambda*Bjt+I, Brj; Brj', 0];

    % Calculate the projected Newton direction
    Jj = Jj + 1e-12*eye(j+1);
    delta   = - Jj \ Fj;
    dy      = delta(1:end-1);
    dlamb   = delta(end); 

    % To ensure a positive lagrange multiplier
    if dlamb >0
        gamma_init = 1.0;
    else
        temp = -eta*lambda/dlamb;
        gamma_init = min(1.0, temp);
    end

    % Backtracking linesearch
    gamma = gamma_init;
    y_new = [y+gamma*dy; 0];
    lambda_new = lambda + gamma*dlamb;
    Bjbar = B(1:j+1,1:j+1);
    rjbar = Bjbar*y_new-bbeta*e1;
    Fjbar = [lambda_new*Bjbar'*rjbar+y_new; 0.5*sum(rjbar.^2)-0.5*tau_m];

    while (0.5*sum(Fjbar.^2) >=  (0.5-c*gamma)*sum(Fj.^2)) 
        gamma = gamma * eta; 
        y_new = [y+gamma*dy; 0];
        lambda_new = lambda + gamma*dlamb;
        rjbar = Bjbar*y_new-bbeta*e1;
        Fjbar = [lambda_new*Bjbar'*rjbar+y_new; 0.5*sum(rjbar.^2)-0.5*tau_m];
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

    lambda = lambda_new;
    y      = y_new(1:j);
    x      = V(:,1:j)*y;

    lamb(j)= lambda;
    X(:,j) = x;
    res(j) = norm(Bj*y-bbeta*e1);
    nx(j)  = norm(y);
    nh(j)  = 0.5 * sum(Fjbar.^2);

    if abs(res(j)^2-tau_m) <= 1e-8 && flag == 1
        toc 
        flag = 0;
    end 
    
    if(nh(j) < tol)
        disp(['-----------------------------------']);
        disp(['---- Projected Newton CONVERGED in iteration ',num2str(j), ' ----']);
        disp(['-----------------------------------'])    ;    
        break
    end
    
end

if(j == k)
    disp(['-----------------------------------']);
    disp(['------- Projected Newton NOT CONVERGED --------']);
    disp(['-----------------------------------']);
end

% Concatenate output
X    = X(:,1:j);
res  = res(1:j);
nx   = nx(1:j);
nh   = nh(1:j);
lamb = lamb(1:j);

end

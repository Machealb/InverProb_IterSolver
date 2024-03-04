function [bbeta, B, U, V, Vb] = gen_GKB(A, b, M, N, k, tol, reorth)
    % Generalized Golub-Kahan bidiagonalization of A: (R^n, <.>_N^{-1}) ---> (R^m, <.>_M^{-1}) 
    % with starting vector b, where N and M are symmetric positive definite matrices.
    % Reorthogonalization (if available) is implemented on u_i and v_i.
    %
    % It is used to develope iterative regularization methods based on subspace projection onto span(V_k), to
    % solve the large-scale Bayesian inverse problem:
    %   b = Ax + e,   e~N(0,M)
    %   and the prior of x is x~N(0,\lambda^{-1}N),
    % which leads to the general-form Tikhonov regularization
    %   min{||Ax-b||_{M^{-1}}^2 + lambda*||x||_{N^{-1}}^2}.
    % 
    % Using subspace projection regularization method, we seek iterative solutions to 
    %     min_{x\in X_k} ||x||_{N^{-1}},  X_k = {x: min_{x\in S_k}} ||Ax-b||_{M^{-1}} },
    % where S_k=span(V_k) is called the solution subspace.
    %
    % Bases on subspace projection regularization, there is also a hybrid iterative regularization
    % method, which seeks the solution to
    %       min_{x\in S_k}  {||Ax-b||_{M^{-1}}^2 + lambda_k*||x||_{N^{-1}}^2} ,
    % where at each step, lambda_k should be determined by some criterion.
    %
    % Inputs:
    %   A: either (a) a full or sparse mxn matrix;
    %             (b) a matrix object that performs the matrix*vector operation
    %   b: right-hand side vector
    %   M: covaraince matrix of noise e, e~N(0,M), symmetric positive definite
    %   N: scaled covaraince matrix of prior x, x~N(0,\lambda^{-1}N), symmetric positive definite
    %   k: the maximum number of iterations 
    %   tol: stopping tolerance of pcg.m for solving M*sb = s
    %       if tol=0, then solve it directly 
    %   reorth: 
    %       0: no reorthogonalization
    %       1: full reorthogonaliation, MGS
    %       2: double reorthogonaliation, MGS
    %
    % Outputs:
    %   bbeta: M^{-1}-norm of b
    %   B: (k+1)x(k+1) lower bidiagonal matrix
    %   U: mx(k+1) column M^{-1}-orthornormal matrix
    %   V: nx(k+1) matrix, column N^{-1}-orthonormal, spans the solution subspace 
    %   Vb: nx(k+1) matrix, Vb = N^{-1}V
    %
    % Reference: [1]. Haibo Li, Subspace projection regularization for large-scale Bayesian
    %  inverse problems, preprint, 2023.
    % [2]. J. Chung and A. K. Saibaba. Generalized hybrid iterative methods for large-scale Bayesian
    %  inverse problems. SIAM J. Sci. Comput., 39(5):S24{S46, 2017.
    %
    % Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
    % 04, Sept, 2023.
    % 
    
    % Check for acceptable number of input arguments
    if nargin < 7
        error('Not Enough Inputs')
    end

    if isa(A, 'function_handle') && tol == 0
        error('Tol must not be 0 for a funtional handel A')
    end

    [m, n] = sizem1(A); 
    if size(M,2) == 1 && size(M,1) ~= m 
        error('The dimension of diagonal matrix M is not consistent with A')
    end
    
    % compute the inverse of M directly if M is diagonal
    if size(M,2) == 1
        dm = 1.0 ./ M;
        M_inv = sparse(diag(dm));
    end
        
    % declares the matrix size
    B = zeros(k+1, k+1);
    U = zeros(m, k+1);
    V = zeros(n, k+1);
    Ub = zeros(m, k+1);
    Vb = zeros(n, k+1);
    
    % start iteration, compute u_1, z_1
    if tol == 0
        if size(M,2) == 1
            sb = M_inv * b;  % if M is diagonal, it should be computed element-wise
        else
            sb = M \ b;  
        end
    else
        sb = pcg(M, b, tol, 2*n);
    end
    bbeta = sqrt(sb'*b);
    u = b / bbeta;  U(:,1) = u;
    ub = sb / bbeta;  Ub(:,1) = ub;

    rb = A' * ub;  r = N * rb;
    alpha = sqrt(rb'*r);
    vb = rb / alpha;  Vb(:,1) = vb;
    v  = r / alpha;   V(:,1) = v;
    B(1,1) = alpha;

    % start iteration
    for j = 1:k
		fprintf('Running gen-GKB process: the %d-th step --------------------\n', j);
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

        if tol == 0
            if size(M,2) == 1
                sb = M_inv * s;  % if M is diagonal, it should be computed element-wise
            else
                sb = M \ s;  
            end
        else
            sb = pcg(M, s, tol, 2*n);
        end
        beta = sqrt(sb'*s);
        u = s / beta;  U(:,j+1) = u;
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

        r = N * rb;
        alpha = sqrt(rb'*r);
        vb = rb / alpha;  Vb(:,j+1) = vb;
        v  = r / alpha;   V(:,j+1) = v;  
        B(j+1,j+1) = alpha;
    end
    
end
    
    

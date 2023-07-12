function [bbeta, B, U, Z] = pGKB(A, b, M, alpha0, k, tol, reorth)
    % Precondtioned Lanczos bidiagonalization using the symmetric semi-definite matrix M.
    % Let G = A'A + alpha*M, which is positive definite.
    % Suppose G = R^{T}R is the Cholesky factorization of G. 
    % pGKB implicitly do Golub-Kahan bidiagonalization (GKB) of {AR^{-1}, b} and then 
    % generate right Lanczos vectors by left multiplying R^{-1}. 
    % R is actually a right precoditioned of A for GKB.
    %
    % In practical implementation, pGKB do GKB under the 2-inner product for generating left Lanczos vectors,
    % and do GKB under the G-inner product for generating right Lanczos vectors. The process reduct {A, M} to 
    % a bidiagonal matrix $B_k$, and generating 2-orthornormal matrix U_k and G-orthornormal matrix Z_k.
    % At each outer iteration, a linear system Gx = A'*u needs to be solved, by a directly method (for small-scale 
    % matrix) or iterative method such as CG.
    %
    % It is used to develope iterative regularization methods based on subspace projection onto span(Z_k), to
    % solve the general-form regularization problem:
    %   min{||Ax-b||_{2}^2 + lambda x'*M*x}
    % by 
    % 1. subspace projection regularization
    %    min{x'*M*x} s.t. min||Ax-b||_2, where x \in span(Z_k) at the k-th step;
    % 2. hybrid regularization method.
    %
    % Reorthogonalization (if available) is implemented on u_i and z_i
    %
    % Inputs:
    %   A: either (a) a full or sparse mxn matrix;
    %             (b) a matrix object that performs the matrix*vector operation
    %   b: right-hand side vector
    %   M: regularization matrix, symmetric positive semi-definite
    %   alpha: parameter to control the condition number of G
    %   k: the maximum number of iterations 
    %   tol: stopping tolerance of pcg.m for solving Gx = A'u
    %       if tol=0, then solve it directly 
    %   reorth: 
    %       0: no reorthogonalization
    %       1: full reorthogonaliation, MGS
    %       2: double reorthogonaliation, MGS
    %
    % Outputs:
    %   beta: 2-norm of b
    %   B: (k+1)x(k+1) lower bidiagonal matrix
    %   U: mx(k+1) column 2-orthornormal matrix
    %   Z: nx(k+1) matrix, column G-orthonormal, spans the solution subspace 
    %
    % Reference: [1]. Haibo Li, A preconditioned Krylov subspace method for linear inverse 
    % problems with general-form Tikhonov regularization, preprint, 2023.
    % [2]. Simon R. Arridge,  Marta M. Betcke, and Lauri Harhanen. Iterated preconditioned LSQR 
    % method for inverse problems on unstructured grids[J]. Inverse Problems, 2014, 30(7): 075009.
    %
    % Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
    % 15, June, 2023.
    % 
    
    % Check for acceptable number of input arguments
    if nargin < 7
        error('Not Enough Inputs')
    end
    
    if size(M,1) ~= size(M,2)
        error('M needs to be square')
    end
    [m, n] = sizem(A); 
    if n ~= size(M,1) || m~= size(b,1)
        error('The dimensions are not consistent')
    end

    if isa(A, 'function_handle') && tol == 0
        error('Tol must not be 0 for a funtional handel A')
    end

    % declares the matrix size
    B = zeros(k+1, k+1);
    U = zeros(m, k+1);
    Z = zeros(n, k+1);
    
    % start iteration, compute u_1, z_1
    if tol == 0
        G = A'*A + alpha0*M;  % need not be formed explicitly for large-scale matrix or matrix operator
    end
    bbeta = norm(b);
    u = b / bbeta;  U(:,1) = u;
    uu = mvpt(A, u);  % A'*u
    if tol == 0
        rz = G \ uu;
    else
        rz = pcg(@(z)Gfun(z, A, M, alpha0), uu, tol, 2*n);
    end
    alpha = sqrt(uu' * rz);  B(1,1) = alpha;
    z = rz / alpha;  Z(:,1) = z;

    % start iteration
    for j = 1:k
		fprintf('Running pGKB process: the %d-th step ===================\n', j);
        % compute u in 2-inner product, u_i should be 2-orthogonal
        s = mvp(A, z) - alpha * u;  % mvp(A, z) is A*z
        if reorth == 1  % full reorthogonalization of u, in 2-inner product
            for i = 1:j 
                s = s - U(:,i)*(U(:,i)'*s);
            end
        elseif reorth == 2  % double reorthogonalization of u
            for i = 1:j 
                s = s - U(:,i)*(U(:,i)'*s);
            end
            for i = 1:j 
                s = s - U(:,i)*(U(:,i)'*s);
            end
        else
            % pass
        end
        beta = norm(s);  B(j+1,j) = beta;
        u = s / beta;    U(:,j+1) = u;

        % compute z in G-inner product, z_i should be G-orthogonal
        uu = mvpt(A, u);  % A'*u
        if tol == 0
            rz = G \ uu;
        else
            rz =pcg(@(z)Gfun(z, A, M, alpha0), uu, tol, 2*n);
        end
        rz = rz - beta * z;
        if reorth == 1  % full reorthogonalization of z, in G-inner product
            rr = Gfun(rz, A, M, alpha0);  % G*rz
            %rr = G * rz;
            for i = 1:j 
                rz = rz - Z(:,i)*(Z(:,i)'*rr);  % CGS to save reorthogonalization computation
            end
        elseif reorth == 2  % double reorthogonalization of z
            rr = Gfun(rz, A, M, alpha0);  % G*rz
            %rr = G * rz;
            for i = 1:j 
                rz = rz - Z(:,i)*(Z(:,i)'*rr);  % CGS, 
            end
            rr = Gfun(rz, A, M, alpha0);  % G*rz
            %rr = G * rz;
            for i = 1:j 
                rz = rz - Z(:,i)*(Z(:,i)'*rr);
            end
        else
            % pass
        end  
        alpha = sqrt(rz' * Gfun(rz, A, M, alpha0));  
        %alpha = sqrt(rz' * G * rz);
        B(j+1,j+1) = alpha;
        z = rz / alpha;  Z(:,j+1) = z;        
    end
    
end
    
    

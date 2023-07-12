function [X, res, eta, B, U, Z] = pLSQR(A, M, b, k, reorth, tol)
    % Priorcondtioned LSQR using the symmetric semidefinite matrix M.
    % Suppose M = L^{T}L is the Cholesky factorization of M, and the QR factorization
    % [A
    %  L] = QR,
    % and let C = R^{T}R = A^{T}A + M.
    %  pLSQR solves 
    %   min||R^{-1}Ax - R^{-1}b||_C,
    % using the LSQR in C-inner product space, which is the iterative solution of
    % min||x||_C  s.t. min||Ax-b||_2.
    % The Cholesky and L need not compute explicity.
    % pLSQR is mathematically equivalent to tLSQR for solving the precondioned
    %   ||b - AR^{-1}(Rx)||_2
    % by Lanczos bidiagonaliation of AR^{-1}, and thus it is equivalent to
    % the JBDQR algorithm applied to {A, L}.
    %
    % Inputs:
    %   A: either (a) a full or sparse mxn matrix;
    %             (b) a matrix object that performs the matrix*vector operation
    %   b: right-hand side vector
    %   A and b construct the ill-posed linear system: Ax + e = b, where e is the noise
    %   M: the priorconditioner matrix
    %   k: the maximum number of iterations  
    %   reorth: 
    %       0: no reorthogonalization
    %       1: full reorthogonaliation, MGS
    %       2: double reorthogonaliation, MGS
    %   tol: stopping tolerance for iteratively solving Cx=b
    %
    % Outputs: 
    %   X: store the first k regularized solution
    %   rel: strore residual norm of the first k regularized solution
    %   eta: strore solution norm of the first k regularized solution
    %
    % Reference: Simon R. Arridge,  Marta M. Betcke, and Lauri Harhanen. Iterated preconditioned LSQR method 
    % for inverse problems on unstructured grids[J]. Inverse Problems, 2014, 30(7): 075009.
    %
    % Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
    % 15, June, 2023.
    % 
    % Check for acceptable number of input arguments
    if nargin < 5
        error('Not Enough Inputs')
    end
    
    if size(M,1) ~= size(M,2)
        error('M needs to be square')
    end

    [m, n] = size(A); 
    if n ~= size(M,1) || m~= size(b,1)
        error('The dimensions are not consistent')
    end
    
    % declares the matrix size
    B = zeros(k+1, k+1);
    U = zeros(m, k+1);
    Z = zeros(n, k+1);
    X = zeros(n, k);
    res = zeros(k ,1);  % residual norm
    eta = zeros(k ,1);  % solution norm
    
    % start iteration, compute u_1, z_1
    fprintf('Start the pLSQR iteration ===========================================\n');
    C = A'*A + M;  % C should be symmetric definite
    bbeta = norm(b);
    u = b / bbeta;  U(:,1) = u;
    r = A' * u;
    if nargin == 6
        rz = pcg(C, r, tol, 2*n);  % can be obtained by solving C * rz = r iteratively, e.g., by CG
    else
        rz = C \ r;  % directly computing matrix inversion
    end
    alpha = sqrt(r' * rz);  B(1,1) = alpha;
    % alpha = sqrt(rz' * C * rz);  B(1,1) = alpha;
    z = rz / alpha;  Z(:,1) = z;

    % Prepare for mLSQR iteration.
    w = z;
    phi_bar = bbeta;
    rho_bar = alpha;
    x = zeros(n, 1);

    % The j-th step preconditioned Lanczos Bidiagonalization and pLSQR iteration
    for j = 1:k 
        % compute u in 2-inner product
        s = A * z - alpha * u;
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
        % compute z in M-inner product, z_i should be M-orthogonal
        % uu = C \ (A' * u);
        r = A' * u;
        if nargin == 6
            uu = pcg(C, r, tol, 2*n);  % obtained by solving M * uu = r iteratively, e.g., by CG
        else
            uu = C \ r;  % directly computing matrix inversion
        end
        rz = uu - beta * z;
        if reorth == 1  % full reorthogonalization of z, in M-inner product
            rr = C * rz;
            for i = 1:j 
                rz = rz - Z(:,i)*(Z(:,i)'*rr);  % CGS
            end
        elseif reorth == 2  % double reorthogonalization of z
            rr = C * rz;
            for i = 1:j 
                rz = rz - Z(:,i)*(Z(:,i)'*rr);
            end
            rr = C * rz;
            for i = 1:j 
                rz = rz - Z(:,i)*(Z(:,i)'*rr);
            end
        else
            % pass
        end  
        alpha = sqrt(rz' * C * rz);   B(j+1,j+1) = alpha;
        z = rz / alpha;  Z(:,j+1) = z;   

        % Construct and apply orthogonal transformation.
        rrho = sqrt(rho_bar^2 + beta^2); 
        c1 = rho_bar / rrho;
        s1 = beta / rrho; 
        theta = s1 * alpha; 
        rho_bar = -c1 * alpha;
        phi = c1 * phi_bar;
        phi_bar = s1 * phi_bar;  

        % Update the solution and w_i
        x = x + (phi/rrho) * w;  
        X(:, j) = x;
        w = z - (theta/rrho) * w;
        er = b - A*x;
        res(j) = norm(er);
        eta(j) = norm(x);
    end
    
end
    
    
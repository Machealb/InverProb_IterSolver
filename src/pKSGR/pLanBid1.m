function [bbeta, B, U, Z, P] = pLanBid1(A, M, b, k, reorth)
    % Precondtioned Lanczos bidiagonalization using the symmetric definite matrix M.
    % Suppose M = L^{T}L is the Cholesky factorization of M. 
    % pLanBid implicitly do bidiagonalization on {L^{-1}A, L^{-1}b} using the M-inner product,
    % with the aim to so3lve  
    %   min||L^{-1}Ax - L^{-1}b||_M,
    % using the LSQR in M-inner product space, which is the iterative solution of
    % min||x||_M  s.t. min||Ax-b||_2.
    % The Cholesky and L need not compute explicity.
    % pLanBid is mathematically equivalent to (preconditioned) Lanczos bidiagonaliation of
    %   {AL^{-1}, b}   
    % in the standart Euclidian inner product for constructing solution space 
    % (after transforming solution space by multiplying L^{-1}).
    %
    % Inputs:
    %   A: either (a) a full or sparse mxn matrix;
    %             (b) a matrix object that performs the matrix*vector operation
    %   b: initial vector
    %   M: the preconditioner matrixm, symmetric definite
    %   k: the maximum number of iterations  
    %   reorth: 
    %       0: no reorthogonalization
    %       1: full reorthogonaliation, MGS
    %
    % Outputs:
    %   beta: 2-norm of b
    %   B: (k+1)xk lower bidiagonal matrix
    %   U: mx(k+1) column orthornormal matrix
    %   Z: nxk matrix, Z = L^{-1}V, which spans the solution subspace 
    %   P: nxk matrix, P = MZ, assisting with reorthogonalization of z (avoid to many M*v computations)
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
    B = zeros(k+1, k);
    U = zeros(m, k+1);
    Z = zeros(n, k+1);
    P = zeros(n, k+1);
    
    % start iteration, compute u_1, z_1
    bbeta = norm(b);
    u = b / bbeta;  U(:,1) = u;
    r = A' * u;
    rz = M \ r;  % can be obtained by solving M * rz = r iteratively, e.g., by CG
    alpha = sqrt(r' * rz);  B(1,1) = alpha;
    p = rz / alpha;  P(:,1) = p;
    z = rz / alpha;  Z(:,1) = z;

    % start iteration
    for j = 1:k
        % compute u in 2-inner product
        s = A * z - alpha * u;
        if reorth == 1  % full reorthogonalization of u, in 2-inner product
            for i = 1:j 
                s = s - U(:,i)*(U(:,i)'*s);
            end
        end
        beta = norm(s);  B(j+1,j) = beta;
        u = s / beta;    U(:,j+1) = u;
        % compute z in M-inner product, z_i should be M-orthogonal
        r = A' * u - beta * p;
        if reorth == 1  % full reorthogonalization of p over z_i
            for i = 1:j 
                r = r - Z(:,i)*(Z(:,i)*r);
            end
        end
        rz = M \ r;  
        alpha = sqrt(r' * rz);  B(j+1,j+1) = alpha;
        p = r / alpha;  P(:,j+1) = p;
        z = rz / alpha;  Z(:,j+1) = z;        
    end
    
end
    
    
function [bbeta, B, U, V, Z, X] = tLSQR(A, L, b, k, reorth)
    % Transformed LSQR based on the bidiagonal reduction of AL^{-1}, where L^"{-1} is
    % implicitly computed by solving linear equations, solve the preconditioned
    %    ||b - AL^{-1}(Lx)||_2
    % by Lanczos bidiagonaliation of AL^{-1}.
    %
    % Inputs:
    %   A: either (a) a full or sparse mxn matrix;
    %             (b) a matrix object that performs the matrix*vector operation
    %   b: right-hand side vector
    %   A and b construct the ill-posed linear system: Ax + e = b, where e is the noise
    %   L: the preconditioner matrix
    %   k: the maximum number of iterations  
    %   reorth: 
    %       0: no reorthogonalization
    %       1: full reorthogonaliation, MGS
    %
    % Outputs:
    %   beta: 2-norm of b
    %   B: (k+1)xk lower bidiagonal matrix
    %   U: mx(k+1) column orthornormal matrix
    %   V: nxk column orthornormal matrix
    %   Z: nxk matrix, Z = L^{-1}V, which spans the solution subspace 
    %   X: store the first k regularized solution
    %
    % Reference: Simon R. Arridge,  Marta M. Betcke, and Lauri Harhanen. Iterated preconditioned LSQR method 
    % for inverse problems on unstructured grids[J]. Inverse Problems, 2014, 30(7): 075009.
    %
    % Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
    % 23, May, 2023.
    %
    
    % Check for acceptable number of input arguments
    if nargin < 5
        error('Not Enough Inputs')
    end
    
    if size(L,1) ~= size(L,2)
        error('L needs to be square')
    end

    [m, n] = size(A); 
    if n ~= size(L,1) | m~= size(b,1)
        error('The dimensions are not consistent')
    end
    
    % declares the matrix size
    B = zeros(k+1, k);
    U = zeros(m, k+1);
    V = zeros(n, k);
    Z = zeros(n, k);
    X = zeros(n, k);
    
    % start iteration
    Lt = L';
    bbeta = norm(b);  beta = bbeta;
    u = b / bbeta;  U(:,1) = u;
    for j = 1:k
        r = A' * u;
        r = Lt \ r;  % can be obtained by solving Lt * r = A'*u iteratively
        % compute v, z 
        if j > 1
            r = r - beta * v;
            if reorth == 1  % full reorthogonalization of v
                for i = 1 : j-1
                    r = r - V(:,i)*V(:,i)'*r;  % MGS
                end
            end
        end
        alpha = norm(r);  B(j,j) = alpha;
        v = r / alpha;    V(:,j) = v;
        z = L \ v;        Z(:,j) = z;   % can be obtained by solving L * z = v iteratively

        % compute u
        p = A * z - alpha * u;
        if reorth == 1  % full reorthogonalization of u
            for i = 1:j
                p = p - U(:,i)*U(:,i)'*p;  % MGS
            end
        end
        beta = norm(p);  B(j+1,j) = beta;
        u = p / beta;    U(:,j+1) = u;

        % computed x_j
        e1 = zeros(j+1,1);  e1(1,1) = 1;
        y_j = B(1:j+1,1:j) \ (bbeta*e1);
        x_j = Z(:,1:j) * y_j;
        X(:,j) = x_j;
    end
    
end
    
    
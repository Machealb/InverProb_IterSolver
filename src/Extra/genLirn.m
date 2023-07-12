function L = genLirn(x, reg, tau, varargin)
% This function geneates a matrix L based on the itratively reweighted norm at a 
% a single step, where the current solution is x
%
% Inputs:
%   x: a column vector
%   tau: threshold to avoid zero divisors
%   reg: type of regularization terms:
%       1. 'lp': l_p regularization
%       2. '1dTV_0': 1-D TV regularization, without adding boundary condition
%       3. '1dTV_1': 1-D TV regularization, adding boundary condition
%       4. '2dTV': 2-D TV regularization
%   varargin:
%       reg='lp', p = varargin(1), L_p norm,
%       reg='1dTV', not need a varargin
%       reg='2dTV', M = varargin(1), N = varargin(N). Denote an MxN image
% Outputs:
%       L: a sparse matrix
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 18, May, 2023.
%

% Check for acceptable number of input arguments
if nargin < 3
    error('Not Enough Inputs')
end

if ~ischar(reg)
    error('reg  should a be string');
end

x = x(:);
n = size(x, 1);

if strcmp(reg, 'lp')
    p = varargin{1};
elseif strcmp(reg, '1dTV_0') || strcmp(reg, '1dTV_1')
    if nargin > 3
        error('Too many Inputs for 1D TV')
    end
elseif strcmp(reg, '2dTV')
    M = varargin{1};  
    N = varargin{2};
    if M*N ~= n
        error('The image and vectorized input does not much')
    end
else
    error('Wrong regularization term')
end

% iteratively reweighed norm
if strcmp(reg, 'lp')
    q = (p-2)/2;
    x = abs(x);
    W = diag((f_tau(x)).^q);
    L = sparse(W);
elseif strcmp(reg, '1dTV_0')  % without adding boundary condition for discrete 1D difference
    D = zeros(n-1, n);
    for i = 1:n-1
        D(i,i) = 1;
        D(i,i+1) = -1;
    end
    % D = [D; zeros(1,n)];  % optional
    y = D*x;
    y = abs(y);
    W = diag((f_tau(y)).^(-1/2));
    L = sparse(W*D);
elseif strcmp(reg, '1dTV_1')  % add boundary condition for discrete 1D difference
    D = zeros(n, n);
    for i = 1:n-1
        D(i,i) = 1;
        D(i,i+1) = -1;
    end
    D(n,n) = 1;
    % D = [D; zeros(1,n)];  % optional
    y = D*x;
    y = abs(y);
    W = diag((f_tau(y)).^(-1/2));
    L = sparse(W*D);
elseif strcmp(reg, '2dTV')
    D1 = zeros(M-1, M);  D2 = zeros(N-1, N);
    for i = 1:M-1
        D1(i,i) = 1;
        D1(i,i+1) = -1;
    end
    for i = 1:N-1
        D2(i,i) = 1;
        D2(i,i+1) = -1;
    end
    Dh = kron([D2; zeros(1,N)], eye(M));
    Dv = kron(eye(N), [D1; zeros(1,M)]);
    s = f_tau((Dh*x).^2 + (Dv*x).^2);
    w_til = s.^(-1/4);
    W_til = diag(w_til);
    W = [W_til, zeros(n,n); zeros(n,n), W_til];
    L = [W*Dh; W*Dv];
    L = sparse(L);
else
    error('Wrong regularization term')
end

%--------------------------------------------------
function y = f_tau(x)
x = x(:);
k = size(x,1);
y = zeros(k,1);
for i = 1:k
    if x(i) < 0
     error('x should be positive')
    end
    if x(i) < tau
        y(i) = tau;
    else
        y(i) = x(i);
    end
end
end

end
    
    
    
    
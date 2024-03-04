function [U, sm, X, V, W] = gsvd1(A, L)
% Compute the compact generalized SVD of a matrix pair {A,L}.
% Computes the generalized SVD of the matrix pair (A,L). The dimensions of
% A and L must be such that [A;L] does not have fewer rows than columns.
%
% If m >= n >= p then the GSVD has the form:
%    [ A ] = [ U  0 ]*[ diag(sigma)      0    ]
%                     [      0       eye(n-p) ] * inv(X)
%    [ L ] = [ 0  V ]*[  diag(mu)        0    ]
% where
%    U  is  m-by-n ,    sigma  is  p-by-1
%    V  is  p-by-p ,    mu     is  p-by-1
%    X  is  n-by-n .
%    !!! sigma ordered increasing, mu ordered decreasing !!!
% Otherwise the GSVD has a more complicated form.
%
% 1. sm = gsvd1(A,L)
% 2. [U,sm,X,V] = gsvd1(A,L) ,  sm = [sigma,mu]
% 3. [U,sm,X,V,W] = gsvd1(A,L) ,  sm = [sigma,mu]%
% A possible fifth output argument returns W = inv(X).
%
% Reference: C. F. Van Loan, "Computing the CS and the generalized 
% singular value decomposition", Numer. Math. 46 (1985), 479-491. 
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 08, July, 2023.
 
% Initialization
[m, n] = size(A); 
[p, n1] = size(L);
if n1 ~= n
  error('Columns of A and L must be the same')
end
if m+p < n
  error('Dimensions must satisfy m+p >= n')
end

% Call Matlab's GSVD routine
[U, V, W, C, S] = gsvd(full(A), full(L), 0);

if m >= n
  % The overdetermined or square case
  sm = [diag(C(1:p,1:p)), diag(S(1:p,1:p))]; 
  if nargout < 2
    U = sm; 
  else 
    % Full decomposition
    X = inv(W'); 
  end
else
  % The underdetermined case
  sm = [diag(C(1:m+p-n,n-m+1:p)),diag(S(n-m+1:p,n-m+1:p))]; 
  if nargout < 2 
    U = sm; 
  else 
    % Full decomposition
    X = inv(W');
    X = X(:,n-m+1:n); 
  end
end

if nargout==5
  W = W'; 
  
end
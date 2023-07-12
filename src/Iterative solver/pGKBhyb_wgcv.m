function [X, res, Lam, GCV, iterstop] = pGKBhyb_wgcv(A, b, M, alpha, k, tol, adaptW)
% pGKBhyb_su uses the right subspace Z_k generated by pGKB as solution subspace,
% and solves the general-form regularization problem
%   min{||Ax-b||_{2}^2 + lambda x'*M*x}
% by hybrid regularization method, and using the WGCV method
% to update the regularization parameter at each iteration.
%
% Inputs:
%   A: either (a). a full or sparse mxn matrix;
%             (b). a matrix object that performs the matrix*vector operation
%   b: right-hand side vector
%   M: regularization matrix, symmetric positive semi-definite
%   alpha: parameter to control the condition number of G
%   k: the maximum number of iterations 
%   tol: stopping tolerance of pcg.m for pGKB
%   adaptW:
%     0: not adapt weight parameters w, use GCV method
%     1: adapt weight parameters w, use WGCV method
%
% Outputs: 
%   X: store the first k regularized solutions
%   res: strore residual norm of the first k regularized solution
%   Lam: stores the first k regularization parameters
%   GCV - store values of the GCV function, i.e. GCV with w=1.
%   iterstop: the early seopping iteration estimated by DP
% 
% Reference: [1]. Haibo Li, A preconditioned Krylov subspace method for linear inverse 
%   problems with general-form Tikhonov regularization, preprint, 2023.
% [2]. J. Chung, J. G. Nagy and D. P. O' Leary, A weighted-GCV method for Lanczos-hybrid 
%  regularization, Electr. Trans. Numer. Anal., 28 (2008), 149-167.
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 05, July, 2023.
%
% Initialization
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

if min(min(abs(M-M'))) > 1e-14
  error('The matrix must be symmetric')
end
M = (M+M')/2;  % enhance symmtry

X = zeros(n, k); 
res = zeros(k, 1); 
Lam = zeros(k, 1);  
GCV = zeros(k, 1);    % store values of WGCV functions
omega = zeros(k, 1);  % store auxiliary weight parameters
iterstop = 0;    % initialize the early stopping iteration
terminate = 1;   % indicate wether we still need to estimate iterstopb
% adaptW = 1;  
degflat = 1e-6;  % tolerance for estimate WGCV stopping point
warning = 1;     % avoid possible semi-convergence of GCV
step1 = 4;
step2 = 4;

fprintf('Start the pGKB iteration ===================================\n');
[bbeta, B, ~, Z] = pGKB(A, b, M, alpha, k, tol, 1);

h = waitbar(0, 'Beginning iterations: please wait ...');
fprintf('Start the WGCV iteration ===================================\n');

for i = 1:k 
  fprintf('Running WGCV regularizing iteration: the %d-th step ===================\n', i);
  Zk = Z(:,1:i);
  Bk = B(1:i+1, 1:i); 
  Mk = Zk' * M * Zk;
  Ck = srd(Mk);  % squre root decomposition
  vector = [bbeta; zeros(i,1)];

  [Ub, ss, Xb] = gsvd1(Bk, Ck);

  if adaptW  % Use the adaptive weighted GCV method
    omega(i)= min(1.0, findomega(Ub, vector, ss));
    om = mean(omega(1:i));
    lambda = wgcv(Ub, ss, vector, om);
  else
    lambda = wgcv(Ub, ss, vector, 1.0);  % w=1.0, i.e. the standard GCV method
  end
  % Solve the projected problem with Tikhonov regularization
  f = tikhonov(Ub, ss, Xb, vector, lambda);
  x = Zk * f;
  X(:,i) = x;
  Lam(i) = lambda^2;  % regularization parameter of the projected problem
  res(i) = norm(Bk*f-vector);
  GCV(i) = GCVstopfun(lambda, Ub, ss, vector);

  % use the GCV value to find the stopping point
  if terminate && warning && i > step1
    l = i - step1;
    if GCV(l) < min(GCV(l+1:i))  
      iterstop = l;  % stop at the minimum GCV to avoid possible semi-convergence      terminate = 0;
    end
  end
  % If GCV curve is flat, we stop and avoid bumps in the GCV curve by using a window of step2+1 iterations 
  if terminate && i > (step2+1)
    if abs((GCV(i)-GCV(i-1)))/GCV(2) <= degflat 
    %if abs((GCV(i)-GCV(i-1)))/GCV(i-1) <= degflat
      flag = 0;
      for j = 0:step2
        if abs((GCV(i-j)-GCV(i-j-1)))/GCV(2) > degflat 
          flag = flag + 1;
        end
      end
      if flag == 0
        iterstop = i;
        terminate = 0;
      end
    end
  end
  waitbar(i/k, h)
end
close(h);

% check if the stopping iteration is satisfied
if terminate == 1
  iterstop = k;
  fprintf('The WGCV method has not been stabalized. \n');
end

end


%-----------------------SUBFUNCTION---------------------------------------
function omega = findomega(U, b, s)
%function omega = findomega(bhat, delta0, s)
%  This function computes a value for the omega parameter.
%  The method assumes the 'optimal' regularization parameter to be the
%  smallest (generalized) singular value.  Then we take the derivative of the GCV
%  function with respect to alpha, evaluate it at alpha_opt, set the
%  derivative equal to zero and then solve for omega.
%  First assume the 'optimal' regularization parameter to be the 
%  smallest singular value.
%
%  Input:  bhat - vector U'*b, where U = left (generalized) singular vectors
%          s - vector containing the (generalized) singular values
%
%  Output: omega - computed value for the omega parameter
%
[m, n] = size(U);
bhat = U' * b;
if m > n
  delta0 = norm(b)^2 - norm(bhat)^2;  % ||(I-U*U')b||^2
else
  delta0 = 0;
end

[p, ps] = size(s);  % for the SVD case: p=n, ps=1
if ps==2
  s = s(p:-1:1, 1) ./ s(p:-1:1, 2);  % generalized singular values in decreasing order
  bhat = bhat(p:-1:1);
end
alpha1 = s(p);  % suppose it is the optimal regularization parameter

% compute needed elements for derivative of the WGCV function
s2 = abs(s) .^ 2;
alpha2 = alpha1^2;

tt = 1.0 ./ (s2 + alpha2);

t1 = sum(s2 .* tt) + n - p;
t2 = abs(bhat(1:p).*alpha1.*s) .^2;
t3 = sum(t2 .* abs((tt.^3)));

t4 = sum((s.*tt) .^2);
t5 = sum((abs(alpha2.*bhat(1:p).*tt)).^2);

v1 = abs(bhat(1:p).*s).^2;
v2 = sum(v1.* abs((tt.^3)));

% compute omega by letting derivative of WGCV to be zero
omega = (m*alpha2*v2) / (t1*t3 + t4*(t5 + delta0));
end


% ---------------SUBFUNCTION ---------------------------------------
function G = GCVstopfun(alpha1, U, s, b)
%  This function evaluates the GCV function G(1, alpha1), that will be used
%  to determine the early stopping iteration
%
[m,n] = size(U); 
[p,ps] = size(s);
beta = U'*b; 
beta2 = norm(b)^2 - norm(beta)^2;  % ||(I-U*U')b||^2
if ps==2
  s = s(p:-1:1, 1) ./ s(p:-1:1, 2); 
  beta = beta(p:-1:1);
end

s2 = s.^2;

% Intrinsic residual
delta0 = 0;
if m > n && beta2 > 0 
  delta0 = beta2; 
end

G = wgcvfun(alpha1, s2, beta(1:p), delta0,m, n, 1);
end
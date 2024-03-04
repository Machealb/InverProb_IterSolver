function [X, res, Lam, GCV, iterstop] = genGKBhyb_wgcv(A, b, M, N, k, tol, adaptW)
% genGKBhyb_wgcv uses the solution subspace span(Z_k) generated by gen-GKB to project the 
% original Tikhonov regularization problem 
%   min{||Ax-b||_{M^{-1}}^2 + lambda*||x||_{N^{-1}}^2}
% to S_k = span(Z_k), that becomes
%       min_{x\in S_k}  {||Ax-b||_{M^{-1}}^2 + lambda_k*||x||_{N^{-1}}^2} .
% This projected problem leads to
%       min{||B_k*y-beta_1*e_1||^2+\lambda_k*||y||^2},
% and should be solved by a hybrid approach, where at where at each step, 
% lambda_k should be determined by WGCV.
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
% Reference: [1]. Haibo Li, Subspace projection regularization for large-scale Bayesian
%  inverse problems, preprint, 2023.
% [2]. J. Chung and A. K. Saibaba. Generalized hybrid iterative methods for large-scale Bayesian
%  inverse problems. SIAM J. Sci. Comput., 39(5):S24{S46, 2017.
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 04, Sept, 2023.
%
% Initialization
if nargin < 7
  error('Not Enough Inputs')
end

[m, n] = sizem1(A); 
if n ~= size(N,1) || m~= size(b,1)
  error('The dimensions are not consistent')
end

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

fprintf('Start the genGKBhyb iteration ===================================\n');
[bbeta, B, ~, Z, ~] = gen_GKB(A, b, M, N, k, tol, 1);

h = waitbar(0, 'Beginning iterations: please wait ...');
fprintf('Start the WGCV iteration ++++++++++++++++++++++++++++++++++++\n');
figure;

for i = 1:k 
  fprintf('Running WGCV regularizing iteration: the %d-th step ++++++++++++++++++\n', i);
  Zk = Z(:,1:i);
  Bk = B(1:i+1, 1:i); 
  Ck = eye(i);
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
      iterstop = l;  % stop at the minimum GCV to avoid possible semi-convergence   
      fprintf('********* WGCV jump! ********** \n');   
      terminate = 0;
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
        %iterstop = i;
        [~, iterstop1] = min(GCV(i-step2:i));
        iterstop = iterstop1 + i -step2 -1;
        if iterstop == i
            fprintf('********* WGCV decreases flat ********* \n');
        else
            fprintf('********* WGCV jump! ********** \n');
        end
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
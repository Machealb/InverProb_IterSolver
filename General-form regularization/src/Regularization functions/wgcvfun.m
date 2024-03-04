function G = wgcvfun(lambda, s2, beta, delta0, m, n, omega)
% Compute the value of WGCV function
%         || A*x_lam - b ||^2
%    G = ---------------------
%        (trace(I - wA*A_I)^2
%  when the explict expression of it is known.
% 
% Inputs:
%   lambda: regularization parameter
%   s2: square of (generalized) singular values of {A,L}
%   beta: U'*b, U are left (generalized) singular vectors of {A,L}
%   delta0: Intrinsic residual, ||(I-U*U')b||^2
%   m, n: [m,n]=size(A)
%   omega: weight parameter
%
% Output:
%   G: value of WGCV function
% 
% Reference: J. Chung, J. G. Nagy and D. P. O' Leary, A weighted-GCV method for 
% Lanczos-hybrid regularization, Electr. Trans. Numer. Anal., 28 (2008), 149-167.
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 06, July, 2023.
%

% Note that f = 1 - filter_factors
f1 = lambda^2./(s2 + lambda^2);  
res2 = norm(f1.*beta)^2 + delta0;

a = m - n + sum(f1);
tr = (1-omega)*m + omega*a;
tr2 = tr^2;

G = res2 / tr2;

end

function [x, e, lambda] = Tikopt(A, L, b, x_true)
% This function computes a Tikhonov regularized solution of
%       min{||Ax-b||^2 + lambda^2*||Lx||^2}
% for mxn matrix A and pxn matrix L, where lambda is the
% optimal parameter (x_true is known).
%  
% Input: 
%   A, L:  min{||Ax-b||^2 + lambda^2||Lx||^2
%   b - noisy data
%   x_true - true solution
%
%  Output: 
%    x - opitimal Tikhonov solution
%    e - solution error
%   lambda - optimal regularization parameter
%       
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 04, Sept, 2023

[U, sm, X, ~, ~] = cgsvd(A,L);
b_hat = U'*b;
lambda = fminbnd('TikErrors', 1e-6, 10, optimset('Display','off'), sm, X, b_hat, x_true);
e = TikErrors(lambda, sm, X, b_hat, x_true);

c = sm(:,1);
s = sm(:,2);
[n, ~] = size(b_hat);
[p, ~] = size(sm);

%%
x_lam = zeros(n,1);  % GSVD filtered solution
x_0 = zeros(n,1);  % GSVD none-filtered part
for i = p+1:n
    x_0 = x_0 + b_hat(i) * X(:,i);
end
for i=1:p
	x_lam = x_lam + c(i)/(c(i)^2+lambda^2 * s(i)^2) * b_hat(i) * X(:,i);
end
x = x_lam + x_0;



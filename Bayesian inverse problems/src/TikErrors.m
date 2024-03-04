function E = TikErrors(lambda, sm, X, b_hat, x_true)
% This function computes the error for general-form Tikhonov Regularization
%    min{||Ax-b||^2 + lambda^2*||Lx||^2}
% for mxn matrix A and pxn matrix L. Assume m >= n >= p.
%
% Input:
%   A, L:  min{||Ax-b||^2 + lambda^2||Lx||^2}
%  lambda - regularization parameters, can be a vector constituted by many values
%  b_hat - U'*b, where b is the noisy data, U - generalized left vector of A 
%  X - generalized right vectors
%  x_true - true solution
%
% Output:
%  E - 2-norm error at the given lambda, ||x_lambda-x_true||_2
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 04, Sept, 2023.

c = sm(:,1);
s = sm(:,2);
[n, ~] = size(b_hat);
[p, ~] = size(sm);

E = zeros(length(lambda), 1);
x_0 = zeros(n,1);  % GSVD none-filtered part
for i = p+1:n
    x_0 = x_0 + b_hat(i) * X(:,i);
end

for l = 1:length(E)
    x_lam = zeros(n,1);  % GSVD filtered solution
    for i=1:p
        x_lam = x_lam + c(i)/(c(i)^2+lambda(l)^2 * s(i)^2) * b_hat(i) * X(:,i);
    end
    x_lam = x_lam + x_0;
    % size(x_true)
    y = x_lam - x_true;
    E(l) = sum(abs(y).^2);
end

E = sqrt(E);

end
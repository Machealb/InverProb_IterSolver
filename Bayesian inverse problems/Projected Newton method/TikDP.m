function [x, lambda] = TikDP(A, L, b, tau)
% This function computes a Tikhonov regularized solution of
%       min{||Ax-b||^2 + lambda^2*||Lx||^2}
% for mxn matrix A and pxn matrix L, where lambda is the
% regularization parameter estimated by discrepancy principle (DP).
%  
% Input: 
%   A, L:  min{||Ax-b||^2 + lambda^2||Lx||^2
%   b - noisy data
%   tau: used to form tau*||e||_2
%
%  Output: 
%    x - opitimal Tikhonov solution
%    e - solution error
%    lambda - regularization parameter by DP
%       
%  Haibo Li, School of Mathematics and Statistics, The University of Melbourne
%  01, Mar, 2024.

[m, n] = size(A);
[U, sm, X, ~, ~] = cgsvd(A,L);
b_hat = U'*b;
tol_dp = sqrt(tau*m);

func = @(x) abs(res_err(x, sm, U, b)-tol_dp);
% func = @(x) er_dp(x, sm, U, b, tol_dp);
lambda = fminbnd(func, 1e-2, 100);

% lambda = fminbnd(er_dp, 1e-6, 10, optimset('Display','off'), sm, U, b, tol_dp);
% lambda = fminbnd(res_err, 1e-6, 10, optimset('Display','off'), sm, U, b);

c = sm(:,1);
s = sm(:,2);
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

end


%----------------------------------------------
function er = er_dp(lambda, sm, U, b, tol)
    er1 = res_err(lambda, sm, U, b);
    er  = zeros(length(er1), 1);
    for l = 1:length(er1)
        if er1(l) > tol
            er(l) = 10000;
        else
            er(l) = tol - er1(l);
        end
    end

end

function err = res_err(lambda, sm, U, b)
    b_hat = U'*b;
    [m, n] = size(U);
    [p, ~] = size(sm);    % m>=n>=p
    c = sm(:,1);
    s = sm(:,2);
    
    if m > n
        res0 = b - U*b_hat;
        er0 = sum(res0.^2);
    else 
        er0 = 0;
    end

    E = zeros(length(lambda), 1);
    coef = zeros(n,1);    % coeficients of the solution w.r.t X
    for i = p+1:n
        coef(i) = b_hat(i);
    end

    for l = 1:length(E)
        for i=1:p
            coef(i) = c(i)/(c(i)^2+lambda(l)^2 * s(i)^2) * b_hat(i);
        end
        res = c(1:n) .* coef - b_hat;
        E(l) = sum(res.^2) + er0;
    end
    
    err = sqrt(E);

end

function R = srd(A)
% Compute the compact form of Squre Root Decomposition of a positive 
% semi-definite matrix A:  A = R'*R, where
% R is of the form p x n and has full row rank p.
% When p < n, it means that A is not positive definite.
%
% Inputs:
%   A: either (a) a full or sparse mxn matrix;
%   must be symmetric semi-definite.
%
% Outputs: 
%   R: Squre Root of A:  A = R'*R
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 06, July, 2023.
%
% Initialization
% 
[m, n] = size(A);
if m ~= n 
    error('The matrix must be square')
end

if min(min(abs(A-A'))) > 1e-14
    error('The matrix must be symmetric')
end

A = (A+A')/2;
A = full(A);

[U, S, ~] = svd(A);
r = rank(S);
s = diag(S);
U1 = U(:,1:r);
s1 = s(1:r);
ss1 = sqrt(s1);
S1 = diag(ss1);
R = (U1*S1)';

end
function [Q, R] = qr1(A)
% Modified QR factorization of a matrix A with shape (m, n) where m >= n .
% The returned matrix Q is an m*n orthogonal matrix, and the returned
% matrix R is an n*n upper triangular matrix with positive diagonal elements.
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 08, July, 2023.
%

[~, n] = size(A);
[q, r]=qr(A);
q=q(:, 1:n);
r=r(1:n, :);

P=eye(n);
for i=1:n
    if (r(i,i) < 0)
        P(i,i) = -1;
    end
end

Q = q * P;
R = P * r;
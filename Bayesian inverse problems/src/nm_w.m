function a = nm_w(x, N)
% N-weighted norm of x, N is symmetric
% positive definite

y = N * x;
nn = x' * y;
a = sqrt(nn);
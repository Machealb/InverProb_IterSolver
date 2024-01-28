clear; clc;

tau = 1e-6;
M = 100;
N = 100;
n = M*N;

x = ones(M*N,1);


D1 = zeros(M-1, M);  D2 = zeros(N-1, N);
for i = 1:M-1
D1(i,i) = 1;
D1(i,i+1) = -1;
end

for i = 1:N-1
    D2(i,i) = 1;    
    D2(i,i+1) = -1;
end
  
D1 = sparse(D1);   D2 = sparse(D2);
Dh = kron([D2; sparse(1,N)], speye(M));
Dv = kron(speye(N), [D1; sparse(1,M)]);
s = f_tau((Dh*x).^2 + (Dv*x).^2);
w_til = s.^(-1/4);
nn = length(w_til);
W_til = spdiags(w_til(:), 0, nn, nn);
W = [W_til, sparse(n,n); sparse(n,n), W_til];
L = W*[Dh; Dv];
L = sparse(L);


function y = f_tau(x)
tau = 1e-6;
x = x(:);
k = size(x,1);
y = zeros(k,1);
for i = 1:k
    if x(i) < 0
     error('x should be positive')
    end
    if x(i) < tau
        y(i) = tau;
    else
        y(i) = x(i);
    end
end
end
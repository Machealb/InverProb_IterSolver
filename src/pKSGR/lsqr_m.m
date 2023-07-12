function [xk, res, eta] = lsqr_m(A, b, k, tol, reorth, nA)
% A modified LSQR solver for linear least squares problem, the stopping 
% criterion is reweitten.
%
% Inputs:
%  A : matrix
%  b : right-hand term. They constitute the linear system min||Ax-b||_2
%  K : the maximum number of iterations 
%  tol: the tolerence used in the stopping criterion, i.e. stop when ||A'*r_k||/(||A||||r_k||) < tol
%  reorth : reorthogonalization for constructing Lanczos vectors by Lanczos bidiagonalization
%    reorth = 0 : no reorthogonalization,
%    reorth = 1 : reorthogonalization of by means of MGS,
%    reorth = 2 : double reorthogonalization
%  nA: norm of A
%      
% Outputs:
%   xk: computed solution at the last iteration
%   X : computed solutions, stored column-wise (at the iterations listed in K)
%   res: relative residual norms ||r_k||:=||Ax_k-b||/||b||
%   eta: norm of ||A'*r_k||/(||A||||r_k||)
%   stop: the iteration index that LSQR stop (satisfies the stopping criterion)
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences 
% June 13, 2023.
%

% Initialization.
b = b(:);
if (size(A,1) ~= size(b,1))
  error('Dimensions of A and b must be consistent.')
end

[~, n] = size(A); 
% nA = normest(A);
nb = norm(b);
%X = [];
res = [];
eta = [];

%The first step Golub-Kanhan Bidiagonalization
beta1 = norm(b);
u = b/beta1;  U(:,1) = u;
r = A'*u;
alpha = norm(r);  B(1,1)=alpha;
v = r/alpha;  V(:,1)=v;

% Prepare for LSQR iteration.
w = V(:,1);
phi_bar = beta1;
rho_bar = B(1,1);
x = zeros(n, 1);

% The i-th step Golub-Kahan-Lanczos Bidiagonalization and LSQR iteration
for i = 1:k
	% construct u_{i+1}
    p = A*v - alpha*u;
    if reorth==0
        % pass
    elseif reorth==1  
    	for j = 1:i  % MGS
    		p = p - U(:,j)*U(:,j)'*p;
    	end
    elseif reorth==2  % double reorthogonalization
    	for j = 1:i
    		p = p - U(:,j)*U(:,j)'*p;
    	end
    	for j = 1:i
    		p = p - U(:,j)*U(:,j)'*p;
    	end
    end
    beta = norm(p);  B(i+1,i) = beta;
    u = p/beta;   U(:,i+1)=u;
    
    % construct v_{i+1}
    r = A'*u - beta*v;
    if reorth==0
        % pass
    elseif reorth==1  
    	for j = 1:i  % MGS
    		r = r - V(:,j)*V(:,j)'*r;
    	end
    elseif reorth==2  % double reorthogonalization
    	for j = 1:i  
    		r = r - V(:,j)*V(:,j)'*r;
    	end
    	for j = 1:i  
    		r = r - V(:,j)*V(:,j)'*r;
    	end
    end
    alpha = norm(r);  B(i+1,i+1) = alpha;
    v = r/alpha;  V(:,i+1)=v;

    % Construct and apply orthogonal transformation.
    rrho = sqrt(rho_bar^2 + B(i+1,i)^2);
    c1 = rho_bar/rrho;
    s1 = B(i+1,i)/rrho; 
    theta = s1*B(i+1,i+1); 
    rho_bar = -c1*B(i+1,i+1);
    phi = c1*phi_bar;
    phi_bar = s1*phi_bar;  % residual norm at the i-th step (if U_k is left orthogonal)

    % Update the solution.
    x = x + (phi/rrho)*w;  
    %X = [X, x];
    w = V(:,i+1) - (theta/rrho)*w;
    er = b - A*x;
    res = [res; norm(er)/nb];
    rel = norm(A'*er) / (nA*norm(er));
    %rel = norm(A'*er) / (nA);
    eta = [eta; rel];
    
    % use the stopping criterion to stop the iteration
    if rel <= tol
        fprintf('Achieve desired accuracy at the %d-th step.\n', i);
        break;   
    end
end

xk = x;
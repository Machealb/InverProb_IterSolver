function [Z, rho, etaW, etaL, iterstop] = jbdqr(A, b, L, k, tol, reorth, eta)
% jbdqr uses the right subspace V_tilde generated by JointBid 
% to solve the general-form regularization problem
%   min{||Ax-b||_{2}^2 + lambda x'*M*x}
% by projecting it as
%   min{||Lx||_2} s.t. min||Ax-b||_2, where x \in span(Z_k) at the k-th step,
% and using discrepancy princple (DP) to stop iteration early.
%
% Inputs:
%   A: either (a) a full or sparse mxn matrix;
%             (b) a matrix object that performs the matrix*vector operation
%   b: right-hand side vector
%   L: regularization matrix
%   k: the maximum number of iterations 
%   tol: stopping tolerance of pcg.m for solving Gx = A'u
%       if tol=0, then solve it directly 
%   reorth: 
%       0: no reorthogonalization
%       1: full reorthogonaliation, MGS
%       2: double reorthogonaliation, MGS
%   eta: \tau*||e||_2 used in DP
%
% Outputs: 
%   Z: store the first k columns used to get regularized solutions
%   rho: strore residual norm of the first k regularized solution
%   etaW: strore ||y_k||
%   etaL: strore ||Lx_k||
%   iterstop: the early seopping iteration estimated by DP
% 
% Reference: [1]. M. E. Kilmer, P. C. Hansen and M. I. Espanol, A projection–based approach to general
%  form Tikhonov regularization, SIAM J. Sci. Comput., 29 (2007), pp. 315–330.
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 06, July, 2023.
% 

% Check for acceptable number of input arguments
if nargout == 5
    if nargin < 7
        error('JBDQR: need noise level for MDP');
    else
        terminate = 1;
        flag = 1;
        iterstop = 0;
    end
end

[m, n] = sizem(A); 
if n ~= size(L,2)
    error('A and L should have the same number of columns')
end
p = size(L,1);
Z = zeros(m+p, k);   
rho = zeros(k,1);   
etaW = zeros(k,1);   
etaL = zeros(k,1);  

fprintf('Start the JBDQR iteration ===========================================\n');
[B, ~, ~, ~, V_tilde, bbeta] = JointBid(A, b, L, k+1, tol, reorth);

% Intialiazation
w = V_tilde(:,1);
phi_bar = bbeta;
rho_bar = B(1,1);
z = zeros(m+p, 1);

fprintf('Start update procedure ===========================================\n');
for l = 1:k
    % Construct and apply orthogonal transformation.
    rrho = sqrt(rho_bar^2 + B(l+1,l)^2);
    c = rho_bar/rrho;
    s =  B(l+1,l)/rrho;
    theta = s*B(l+1,l+1);
    rho_bar = -c*B(l+1,l+1);
    phi = c*phi_bar;
    phi_bar = s*phi_bar;

    % Update the solution.
    z = z + (phi/rrho)*w;
    w = V_tilde(:,l+1) - (theta/rrho)*w;
    Z(:,l) = z;
    rho(l) = abs(phi_bar);
    etaW(l) = norm(z);
    etaL(l) = norm(z(m+1:m+p,:));

    % estimate optimal regularization step by MDP
    if nargin == 7 && terminate && flag 
        if(rho(l) <= eta)
            iterstop = l;
            flag = 0;
        end
    end
end

end

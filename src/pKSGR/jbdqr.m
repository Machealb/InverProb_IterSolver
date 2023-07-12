% 利用联合双对角化过程求解不适定问题
% 投影子问题求解的 y_k 按照递推关系来更新
% eta: 1.01*||e||

function [Z,rho,etaW,etaL,iterstop]=jbdqr(A,L,b,k,tol,reorth,eta)

terminate = 0;  % 标记是否利用MDP确定最佳迭代步数

if nargout == 5
    if nargin < 7
        error('JBDQR: need noise level for MDP');
    else
        terminate = 1;
        flag = 1;
        iterstop = 0;
    end
end

[m,~] = size(A);
p=size(L,1);
Z = zeros(m+p, k);  % 存储V_tilde(k)*y_k
% X = zeros(n, k);  %存储前k个解
rho = zeros(k,1);   % 存储前k个残差Ax_k-b
etaW = zeros(k,1);   % 存储前k个y_k(w_k)的范数
etaL = zeros(k,1);  % 存储前k个Lx_k的范数

fprintf('Start the JBDQR iteration ===========================================\n');
[B, ~, ~, ~, V_tilde, bbeta]=JointBid(A,L,b,k+1,tol,reorth);

% Intialiazation
w = V_tilde(:,1);
phi_bar = bbeta;
rho_bar = B(1,1);
z = zeros(m+p, 1);


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
    if(terminate && flag)
        if(rho(l) <= eta)
            iterstop = l;
            flag = 0;
        end
    end

end

end

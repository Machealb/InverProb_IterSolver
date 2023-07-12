function [reg_min, G, reg_param] = wgcv(U, s, b, omega)
% wgcv plots the WGCV function and find its minimum, where 
% the Tikhonov regularization is used.
% 1. For standard form regularization with {A,I}, wgcv compute
%   [reg_min,G,reg_param] = wgcv(U,s,b,method) 
% where s are singular values of A.
% 2. For general form regularization, with {A,L}, wgcv compute
%   [reg_min,G,reg_param] = wgcv(U,sm,b,method) where 
% sm = [sigma,mu] are generalized singular values of {A,L}.
%
% It plots the WGCV-function
%         || A*x_lam - b ||^2
%    G = ---------------------
%        (trace(I - wA*A_I)^2
% as a function of the regularization parameter reg_param. Here, A_I is a
% matrix which produces the regularized solution.
%
% Inputs:
%   U: left (generalized) singular vectors
%   s: (generalized) singular values
%   b: right-hand side vector
%   omega: weight parameter
%
% Outputs:
%   reg_min: minimer of the WGCV function
%   G: values of the WGCV function on several parameters 'reg_param'
%   reg_param: values of parameters used for determine minimizer of G
% 
% Reference: [1]. G. Wahba, "Spline Models for Observational Data", SIAM, 1990. 
%  [2]. Per Christian Hansen, Regularization Tools: A Matlab package for analysis 
%   and solution of discrete ill-posed problems, Numer. Algor., 6 (1994), 1-35.
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 06, July, 2023.
%

% Initialization
npoints = 200;                      % Number of points on the curve.
smin_ratio = 16*eps;                % Smallest regularization parameter.

[m, n] = size(U); 
[p, ps] = size(s);  % px1 or px2
beta = U'*b; 
beta2 = norm(b)^2 - norm(beta)^2;  % ||(I-U*U')b||^2

% generalized singular values in decreasing order
if ps==2
  s = s(p:-1:1, 1) ./ s(p:-1:1, 2);  
  beta = beta(p:-1:1);
end

% If any output arguments are specified, then the minimum of G is
% identified and the corresponding reg_param and reg_min is returned.
if nargout > 0
  find_min = 1; 
else 
  find_min = 0;
end

% set several values of regularization parameters
reg_param = zeros(npoints,1); 
G = reg_param; 
s2 = s.^2;
reg_param(npoints) = max([s(p), s(1)*smin_ratio]);
ratio = (s(1)/reg_param(npoints))^(1/(npoints-1));
% reg_para are of equiproportional distribution in decreasing order
for i = npoints-1:-1:1 
  reg_param(i) = ratio*reg_param(i+1); 
end

% Compute intrinsic residual, i.e. ||(I-U*U')b||^2
delta0 = 0;
if m > n && beta2 > 0
  delta0 = beta2; 
end

% vector of GCV-function values
for i=1:npoints
  G(i) = wgcvfun(reg_param(i), s2, beta(1:p), delta0, m, n, omega);
end 

% plot WGCV function.
loglog(reg_param,G,'-');
xlabel('\lambda');
ylabel('G(\lambda)');
title('WGCV function');

% find minimum of G as the estimated optimal regularizer
if find_min
  [~, minGi] = min(G);  % initial guess
  reg_min = fminbnd('wgcvfun',...
    reg_param(min(minGi+1,npoints)),reg_param(max(minGi-1,1)),...
    optimset('Display','off'),s2,beta(1:p),delta0,m,n,omega);  % Minimizer.
  minG = wgcvfun(reg_min,s2,beta(1:p),delta0,m,n,omega);  % Minimum of GCV function.

  % prepare for plotting WGCV function
  ax = axis;
  HoldState = ishold; 
  hold on;
  loglog(reg_min,minG,'*r',[reg_min,reg_min],[minG/1000,minG],':r')
  title(['WGCV function, minimum at \lambda = ',num2str(reg_min)])
  axis(ax)
  if ~HoldState
    hold off; 
  end
end

end

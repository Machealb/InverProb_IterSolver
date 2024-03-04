function [Q] = gen_kernel2d(a, b, c, d, N, type, varargin)
% generating kernel matrix Q that is the covaiance of the prior 
% of x for 2-dim problems of the interval [a, b]x[c,d]
% Q^{-1} will play the role as regularization matrix.
%
% Inputs:
%   a, b, c, d: 2D domain [a, b]x[c, d]
%   N:  number of uniform grids for discretizing [a, b]x[c, d]
%   type: kenel type
%     'gauss': K(x1, x2) = exp(-(x1-x2)^2/(2*l^2))
%     'exp':   K(x1, x2) = exp(-(|x1-x2|/l)^gamma)
%     'matern':
%   vargin: super-parameter of the corresponding kernel
%    'gauss': l
%    'exp':   l, gamma
%    'matern': l, nu
%
% Outputs: 
%   Q: covariance matrix of the kernel
% 
% Reference: [1]. Carl E. Rasmussen and Christopher K.I. Williams, Gaussian 
% Processes for Machine Learning”, MIT Press 2006 Gaussian, Chapter 4.2.
%
%  Haibo Li, School of Mathematics and Statistics, The University of Melbourne
%  01, Mar, 2024.

% Initialization
if nargin < 7
    error('Not enough inputs')
end

if strcmp(type, 'gauss')
    l = varargin{1};
elseif strcmp(type, 'exp')
    l = varargin{1};
    gamma1 = varargin{2};
elseif strcmp(type, 'matern')
    l = varargin{1};
    nu= varargin{2};
else
    % pass
end

% discretize [a, b]x[c, d] by uniform grids, and form N^2x1 point-vector, by column
x = linspace(a, b, N);
y = linspace(c, d, N);
x = x(:);
y = y(:);
X1 = kron(x, ones(1,N));    
X1 = X1(:);        % x-axis of points
Y1 = kron(ones(N,1), y'); 
Y1 = Y1(:);        % y-axis of points

% clear(x);  clear(y);
% clear(X1); clear(Y1);

% interaction between (X1,Y1) and (X1,Y1), form [(XX1, YY1), (XX2, YY2)]
n = N * N;
XX1 = kron(X1, ones(1,n));   
XX1 = XX1(:);
YY1 = kron(Y1, ones(1,n));
YY1 = YY1(:);
XX2 = kron(ones(n,1), X1');
XX2 = XX2(:);
YY2 = kron(ones(n,1), Y1');
YY2 = YY2(:);

% % construct kernel covariance matrix, by vectorized function evaluation
if strcmp(type, 'gauss')
    func = @(x1,x2,y1,y2) gauss2d(x1,x2,y1,y2,l);
elseif strcmp(type, 'exp')
    func = @(x1,x2,y1,y2) exp2d(x1,x2,y1,y2,l,gamma1);
elseif strcmp(type, 'matern')
    func = @(x1,x2,y1,y2) matern2d(x1,x2,y1,y2,l,nu);
end

VV = arrayfun(func, XX1, YY1, XX2, YY2);  % applies the function to each elements of arrays
Q  = reshape(VV, n, n);

end


%-------- 2d kernel functions -------------
function val = gauss2d(x1,x2,y1,y2,l)
    r2   = (x1-y1)^2 + (x2-y2)^2;
    val = exp(-r2) / (2.0*l^2);
end

function val = exp2d(x1,x2,y1,y2,l,gamma1)
    r   = sqrt((x1-y1)^2 + (x2-y2)^2);
    t   = -(r/l)^gamma1;
    val = exp(t);
end

function val = matern2d(x1,x2,y1,y2,l,nu)
    r   = sqrt((x1-y1)^2 + (x2-y2)^2);
    h   = sqrt(2*nu) * r / l;
    if r == 0
        val = 1.0;
    else
        val = 2^(1-nu) / gamma(nu) * h^nu * besselk(nu,h);
    end
end

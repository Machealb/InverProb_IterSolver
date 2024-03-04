function Q = gen_kernel1d(a, b, n, type, varargin)
% generating kernel matrix Q that is the covaiance of the prior 
% of x for 1-dim problems of the interval [a, b],
% Q^{-1} will play the role as regularization matrix.
%
% The interval [a, b] is discretized with midpoint rule.
%
% Inputs:
%   a, b: interval [a, b]
%   n:  number of midpoints for discretizing [a, b]
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
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 09, Sept, 2023.

% Initialization
if nargin < 5
    error('Not enough inputs')
end

if strcmp(type, 'gauss')
    l = varargin{1};
elseif strcmp(type, 'exp')
    l = varargin{1};
    gamma = varargin{2};
elseif strcmp(type, 'matern')
    l = varargin{1};
    nu= varargin{2};
else
    % pass
end

% discretized x at the mid-points
s = (b-a) / n;
x = zeros(n, 1);
Q = zeros(n, n);
for i = 1:n 
    x(i) = a + s * (i-1/2);
end


% construct kernel covariance matrix
for i = 1:n 
    for j = 1:n
        r = abs(x(i) - x(j));
        if strcmp(type, 'gauss')
            Q(i,j) = exp(-r^2/(2*l^2));
        elseif strcmp(type, 'exp')
            Q(i,j) = exp(-(r/l)^gamma);
        elseif strcmp(type, 'matern')
            h = sqrt(2*nu) * r / l;
            Q(i,j) = 2^(1-nu) / gamma(nu) * h^nu * besselk(nu,h);
        else
            % pass
        end
    end
end

end
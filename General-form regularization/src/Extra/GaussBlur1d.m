function A = GaussBlur1d(n, sigma, k, boundary)
% A is the ill-posed matrix of 1 dimensional Gauss blur with kernel f(t),
%   f(t) = 1/(sqrt(2pi)*sigma) * exp(1x^2/(2*sigma^2))
% is the standart center Gaussian ditrbution.
%
% The parameter sigma controls the width of the Gaussian point spread
% function and thus the amount of smoothing (the larger the sigma, the wider
% the function and the more ill posed the problem). If sigma is not
% specified, sigma = 0.7 is used.
%
% The discrete convolution y(t) = (f*x)(t) defined as:
%   \sum_{t=-k}^{t=k}f(0-t*sigma)*x(t),
% where k it the band of the discrete convolution kernel.
% Usually k is set as the minimum integer no less than 3*sigma.
% The vector of the descretized f(x) used for deconvolution is
% [f(-k),...,f(-1),f(0),f(1),...,f(k)].
%
% Inputs:
%   n: dimension of the 1-D signal (n should big than k)
%   sigma: standard deviation of the Gaussian distribution
%   k: band of the discrete convolution kernel
%   boundary: padding methods at boundary of the 1-D signal
%       'zero': zero boundary condition，
%           A is a Toeplitz matrix            
%       'periodic': periodic boundary condition，
%           A is a circuland matrix
%       'reflextive': reflextive boundary condition，
%           A is a Toeplitz+Hankel matrix
%
% Outputs:
%   A: The matrix of the 1-D Gaussian blur
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 18, May, 2023.
% 

% Check for acceptable number of input arguments
if nargin == 1
    sigma = 0.7;  k = 3;  boundary = 'zero';
elseif nargin == 2
    k = ceil(3*sigma);  boundary = 'zero';
elseif nargin == 3
    boundary = 'zero';
else

end

% Construct the symmetric Toeplitz matrix
d = zeros(n,1);  d = d(:);
d(1) = f(0);
for i = 1:k
    d(1+i) = f(i);
end

T = toeplitz(d);

% Construct the matrix corresponding to the periodic padding
if strcmp(boundary, 'periodic')
    C1 = zeros(n,n);  
    for i = 1:k
        for j = n-k+i:n
            C1(i,j) = d(1+n+i-j);
        end
    end
    C2 = zeros(n,n);
    for i = n-k+1:n
        for j = 1:k-n+i
            C2(i,j) = d(1+n-i+j);
        end
    end
end

% Construct the Hankel matrix corresponding to the reflextive padding
if strcmp(boundary, 'periodic')
    H1 = hankel(d);
    H2 = zeros(n,n);
    for i = n-k+1:n
        for j = 2*n-k-i+1:n
            H2(i,j) = d(1+2*n - i -j + 1);
        end
    end
end

% Construct the matrix as a Kronecker product.
if strcmp(boundary, 'zero')
    A = T;
elseif strcmp(boundary, 'periodic')
    A = T + C1 + C2;
elseif strcmp(boundary, 'reflective')
    A = T + H1 + H2;
else
    error('Not have this boudary condition')
end

A = sparse(A);

% z = [exp(-((0:band-1).^2)/(2*sigma^2)),zeros(1,N-band)];
% A = toeplitz(z);

%%----------------------------
function y = f(x)
    y = 1/(sqrt(2*pi)*sigma) * exp(-x^2/(2*sigma^2));
end

end
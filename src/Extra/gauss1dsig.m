function [A, b, x] = gauss1dsig(n, sigma)
% Generates a 1-D square wave signal x, and convolve it
% by a Gaussian kernel.
% See the function GauuBlur1d.m
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 18, May, 2023.
% 

if n ~= 800
    error('Only n=800 can be set now \n')
end

n = 800;
x = zeros(n,1);
x(1:100) = 0.0;
x(101:200) = -0.2;
x(201:400) = 0.6;
x(401:500) = 0.2;
x(501:650) = -0.4;
x(651:800) = 0.0;

A = GaussBlur1d(n, sigma);
b = A * x;
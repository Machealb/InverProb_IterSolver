function [f, I] =vec2fun(y, a, b)
% Translate a vector to its corresponing function that 
% used to plot the function, where the function f and the 
% vector y satisfies:
%       y = (f(x1), ... f(xn)), and 
% (x1, ..., xn) is the unuform grids by discretizing [a, b] by
% midrules, i.e.
% xi = a + (i-1/2)*(b-a)/n
% 
% Inputs:
%   y: the discretized value of f on unuform grids of [a, b]
%   a, b: the definition domain of f
% 
% Outputs:
% f: f=y, i.e., fi = f(xi)=yi
% I: the real discretized interval, I(i)=xi
%
% Haibo Li, School of Mathematics and Statistics, The University of Melbourne
% 05, Oct, 2023.

n = size(y, 1);
I = zeros(n, 1);
f = y;
h = (b-a) / n;
for i = 1:n 
    I(i) = a + h * (i-1/2);
end

end
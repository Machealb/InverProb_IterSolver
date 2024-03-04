clear; clc;

% X = 0.4; 
% Y = 0.5;

n = 50;
[X, Y] = meshgrid(linspace(0,1,n));
% X = X(:);
% Y = Y(:);

x = 0.7*exp( -((X-0.4)/0.12).^2 - ((Y-0.5)/0.15).^2) + ...
        exp( -((X-0.7)/0.1 ).^2 - ((Y-0.4)/0.08).^2);
    
    
figure;
surf(X, Y, x);
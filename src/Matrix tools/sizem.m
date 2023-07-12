function [m ,n] = sizem(A);
% Get the size for a matrix A, where A can be a 
%   (a). full or sparse mxn matrix;
%   (b). a matrix object that performs the matrix*vector operation
%   (c). a functional handle
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 08, July, 2023.
%

if isa(A, 'function_handle')
    dim = A([], 'size');
    m = dim(1);
    n = dim(2);
else
    [m, n] = size(A);
end


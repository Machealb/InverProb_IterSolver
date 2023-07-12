function y = mvpt(A, x)
% Matrix vector multiplication  with transpose of A :
%   y = A * x,
% written for the case that A is a functional handle
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 08, July, 2023.
%

if isa(A, 'function_handle')
    flag = 'transp';
    y = A(x, flag);
else
    y = A' * x;
end

end
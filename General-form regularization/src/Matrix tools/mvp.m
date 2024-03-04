function y = mvp(A, x)
% Matrix vector multiplication:
%   y = A * x,
% written for the case that A is a functional handle
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 08, July, 2023.
%

if isa(A, 'function_handle')
    flag = 'notransp';
    y = A(x, flag);
else
    y = A * x;
end

end
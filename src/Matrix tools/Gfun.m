function y = Gfun(z, A, M, alpha)
% Let G = A'A+alpha*M, where M is symmetric.
% This function computes G*z for a vector z.
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 04, July, 2023.

if isa(A, 'function_handle')
    az = A(z, 'notransp');
    v1 = A(az, 'transp');
else
    v1 = A' * (A * z);
end

v2 = alpha * M * z;
y = v1 + v2;

end

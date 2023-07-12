function y =afun(z,A,B,transp_flag)
if strcmp(transp_flag,'transp')   % y = (A(I_n-BB^T))' * z;
    m = size(A,1);
    p = size(B,1);
    s = A'*z(1:m);
    t = B'*z(m+1:m+p);
    y = s + t;
elseif strcmp(transp_flag,'notransp') % y = (A(I_n-BB^T)) * z;
    s = A*z;
    t= B*z;
    y = [s;t];
    
end
end

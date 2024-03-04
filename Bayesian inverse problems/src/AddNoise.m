function b = AddNoise(b_true, type, nel)
% add noise e to b_true, with relative noise level
% nel = ||e||_2/||b_true||_2
%
% Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
% 06, July, 2023.
%
[m, n] = size(b_true);

if strcmp(type, 'gauss')
    e = randn(m, n);
    nr = norm(e);
    e =  e / nr * nel * norm(b_true);
else
    % pass
end

b = b_true + e;

end
function [e, Sigma] = genNoise(b_true, nel, type)
    % generate Gaussian noise e with zero mean and covariance Sigma,
    % where Sigma is a diagonal matrix.
    %
    % Inputs:
    %   b_true: true right-hand side
    %   nel: E(||e||^2)/||b_true|| = nel
    %   type: 
    %      'white': white noise, Sigma = eta*I;
    %      'nonwt': non-white noise, but Sigma is diagonal
    %
    % Outputs:
    %   e: the n-dim noise vector (one sample)
    %   Sigma: covariance matrix of e
    %
    % Haibo Li, Institute of Computing Technology, Chinese Academy of Sciences
    % 10, Sept, 2023.
    %
    
    rng('default')  % random seed, for reproducibility
    m = size(b_true, 1);
    nb = norm(b_true);
    mu = zeros(1, m);

    if strcmp(type, 'white')
        % eta = (nb * nel)^2 / m;
        % Sigma = eta * eye(m);    % the case for the white noise need to be modified!!
        s = ones(m,1);
        a = sum(s);
        gamma = (nb * nel)^2 / a;
        S = spdiags(s, 0, m, m);
        Sigma = gamma * S;
        e1 = randn(m, 1);
        nr = norm(e1);
        e =  e1 / nr * nel * norm(b_true);
    else
        s = randi([1,5], m, 1);  % mx1 uniform random vector of integer between [1, 5]
        a = sum(s);
        gamma = (nb * nel)^2 / a;
        S = spdiags(s, 0, m, m);
        Sigma = gamma * S;
        e1 = randn(m, 1);
        e  = sqrt(gamma*s) .* e1;
    end

    e = e(:);
    
end
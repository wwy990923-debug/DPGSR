function [W] = Reverse(S,T,groupInfo,lambda)
% Reverse path for DP-GSR.
%
% Solve:
%   min_W ||S*W - T||_F^2 + lambda * sum_j sum_k ||W(j, Omega_k)||_2
%
% For DP-GSR reverse path, call:
%   W = Reverse(T_test, S_train, nt_or_Slabel, lambda2)
%
% Inputs:
%   S         - reverse dictionary, usually test data T_test, size d x M
%   T         - reverse target, usually training data S_train, size d x N
%   groupInfo - scalar: nt (equal class sizes); vector: T_label (variable sizes)
%   lambda    - reverse regularization parameter
%
% Output:
%   W         - reverse coefficient matrix, size M x N

tol = 1e-4;
maxIter = 500;

[d1, n] = size(S);   % n = M, number of test samples
[d2, p] = size(T);   % p = N, number of training samples

if d1 ~= d2
    error('S and T must have the same feature dimension.');
end

if numel(groupInfo) == 1
    nt = groupInfo;
    if mod(p, nt) ~= 0
        error('The number of columns in T must be divisible by nt.');
    end
    T_label = repelem(1:(p/nt), nt);
else
    T_label = groupInfo(:)';
end

if length(T_label) ~= p
    error('Length of T_label/groupInfo must equal the number of columns in T.');
end
classes = unique(T_label, 'stable');

rho = 1.2;
max_mu = 1e20;
mu = 1.1;

%% Initialize variables
W  = zeros(n,p);
F  = zeros(n,p);
Y1 = zeros(n,p);

iter = 0;
obj_values = zeros(maxIter, 1);
loss_values = zeros(maxIter, 1);

while iter < maxIter

    iter = iter + 1;

    %% ============================================================
    % Update F: group L2 shrinkage (row-wise, per class block)
    %% ============================================================

    temp = W - Y1/mu;                  % n x p = M x N

    F = zeros(n, p);
    for kk = 1:length(classes)
        idx = (T_label == classes(kk));
        F(:, idx) = solve_L12(temp(:, idx), lambda/mu);
    end

    %% ============================================================
    % Update W
    %% ============================================================

    inv_b = inv(2*S'*S + mu*eye(n));

    W = inv_b * (2*S'*T + mu*(F + Y1/mu));

    %% ============================================================
    % Check convergence
    %% ============================================================

    leq1 = F - W;
    stopC = max(abs(leq1(:)));

    loss_values(iter) = stopC;

    obj_val = norm(S*W - T, 'fro')^2;
    for kk = 1:length(classes)
        idx = (T_label == classes(kk));
        obj_val = obj_val + lambda * sum(sqrt(sum(W(:, idx).^2, 2)));
    end
    obj_values(iter) = obj_val;

    if (iter == 1 || mod(iter,50) == 0 || stopC < tol)
        disp(['iter ' num2str(iter) ...
            ', mu=' num2str(mu,'%2.1e') ...
            ', rank=' num2str(rank(W,1e-4*norm(W,2))) ...
            ', stopALM=' num2str(stopC,'%2.3e') ]);
    end

    if stopC < tol
        break;
    else
        Y1 = Y1 + mu*leq1;
        mu = min(max_mu, mu*rho);
    end
end

end


function [E] = solve_L12(M,lambda)
n = size(M,1);
E = M;
for i = 1:n
    E(i,:) = solve_l1(M(i,:),lambda);
end
end


function [x] = solve_l1(w,lambda)
nw = norm(w,2);
if nw > lambda
    x = (1 - lambda/nw) * w;
else
    x = zeros(size(w));
end
end

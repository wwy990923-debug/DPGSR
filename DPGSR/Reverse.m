function [W] = Reverse(S,T,nt,lambda)
% Reverse path for DP-GSR.
%
% Solve:
%   min_W ||S*W - T||_F^2 + lambda * sum_j sum_k ||W(j, Omega_k)||_2
%
% For DP-GSR reverse path, call:
%   W = Reverse(T_test, S_train, nt, lambda2)
%
% Inputs:
%   S      - reverse dictionary, usually test data T_test, size d x M
%   T      - reverse target, usually training data S_train, size d x N
%   nt     - number of training samples per class
%   lambda - reverse regularization parameter
%
% Output:
%   W      - reverse coefficient matrix, size M x N

tol = 1e-4;
maxIter = 500;

[d1, n] = size(S);   % n = M, number of test samples
[d2, p] = size(T);   % p = N, number of training samples

if d1 ~= d2
    error('S and T must have the same feature dimension.');
end

if mod(p, nt) ~= 0
    error('The number of columns in T must be divisible by nt.');
end

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
    % Update F: group L2 shrinkage via L12
    %
    % W is M x N. Row j = test sample j.
    % Columns grouped by class: nt consecutive cols per class.
    % Group: W(j, Omega_k) = nt elements in row j for class k.
    % Reshape: (M*K) x nt, each ROW = one group → L12 (row-wise L2)
    %% ============================================================

    temp = W - Y1/mu;                  % M x N = n x p

    % MATLAB reshape is column-major. Transpose first so each row of P is
    % exactly one reverse group W(j, Omega_k), with nt consecutive atoms.
    P = reshape(temp', nt, [])';       % (M*K) x nt, each ROW = one group
    J = solve_L12(P, lambda/mu);       % row-wise L2 shrinkage

    F = reshape(J', p, n)';            % back to M x N

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
    for j = 1:n
        wj = W(j, :)';
        Qj = reshape(wj, nt, p/nt);
        obj_val = obj_val + lambda * sum(sqrt(sum(Qj.^2, 1)));
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

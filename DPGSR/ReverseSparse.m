function [W] = ReverseSparse(S, T, nt, lambda2, lambda3)
% ReverseSparse — 带逐样本稀疏约束的逆向求解器
%
% min_W  ||S*W - T||_F^2 + lambda2 * sum_j sum_k ||W(j,Omega_k)||_2
%                        + lambda3 * ||W||_1
%
%  lambda3: element-wise L1 稀疏度，让每个测试样本只重构少数训练样本
%           会大幅降低 Z 的协同性，提升逐样本判别力
%
% 本函数实现"把逐样本 Z(j,:) 从协同编码变为稀疏匹配检索"

if nargin < 5 || isempty(lambda3)
    lambda3 = 0;  % 退化回标准 Reverse
end

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

%% Initialize
W  = zeros(n,p);
F  = zeros(n,p);    % group sparsity auxiliary
G  = zeros(n,p);    % L1 sparsity auxiliary
Y1 = zeros(n,p);    % multiplier for group penalty
Y2 = zeros(n,p);    % multiplier for L1 penalty

iter = 0;
obj_values = zeros(maxIter, 1);

while iter < maxIter
    iter = iter + 1;

    %% Update W (least squares with two ADMM penalties)
    % min ||S*W-T||^2 + mu/2*||W-F+Y1/mu||^2 + mu/2*||W-G+Y2/mu||^2
    % → (2S'S + 2mu I)W = 2S'T + mu(F-Y1/mu + G-Y2/mu)

    M_w = 2*(S'*S) + 2*mu*eye(n);
    RHS_w = 2*(S'*T) + mu*(F + G - (Y1+Y2)/mu);
    W = M_w \ RHS_w;

    %% Update F (group L12 shrinkage)
    temp = W - Y1/mu;
    P = reshape(temp', nt, [])';
    J = solve_L12(P, lambda2/mu);
    F = reshape(J', p, n)';

    %% Update G (element-wise L1 shrinkage)
    temp2 = W - Y2/mu;
    G = soft_threshold(temp2, lambda3/mu);

    %% Multiplier updates
    leq1 = F - W;
    leq2 = G - W;
    Y1 = Y1 + mu*leq1;
    Y2 = Y2 + mu*leq2;

    %% Convergence
    stopC = max(max(abs(leq1(:))), max(abs(leq2(:))));
    if iter == 1 || mod(iter,50) == 0 || stopC < tol
        obj_val = norm(S*W - T, 'fro')^2;
        for j = 1:n
            wj = W(j,:)';
            Qj = reshape(wj, nt, p/nt);
            obj_val = obj_val + lambda2 * sum(sqrt(sum(Qj.^2,1)));
        end
        obj_val = obj_val + lambda3 * sum(abs(W(:)));
        obj_values(iter) = obj_val;
        fprintf('iter %d, mu=%.1e, stop=%.3e, W_nnz=%.1f%%, obj=%.4e\n', ...
            iter, mu, stopC, 100*nnz(W)/numel(W), obj_val);
    end

    if stopC < tol
        break;
    else
        mu = min(max_mu, mu*rho);
    end
end

fprintf('  ReverseSparse done: %d iters, W density=%.1f%%\n', ...
    iter, 100*nnz(W)/numel(W));
end

function [E] = solve_L12(M, lambda)
n = size(M,1);
E = M;
for i = 1:n
    E(i,:) = solve_l1(M(i,:), lambda);
end
end

function [x] = solve_l1(w, lambda)
nw = norm(w,2);
if nw > lambda
    x = (1 - lambda/nw) * w;
else
    x = zeros(size(w));
end
end

function [x] = soft_threshold(v, lambda)
x = sign(v) .* max(abs(v) - lambda, 0);
end

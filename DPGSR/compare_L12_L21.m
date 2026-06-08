% compare_L12_L21.m
% Side-by-side comparison of L12 vs L21 reverse implementations

rng(1);
load('yaleborigin.mat');
data = dataset;

K = 3; number = 64; nTrain = 4; nt = nTrain;
tc = cell(K,1); sc = cell(K,1);
for c = 1:K
    p = randperm(number);
    tc{c} = p(1:nTrain); sc{c} = p(nTrain+1:end);
end
[S, Sl] = build_set(data, tc, number);
[T, Tl] = build_set(data, sc, number);

% Run DPGSR with the CURRENT Reverse.m (L12 version)
[~, ~, out_l12] = DPGSR_Fusion(S, T, Sl, Tl, nt, 1e-2, 1, 'adaptive');
D_l12 = out_l12.D;

% Now manually compute ONE reverse iteration with both methods
[d, M] = size(T);   % T = test = S in Reverse
[d, N] = size(S);   % S = train = T in Reverse

% Setup reverse variables
W = zeros(M, N);
F1 = zeros(M, N);  % L12 version
F2 = zeros(M, N);  % L21 version
Y1 = zeros(M, N);
mu = 1.1;
lambda = 1;

% One iteration
temp = W - Y1/mu;  % M x N

% --- L12 version (current Reverse.m) ---
tempT = temp';  % N x M
P_l12 = reshape(tempT, (N/nt)*M, nt);  % (K*M) x nt
J_l12 = solve_L12_test(P_l12, lambda/mu);
F1_result = reshape(J_l12, N, M)';

% --- L21 version (transpose approach) ---
tempT2 = temp';  % N x M
P_l21 = reshape(tempT2, nt, (N/nt)*M);  % nt x (K*M)
J_l21 = solve_L21_test(P_l21, lambda/mu);
F2_result = reshape(J_l21, N, M)';

diff_F = max(abs(F1_result(:) - F2_result(:)));

fprintf('Max difference F1(F12) vs F2(L21): %.10e\n', diff_F);

if diff_F < 1e-15
    fprintf('L12 and L21 produce IDENTICAL results!\n');
else
    fprintf('L12 and L21 are DIFFERENT!\n');
    % Show first few groups
    fprintf('\nFirst 3 groups comparison:\n');
    for g = 1:3
        g_l12 = P_l12(g, :);
        g_l21 = P_l21(:, g)';
        fprintf('Group %d: L12=[%.4f %.4f], L21=[%.4f %.4f]\n', ...
            g, g_l12(1), g_l12(min(2,end)), g_l21(1), g_l21(min(2,end)));
    end

    fprintf('\nAfter shrinkage:\n');
    for g = 1:3
        g_l12 = J_l12(g, :);
        g_l21 = J_l21(:, g)';
        fprintf('Group %d: L12=[%.4f %.4f], L21=[%.4f %.4f]\n', ...
            g, g_l12(1), g_l12(min(2,end)), g_l21(1), g_l21(min(2,end)));
    end
end

function [X, label] = build_set(dataset, index_cell, n_per_class)
K = length(index_cell); X = []; label = [];
for c = 1:K
    for ii = 1:length(index_cell{c})
        X = [X, dataset(:, (c-1)*n_per_class + index_cell{c}(ii))];
        label = [label, c];
    end
end
end

function [E] = solve_L12_test(M, lambda)
n = size(M,1); E = M;
for i = 1:n
    E(i,:) = solve_l1_test(M(i,:), lambda);
end
end

function [E] = solve_L21_test(M, lambda)
n = size(M,2); E = M;
for i = 1:n
    E(:,i) = solve_l2_test(M(:,i), lambda);
end
end

function [x] = solve_l1_test(w, lambda)
nw = norm(w,2);
if nw > lambda; x = (1 - lambda/nw) * w; else; x = zeros(size(w)); end
end

function [x] = solve_l2_test(w, lambda)
nw = norm(w,2);
if nw > lambda; x = (1 - lambda/nw) * w; else; x = zeros(length(w),1); end
end

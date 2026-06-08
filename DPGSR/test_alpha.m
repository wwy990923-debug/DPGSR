% test_alpha.m - Check alpha distribution after S_G fix
rng(1);
load('yaleborigin.mat');
data = dataset;

K = 5;
number = 64;
nTrain = 4;
nt = nTrain;
dataset_sub = data(:, 1:K*number);

train_cell = cell(K,1);
test_cell = cell(K,1);
for c = 1:K
    perm = randperm(number);
    train_cell{c} = perm(1:nTrain);
    test_cell{c} = perm(nTrain+1:end);
end

[S, S_label] = build_set_by_class(dataset_sub, train_cell, number);
[T, T_label] = build_set_by_class(dataset_sub, test_cell, number);

fprintf('K=%d, nt=%d, N=%d, M=%d\n\n', K, nt, size(S,2), size(T,2));

% Test joint solver with different gammas
gammas = [1e-2, 1e-1, 1, 10];
fprintf('=== Joint solver (new S_G) ===\n');
for g = gammas
    [~, acc, out] = DPGSR_Fusion(S, T, S_label, T_label, nt, g, g, 'adaptive', [], 'joint');
    a = out.alpha_j_true;
    fprintf('gamma=%.0e: acc=%.4f, alpha=[min=%.3f, max=%.3f, mean=%.3f, std=%.3f]\n', ...
        g, acc, min(a), max(a), mean(a), std(a));
end

% Compare with separate solver
fprintf('\n=== Separate solver (new S_G) ===\n');
[~, acc, out] = DPGSR_Fusion(S, T, S_label, T_label, nt, 1e-2, 1e-2, 'adaptive', [], 'separate');
a = out.alpha_j_true;
fprintf('lambda=1e-2: acc=%.4f, alpha=[min=%.3f, max=%.3f, mean=%.3f, std=%.3f]\n', ...
    acc, min(a), max(a), mean(a), std(a));

% Show a few individual alpha values
fprintf('\n=== Sample alpha_j values (joint, gamma=1e-2) ===\n');
[~, ~, out] = DPGSR_Fusion(S, T, S_label, T_label, nt, 1e-2, 1e-2, 'adaptive', [], 'joint');
a_sample = out.alpha_j_true(1:min(10, length(out.alpha_j_true)));
fprintf('First 10 samples: '); fprintf('%.3f ', a_sample); fprintf('\n');

function [X, label] = build_set_by_class(dataset, index_cell, number_per_class)
K = length(index_cell);
X = [];
label = [];
for c = 1:K
    idx_list = index_cell{c};
    for ii = 1:length(idx_list)
        col_id = (c-1)*number_per_class + idx_list(ii);
        X = [X, dataset(:, col_id)];
        label = [label, c];
    end
end
end

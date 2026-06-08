% test_nt_sweep.m — 小样本下融合优势是否放大
clc;
rng(1);
load('yaleborigin.mat');
data = dataset;

K = 15;
number = 64;
nSplits = 5;
nt_vals = [2, 3, 4, 5, 6, 8];
lambda1 = 1e-2;
lambda2 = 1e-2;

fprintf('============================================\n');
fprintf('Small-sample DPGSR Fusion Test (K=%d)\n', K);
fprintf('============================================\n\n');

fprintf('%-6s %8s %8s %8s %8s %8s %8s\n', ...
    'nt', 'FwdAcc', 'RevAcc', 'FusAcc', 'Broken', 'Fixed', 'Net');
fprintf('%-6s %8s %8s %8s %8s %8s %8s\n', ...
    '--', '------', '------', '------', '------', '-----', '---');

for ni = 1:length(nt_vals)
    nt = nt_vals(ni);

    all_fwd = []; all_rev = []; all_fus = [];
    all_broken = []; all_fixed = [];

    for m = 1:nSplits
        train_cell = cell(K,1);
        test_cell = cell(K,1);
        for c = 1:K
            perm = randperm(number);
            train_cell{c} = perm(1:nt);
            test_cell{c} = perm(nt+1:end);
        end
        [S, S_label] = build_set(data, train_cell, number);
        [T, T_label] = build_set(data, test_cell, number);

        [pred_fusion, ~, out] = DPGSR_Fusion(S, T, S_label, T_label, nt, ...
            lambda1, lambda2, 'adaptive');

        [~, pred_fwd] = max(out.PF, [], 1);
        [~, pred_rev] = max(out.PR, [], 1);

        fwd_pred = out.classes(pred_fwd);
        rev_pred = out.classes(pred_rev);
        M = length(T_label);

        all_fwd = [all_fwd, sum(fwd_pred==T_label)/M];
        all_rev = [all_rev, sum(rev_pred==T_label)/M];
        all_fus = [all_fus, sum(pred_fusion==T_label)/M];
        all_broken = [all_broken, sum(fwd_pred==T_label & pred_fusion~=T_label)];
        all_fixed  = [all_fixed,  sum(fwd_pred~=T_label & pred_fusion==T_label)];
    end

    fprintf('%-6d %8.4f %8.4f %8.4f %8d %8d %8d\n', ...
        nt, mean(all_fwd), mean(all_rev), mean(all_fus), ...
        round(mean(all_broken)), round(mean(all_fixed)), ...
        round(mean(all_fixed)-mean(all_broken)));
end

fprintf('\nNote: nt=8 is the original K=15 setting.\n');

function [X, label] = build_set(dataset, index_cell, n_per_class)
K = length(index_cell);
X = []; label = [];
for c = 1:K
    for ii = 1:length(index_cell{c})
        col_id = (c-1)*n_per_class + index_cell{c}(ii);
        X = [X, dataset(:, col_id)];
        label = [label, c];
    end
end
end

% test_sparse_rev.m — 逐样本稀疏逆向 vs 标准逆向
clc; rng(1);
load('yaleborigin.mat');
data = double(dataset);

K = 15; number = 64; nTrain = 8; nt = nTrain; nSplits = 3;
lambda1 = 1e-2; lambda2_vals = [1e-2, 1e-1, 1];
lambda3_vals = [0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2];

fprintf('Sparse Reverse Test (K=%d, nt=%d)\n', K, nt);
fprintf('lambda3 > 0: L1 penalty per row of Z\n\n');

fprintf('%-8s %-8s %10s %10s %10s %8s %8s %10s\n', ...
    'lam2', 'lam3', 'FwdAcc', 'RevAcc', 'FusAcc', 'Broken', 'Fixed', 'ZDensity');
fprintf('%-8s %-8s %10s %10s %10s %8s %8s %10s\n', ...
    '----', '----', '------', '------', '------', '------', '-----', '--------');

for li2 = 1:length(lambda2_vals)
    lam2 = lambda2_vals(li2);
    for li3 = 1:length(lambda3_vals)
        lam3 = lambda3_vals(li3);

        fwd_acc = zeros(nSplits,1); rev_acc = zeros(nSplits,1);
        fus_acc = zeros(nSplits,1); broken = zeros(nSplits,1);
        fixed = zeros(nSplits,1); zdens = zeros(nSplits,1);

        for m = 1:nSplits
            tc = cell(K,1); sc = cell(K,1);
            for c = 1:K
                perm = randperm(number);
                tc{c} = perm(1:nTrain);
                sc{c} = perm(nTrain+1:end);
            end
            [S, S_label] = build_set(data, tc, number);
            [T, T_label] = build_set(data, sc, number);

            [pred_fusion, ~, out] = DPGSR_Fusion(S, T, S_label, T_label, nt, ...
                lambda1, lam2, 'adaptive', [], 'separate', [], [], [], [], [], [], lam3);

            [~, pf] = max(out.PF,[],1); [~, pr] = max(out.PR,[],1);
            M = length(T_label);
            fwd_acc(m) = sum(out.classes(pf)==T_label)/M;
            rev_acc(m) = sum(out.classes(pr)==T_label)/M;
            fus_acc(m) = sum(pred_fusion==T_label)/M;
            broken(m) = sum(out.classes(pf)==T_label & pred_fusion~=T_label);
            fixed(m)  = sum(out.classes(pf)~=T_label & pred_fusion==T_label);
            zdens(m) = nnz(out.D) / numel(out.D);
        end

        fprintf('%-8.0e %-8.0e %10.4f %10.4f %10.4f %8d %8d %9.1f%%\n', ...
            lam2, lam3, mean(fwd_acc), mean(rev_acc), mean(fus_acc), ...
            round(mean(broken)), round(mean(fixed)), 100*mean(zdens));
    end
    fprintf('\n');
end

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

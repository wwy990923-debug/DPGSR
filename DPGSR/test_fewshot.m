% test_fewshot.m — 极少训练样本下融合效果
clc;
rng(1);
load('yaleborigin.mat');
data = double(dataset);

K = 15;
number = 64;
nSplits = 5;
nt_vals = [1, 2, 3];
lambda_grid = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1];

fprintf('Few-shot DPGSR Fusion (K=%d)\n', K);
fprintf('nt=1: 1 train/class, nt=2: 2 train/class, nt=3: 3 train/class\n\n');

for ni = 1:length(nt_vals)
    nt = nt_vals(ni);
    fprintf('========== nt=%d ==========\n', nt);

    best_fwd = 0; best_fus = 0; best_lam = [0,0]; best_net = -inf;

    for l1 = 1:length(lambda_grid)
        for l2 = 1:length(lambda_grid)
            lam1 = lambda_grid(l1);
            lam2 = lambda_grid(l2);

            acc_fwd = zeros(nSplits,1); acc_fus = zeros(nSplits,1);
            broken = zeros(nSplits,1); fixed = zeros(nSplits,1);

            for m = 1:nSplits
                tc = cell(K,1); sc = cell(K,1);
                for c = 1:K
                    perm = randperm(number);
                    tc{c} = perm(1:nt);
                    sc{c} = perm(nt+1:end);
                end
                [S, S_label] = build_set(data, tc, number);
                [T, T_label] = build_set(data, sc, number);

                [pred_fusion, ~, out] = DPGSR_Fusion(S, T, S_label, T_label, nt, ...
                    lam1, lam2, 'adaptive');

                [~, pf] = max(out.PF,[],1); fwd_pred = out.classes(pf);
                M = length(T_label);
                acc_fwd(m) = sum(fwd_pred==T_label)/M;
                acc_fus(m) = sum(pred_fusion==T_label)/M;
                broken(m) = sum(fwd_pred==T_label & pred_fusion~=T_label);
                fixed(m)  = sum(fwd_pred~=T_label & pred_fusion==T_label);
            end

            mfwd = mean(acc_fwd); mfus = mean(acc_fus);
            mnet = round(mean(fixed)-mean(broken));
            fprintf('  l1=%.0e l2=%.0e | fus=%.4f fwd=%.4f | brk=%d fix=%d net=%d\n', ...
                lam1, lam2, mfus, mfwd, round(mean(broken)), round(mean(fixed)), mnet);

            if mfus > best_fus
                best_fwd = mfwd; best_fus = mfus; best_lam = [lam1, lam2]; best_net = mnet;
            end
        end
    end

    fprintf('  >>> Best: l1=%.0e l2=%.0e  fus=%.4f fwd=%.4f  net=%d  gap=+.4f\n\n', ...
        best_lam(1), best_lam(2), best_fus, best_fwd, best_net, best_fus-best_fwd);
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

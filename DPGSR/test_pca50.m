% test_pca50.m — PCA-50 维下的 DPGSR 融合
clc; rng(1);
load('pcayaleb50.mat');
data_pca = double(dataset);  % 50 x 2432
load('yaleborigin.mat');
data_raw = double(dataset);  % 2016 x 2432

fprintf('PCA-50: %d dim x %d samples\n', size(data_pca,1), size(data_pca,2));
fprintf('Raw:    %d dim x %d samples\n\n', size(data_raw,1), size(data_raw,2));

K = 15; number = 64; nTrain = 8; nt = nTrain; nSplits = 5;
lambda_grid = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1];

for feat_type = 1:2
    if feat_type == 1
        data = data_raw; label = 'Raw 2016d';
    else
        data = data_pca; label = 'PCA 50d';
    end
    fprintf('=== %s ===\n', label);

    best_fus = -inf; best_fwd = 0; best_lam = [0,0]; best_net = -inf;

    for l1 = 1:length(lambda_grid)
        for l2 = 1:length(lambda_grid)
            lam1 = lambda_grid(l1); lam2 = lambda_grid(l2);
            fwd_acc = zeros(nSplits,1); fus_acc = zeros(nSplits,1);
            rev_acc = zeros(nSplits,1);
            broken = zeros(nSplits,1); fixed = zeros(nSplits,1);

            for m = 1:nSplits
                tc = cell(K,1); sc = cell(K,1);
                for c = 1:K
                    perm = randperm(number);
                    tc{c} = perm(1:nTrain); sc{c} = perm(nTrain+1:end);
                end
                [S, S_label] = build_set(data, tc, number);
                [T, T_label] = build_set(data, sc, number);

                [pred_fusion, ~, out] = DPGSR_Fusion(S, T, S_label, T_label, nt, ...
                    lam1, lam2, 'adaptive');

                [~, pf] = max(out.PF,[],1); [~, pr] = max(out.PR,[],1);
                M = length(T_label);
                fwd_acc(m) = sum(out.classes(pf)==T_label)/M;
                rev_acc(m) = sum(out.classes(pr)==T_label)/M;
                fus_acc(m) = sum(pred_fusion==T_label)/M;
                broken(m) = sum(out.classes(pf)==T_label & pred_fusion~=T_label);
                fixed(m)  = sum(out.classes(pf)~=T_label & pred_fusion==T_label);
            end

            mfwd = mean(fwd_acc); mfus = mean(fus_acc); mrev = mean(rev_acc);
            mnet = round(mean(fixed)-mean(broken));
            fprintf('  l1=%.0e l2=%.0e | fus=%.4f fwd=%.4f rev=%.4f | brk=%d fix=%d net=%d\n', ...
                lam1, lam2, mfus, mfwd, mrev, round(mean(broken)), round(mean(fixed)), mnet);

            if mfus > best_fus
                best_fwd = mfwd; best_fus = mfus; best_lam = [lam1, lam2]; best_net = mnet;
            end
        end
    end
    fprintf('  >>> Best: l1=%.0e l2=%.0e fus=%.4f fwd=%.4f gap=%+.4f net=%d\n\n', ...
        best_lam(1), best_lam(2), best_fus, best_fwd, best_fus-best_fwd, best_net);
end

function [X, label] = build_set(dataset, index_cell, n_per_class)
K = length(index_cell);
X = []; label = [];
for c = 1:K
    n_c = length(index_cell{c});
    for ii = 1:n_c
        col_id = (c-1)*n_per_class + index_cell{c}(ii);
        X = [X, dataset(:, col_id)];
        label = [label, c];
    end
end
end

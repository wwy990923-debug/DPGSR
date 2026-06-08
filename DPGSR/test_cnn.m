% test_cnn.m — ResNet50 深度特征 vs 原始像素
clc;
rng(1);

% 加载 CNN 特征
load('resnet50_features_yaleb.mat');
data_cnn = double(features');  % d x N: 2048 x 2432
fprintf('ResNet50 features: %d dim x %d samples\n', size(data_cnn,1), size(data_cnn,2));

% 加载原始像素（对比基准）
load('yaleborigin.mat');
data_raw = double(dataset);
fprintf('Raw pixels:        %d dim x %d samples\n', size(data_raw,1), size(data_raw,2));

K = 15;
number = 64;
nTrain = 8;
nt = nTrain;
nSplits = 5;
lambda_grid = [1e-2, 1e-1, 1, 10];

% Pre-generate splits
train_cells = cell(nSplits,1); test_cells = cell(nSplits,1);
for m = 1:nSplits
    tc = cell(K,1); sc = cell(K,1);
    for c = 1:K
        perm = randperm(number);
        tc{c} = perm(1:nTrain);
        sc{c} = perm(nTrain+1:end);
    end
    train_cells{m} = tc; test_cells{m} = sc;
end

for feat_type = 1:2
    if feat_type == 1
        data_use = data_raw;
        label = 'Raw Pixels';
    else
        data_use = data_cnn;
        label = 'ResNet50';
    end

    fprintf('\n========================================\n');
    fprintf('  %s  (K=%d, nt=%d, splits=%d)\n', label, K, nt, nSplits);
    fprintf('========================================\n');

    best_fus = -inf; best_fwd = -inf; best_lam = [0,0];

    for l1 = 1:length(lambda_grid)
        for l2 = 1:length(lambda_grid)
            lam1 = lambda_grid(l1);
            lam2 = lambda_grid(l2);
            accs_fwd = zeros(nSplits,1);
            accs_rev = zeros(nSplits,1);
            accs_fus = zeros(nSplits,1);
            brokens  = zeros(nSplits,1);
            fixeds   = zeros(nSplits,1);

            for m = 1:nSplits
                [S, S_label] = build_set(data_use, train_cells{m}, number);
                [T, T_label] = build_set(data_use, test_cells{m}, number);

                [pred_fusion, ~, out] = DPGSR_Fusion(S, T, S_label, T_label, nt, ...
                    lam1, lam2, 'adaptive');

                [~, pf] = max(out.PF,[],1); fwd_pred = out.classes(pf);
                [~, pr] = max(out.PR,[],1); rev_pred = out.classes(pr);
                M = length(T_label);

                accs_fwd(m) = sum(fwd_pred==T_label)/M;
                accs_rev(m) = sum(rev_pred==T_label)/M;
                accs_fus(m) = sum(pred_fusion==T_label)/M;
                brokens(m)  = sum(fwd_pred==T_label & pred_fusion~=T_label);
                fixeds(m)   = sum(fwd_pred~=T_label & pred_fusion==T_label);
            end

            mfwd = mean(accs_fwd); mfus = mean(accs_fus);
            fprintf('  l1=%.0e l2=%.0e | fus=%.4f fwd=%.4f rev=%.4f | brk=%d fix=%d net=%d\n', ...
                lam1, lam2, mfus, mfwd, mean(accs_rev), ...
                round(mean(brokens)), round(mean(fixeds)), ...
                round(mean(fixeds)-mean(brokens)));

            if mfus > best_fus
                best_fus = mfus; best_fwd = mfwd; best_lam = [lam1, lam2];
            end
        end
    end

    fprintf('  >>> Best: l1=%.0e l2=%.0e fus=%.4f fwd=%.4f (gap=+.4f)\n', ...
        best_lam(1), best_lam(2), best_fus, best_fwd, best_fus - best_fwd);
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

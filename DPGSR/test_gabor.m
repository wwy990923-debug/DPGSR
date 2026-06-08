% test_gabor.m — Gabor特征 vs 原始像素 对比
clc;
rng(1);
load('yaleborigin.mat');
data_raw = dataset;

% 确定图像尺寸（Yale B 通常是 192x168 或 32x32）
d = size(data_raw, 1);
img_h = round(sqrt(d * 168/192));  % 估计
img_w = round(d / img_h);
while img_h * img_w ~= d
    img_h = img_h - 1;
    img_w = d / img_h;
    if img_h <= 1, img_h = 32; img_w = 32; break; end
end
fprintf('Image size: %d x %d = %d\n', img_h, img_w, d);

K = 15;
number = 64;
nTrain = 8;
nt = nTrain;
nSplits = 3;
lambda1 = 1e-2;
lambda2 = 1e-2;

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

fprintf('\n=== Extracting Gabor features (this may take a while)... ===\n');
tic;
data_gabor = extract_gabor_features(data_raw, img_h, img_w);
t_gabor = toc;
fprintf('Extraction time: %.1fs, new dim: %d\n', t_gabor, size(data_gabor,1));

fprintf('\n%-20s %8s %8s %8s %8s %8s\n', 'Features', 'FwdAcc', 'RevAcc', 'FusAcc', 'Broken', 'Fixed');
fprintf('%-20s %8s %8s %8s %8s %8s\n', '--------', '------', '------', '------', '------', '-----');

for feat_type = 1:2
    if feat_type == 1
        data_use = data_raw;
        label = 'Raw Pixels';
    else
        data_use = data_gabor;
        label = 'Gabor';
    end

    all_fwd = []; all_rev = []; all_fus = [];
    all_broken = []; all_fixed = [];

    for m = 1:nSplits
        [S, S_label] = build_set(data_use, train_cells{m}, number);
        [T, T_label] = build_set(data_use, test_cells{m}, number);

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

    fprintf('%-20s %8.4f %8.4f %8.4f %8d %8d\n', ...
        label, mean(all_fwd), mean(all_rev), mean(all_fus), ...
        round(mean(all_broken)), round(mean(all_fixed)));
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

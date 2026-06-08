% test_full.m - Full K=38 test with new S_G metric
clc;
rng(1);
load('yaleborigin.mat');
data = dataset;

K = 38;
number = 64;
nTrain = 8;
nt = nTrain;
nmax = 5;  % random splits
lambda_grid = [1e-2, 1e-1, 1, 10];

fprintf('============================================\n');
fprintf('DP-GSR Full Test (K=%d, train/class=%d, splits=%d)\n', K, nTrain, nmax);
fprintf('Alpha: pred-agreement + sigmoid(r0=%.2f,k=%d) + PR-gate(r0=%.3f,k=%d)\n', 1.45, 15, 0.055, 100);
fprintf('============================================\n\n');

% Pre-generate splits
train_cells = cell(nmax, 1);
test_cells = cell(nmax, 1);
for m = 1:nmax
    tc = cell(K,1); sc = cell(K,1);
    for c = 1:K
        perm = randperm(number);
        tc{c} = perm(1:nTrain);
        sc{c} = perm(nTrain+1:end);
    end
    train_cells{m} = tc;
    test_cells{m} = sc;
end

% Joint solver results
fprintf('=== Joint solver sweep ===\n');
best_acc_j = -inf;
best_g_j = [0, 0];
for l1 = 1:length(lambda_grid)
    for l2 = 1:length(lambda_grid)
        g1 = lambda_grid(l1);
        g2 = lambda_grid(l2);
        accs = zeros(nmax, 1);
        accs_fwd = zeros(nmax, 1);
        brokens = zeros(nmax, 1);
        fixeds = zeros(nmax, 1);
        alphas = [];
        times = zeros(nmax, 1);

        for m = 1:nmax
            tic;
            [S, S_label] = build_set_by_class(data, train_cells{m}, number);
            [T, T_label] = build_set_by_class(data, test_cells{m}, number);
            [pred_fusion, acc, out] = DPGSR_Fusion(S, T, S_label, T_label, nt, g1, g2, 'adaptive', [], 'joint');
            times(m) = toc;
            accs(m) = acc;
            alphas = [alphas, out.alpha_j_true];

            [~, pred_fwd] = max(out.PF, [], 1);
            fwd_pred = out.classes(pred_fwd);
            accs_fwd(m) = sum(fwd_pred == T_label) / length(T_label);
            brokens(m) = sum(fwd_pred == T_label & pred_fusion ~= T_label);
            fixeds(m)  = sum(fwd_pred ~= T_label & pred_fusion == T_label);
        end

        avg_acc = mean(accs);
        avg_fwd = mean(accs_fwd);
        avg_time = mean(times);
        fprintf('  g1=%.0e, g2=%.0e: fus=%.4f fwd=%.4f (std=%.4f) | brk=%d fix=%d net=%d | a=%.3f t=%.1fs\n', ...
            g1, g2, avg_acc, avg_fwd, std(accs), round(mean(brokens)), round(mean(fixeds)), ...
            round(mean(fixeds)-mean(brokens)), mean(alphas), avg_time);

        if avg_acc > best_acc_j
            best_acc_j = avg_acc;
            best_g_j = [g1, g2];
        end
    end
end

% Separate solver results
fprintf('\n=== Separate solver sweep ===\n');
best_acc_s = -inf;
best_l_s = [0, 0];
for l1 = 1:length(lambda_grid)
    for l2 = 1:length(lambda_grid)
        lam1 = lambda_grid(l1);
        lam2 = lambda_grid(l2);
        accs = zeros(nmax, 1);
        accs_fwd = zeros(nmax, 1);
        brokens = zeros(nmax, 1);
        fixeds = zeros(nmax, 1);
        alphas = [];
        times = zeros(nmax, 1);

        for m = 1:nmax
            tic;
            [S, S_label] = build_set_by_class(data, train_cells{m}, number);
            [T, T_label] = build_set_by_class(data, test_cells{m}, number);
            [pred_fusion, acc, out] = DPGSR_Fusion(S, T, S_label, T_label, nt, lam1, lam2, 'adaptive', [], 'separate');
            times(m) = toc;
            accs(m) = acc;
            alphas = [alphas, out.alpha_j_true];

            [~, pred_fwd] = max(out.PF, [], 1);
            fwd_pred = out.classes(pred_fwd);
            accs_fwd(m) = sum(fwd_pred == T_label) / length(T_label);
            brokens(m) = sum(fwd_pred == T_label & pred_fusion ~= T_label);
            fixeds(m)  = sum(fwd_pred ~= T_label & pred_fusion == T_label);
        end

        avg_acc = mean(accs);
        avg_fwd = mean(accs_fwd);
        avg_time = mean(times);
        fprintf('  l1=%.0e, l2=%.0e: fus=%.4f fwd=%.4f (std=%.4f) | brk=%d fix=%d net=%d | a=%.3f t=%.1fs\n', ...
            lam1, lam2, avg_acc, avg_fwd, std(accs), round(mean(brokens)), round(mean(fixeds)), ...
            round(mean(fixeds)-mean(brokens)), mean(alphas), avg_time);

        if avg_acc > best_acc_s
            best_acc_s = avg_acc;
            best_l_s = [lam1, lam2];
        end
    end
end

fprintf('\n============================================\n');
fprintf('Best Joint:    g1=%.0e, g2=%.0e, fus=%.4f\n', best_g_j(1), best_g_j(2), best_acc_j);
fprintf('Best Separate: l1=%.0e, l2=%.0e, fus=%.4f\n', best_l_s(1), best_l_s(2), best_acc_s);
fprintf('============================================\n');

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

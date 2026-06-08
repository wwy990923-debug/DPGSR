% diagnose_reverse.m
% 全面诊断逆向路径：lambda2 调参 + PR 公式对比 + D 矩阵分析

clc;
rng(1);
load('yaleborigin.mat');
data = dataset;

K = 15;
number = 64;
nTrain = 8;
nt = nTrain;
nSplits = 5;
lambda1 = 1e-2;  % fixed forward

fprintf('=============================================================\n');
fprintf('Reverse Path Diagnosis (K=%d, splits=%d)\n', K, nSplits);
fprintf('=============================================================\n\n');

%% ============================================================
% Part 1: Sweep lambda2, test multiple PR formulas
%% ============================================================

lambda2_grid = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 10, 100];

% We'll test 6 different PR formulas
n_formulas = 7;
formula_names = {
    '1: SR = ||D(j,idx)|| / (sqrt(nk)*||Sk||_F)  [current]';
    '2: SR = ||D(j,idx)|| / ||D(j,:)||             [relative]';
    '3: SR = ||D(j,idx)||                           [raw norm]';
    '4: SR = sum(D(j,idx).^2)                       [energy]';
    '5: reverse residual (like forward)';
    '6: SR = ||D(j,idx)|| / sqrt(nk)               [per-sample]';
    '7: SR = ||D(j,idx)||_1                         [L1 norm]';
};

fprintf('Part 1: Sweeping lambda2 with %d PR formulas\n', n_formulas);
fprintf('Lambda1 fixed at %.0e\n\n', lambda1);

best_by_formula = -inf * ones(1, n_formulas);
best_l2_by_formula = zeros(1, n_formulas);

% Pre-generate splits
splits_data = cell(nSplits, 1);
for m = 1:nSplits
    train_cell = cell(K,1); test_cell = cell(K,1);
    for c = 1:K
        perm = randperm(number);
        train_cell{c} = perm(1:nTrain);
        test_cell{c} = perm(nTrain+1:end);
    end
    splits_data{m} = {train_cell, test_cell};
end

for li = 1:length(lambda2_grid)
    l2 = lambda2_grid(li);
    fprintf('lambda2=%.0e: ', l2);

    acc_formulas = zeros(1, n_formulas);
    total = 0;

    for m = 1:nSplits
        tc = splits_data{m}{1};
        sc = splits_data{m}{2};
        [S, S_label] = build_set(data, tc, number);
        [T, T_label] = build_set(data, sc, number);

        [pred_fusion, ~, out] = DPGSR_Fusion(S, T, S_label, T_label, nt, ...
            lambda1, l2, 'adaptive');
        [~, pred_fwd] = max(out.PF, [], 1);
        pred_fwd = out.classes(pred_fwd);

        D = out.D;
        M = size(T,2);
        classes = out.classes;
        total = total + M;

        for j = 1:M
            dj = D(j, :);  % 1 x N, coefficients for test sample j

            for kk = 1:K
                cls = classes(kk);
                idx = (S_label == cls);
                nk = sum(idx);
                Ak_norm = norm(S(:, idx), 'fro');
                d_jk = dj(idx);  % coefficients for class k

                % F1: current formula
                SR(1, kk) = norm(d_jk, 2) / (sqrt(nk) * Ak_norm + 1e-12);

                % F2: relative activation
                SR(2, kk) = norm(d_jk, 2) / (norm(dj, 2) + 1e-12);

                % F3: raw norm
                SR(3, kk) = norm(d_jk, 2);

                % F4: energy
                SR(4, kk) = sum(d_jk.^2);

                % F5: reverse residual (computed outside loop)
                % Will compute after

                % F6: per-sample normalized
                SR(6, kk) = norm(d_jk, 2) / (sqrt(nk) + 1e-12);

                % F7: L1 norm
                SR(7, kk) = norm(d_jk, 1);
            end

            % F5: reverse residual
            for kk = 1:K
                cls = classes(kk);
                idx = (S_label == cls);
                % Reconstruct class k training samples from T
                res = norm(S(:, idx) - T * D(:, idx), 'fro')^2;
                SR(5, kk) = -res;  % negative residual (higher = better)
            end

            % Evaluate each formula
            for fi = 1:n_formulas
                sr = SR(fi, :);
                if fi == 5
                    % For residual-based, negative residual
                    [~, pred] = max(sr);
                else
                    pr_f = sr / (sum(sr) + 1e-12);
                    [~, pred] = max(pr_f);
                end
                if classes(pred) == T_label(j)
                    acc_formulas(fi) = acc_formulas(fi) + 1;
                end
            end
        end
    end

    for fi = 1:n_formulas
        acc = acc_formulas(fi) / total;
        fprintf('%s=%.3f  ', formula_names{fi}(1:2), acc);
        if acc > best_by_formula(fi)
            best_by_formula(fi) = acc;
            best_l2_by_formula(fi) = l2;
        end
    end
    fprintf('\n');
end

fprintf('\n--- Best per formula ---\n');
fprintf('%-55s %8s %10s\n', 'Formula', 'lambda2', 'Acc');
fprintf('%-55s %8s %10s\n', '-------', '-------', '---');
for fi = 1:n_formulas
    fprintf('%-55s %8.0e %10.4f\n', formula_names{fi}, best_l2_by_formula(fi), best_by_formula(fi));
end

%% ============================================================
% Part 2: Best reverse vs forward vs fusion
%% ============================================================
fprintf('\n========================================\n');
fprintf('Part 2: Best reverse vs forward accuracy\n');
fprintf('========================================\n\n');

% Find overall best reverse formula + lambda2
[best_rev_acc, best_fi] = max(best_by_formula);
best_l2 = best_l2_by_formula(best_fi);

fprintf('Best reverse: %s\n', formula_names{best_fi});
fprintf('lambda2 = %.0e, accuracy = %.4f\n\n', best_l2, best_rev_acc);

% Run one more time with aligned analysis
rng(42);
train_cell = cell(K,1); test_cell = cell(K,1);
for c = 1:K
    perm = randperm(number);
    train_cell{c} = perm(1:nTrain);
    test_cell{c} = perm(nTrain+1:end);
end
[S, S_label] = build_set(data, train_cell, number);
[T, T_label] = build_set(data, test_cell, number);

[~, ~, out] = DPGSR_Fusion(S, T, S_label, T_label, nt, lambda1, best_l2, 'adaptive');

[~, pred_fwd] = max(out.PF, [], 1); pred_fwd = out.classes(pred_fwd);
[~, pred_rev] = max(out.PR, [], 1); pred_rev = out.classes(pred_rev);
[~, pred_fus] = max(out.Score, [], 1); pred_fus = out.classes(pred_fus);

M = size(T,2);

% Compute best reverse prediction
D = out.D;
best_pred_rev = zeros(1, M);
for j = 1:M
    dj = D(j, :);
    SR_best = zeros(1, K);
    for kk = 1:K
        cls = out.classes(kk);
        idx = (S_label == cls);
        nk = sum(idx);
        d_jk = dj(idx);
        Ak_norm = norm(S(:, idx), 'fro');

        switch best_fi
            case 1; SR_best(kk) = norm(d_jk,2) / (sqrt(nk)*Ak_norm + 1e-12);
            case 2; SR_best(kk) = norm(d_jk,2) / (norm(dj,2) + 1e-12);
            case 3; SR_best(kk) = norm(d_jk,2);
            case 4; SR_best(kk) = sum(d_jk.^2);
            case 5
                res = norm(S(:,idx) - T*D(:,idx), 'fro')^2;
                SR_best(kk) = -res;
            case 6; SR_best(kk) = norm(d_jk,2) / (sqrt(nk) + 1e-12);
            case 7; SR_best(kk) = norm(d_jk,1);
        end
    end
    if best_fi == 5
        [~, pred] = max(SR_best);
    else
        pr_best = SR_best / (sum(SR_best) + 1e-12);
        [~, pred] = max(pr_best);
    end
    best_pred_rev(j) = out.classes(pred);
end

acc_best_rev = sum(best_pred_rev == T_label) / M;
acc_fwd = sum(pred_fwd == T_label) / M;
acc_orig_rev = sum(pred_rev == T_label) / M;
acc_fusion = sum(pred_fus == T_label) / M;

fprintf('Pure Forward        : %.4f\n', acc_fwd);
fprintf('Orig Rev (current)  : %.4f\n', acc_orig_rev);
fprintf('Best Rev (F%d)       : %.4f\n', best_fi, acc_best_rev);
fprintf('Adapt Fusion        : %.4f\n', acc_fusion);

%% ============================================================
% Part 3: Agreement analysis with BEST reverse
%% ============================================================
fprintf('\n--- Agreement analysis with best reverse ---\n');

agree = (pred_fwd == best_pred_rev);
disagree = ~agree;

fprintf('Agree:   %d (%.1f%%), acc=%.4f\n', sum(agree), sum(agree)/M*100, ...
    sum(pred_fwd(agree) == T_label(agree)) / sum(agree));
fprintf('Disagree: %d (%.1f%%)\n', sum(disagree), sum(disagree)/M*100);
fprintf('  When disagree: fwd acc=%.4f,  rev acc=%.4f\n', ...
    sum(pred_fwd(disagree) == T_label(disagree)) / sum(disagree), ...
    sum(best_pred_rev(disagree) == T_label(disagree)) / sum(disagree));

%% ============================================================
% Part 4: With best reverse, test fusion strategies
%% ============================================================
fprintf('\n--- Fusion with BEST reverse ---\n');

% Strategy A: Agreement → output agreed class; Disagreement → forward
acc_agree_fwd = 0;
for j = 1:M
    if pred_fwd(j) == best_pred_rev(j)
        pred = pred_fwd(j);
    else
        pred = pred_fwd(j);
    end
    if pred == T_label(j); acc_agree_fwd = acc_agree_fwd + 1; end
end
fprintf('Agree→agree, Disagree→fwd: acc=%.4f\n', acc_agree_fwd/M);

% Strategy B: Agreement → agreed; Disagreement → PF margin decides
PF = out.PF;
pf_margin = zeros(1, M);
rev_margin = zeros(1, M);
for j = 1:M
    pf_s = sort(PF(:,j), 'descend');
    pf_margin(j) = pf_s(1) - pf_s(2);
    % rev margin using best formula
    sr = zeros(1, K);
    dj = D(j, :);
    for kk = 1:K
        cls = out.classes(kk);
        idx = (S_label == cls);
        nk = sum(idx);
        d_jk = dj(idx);
        Ak_norm = norm(S(:, idx), 'fro');
        if best_fi == 1; sr(kk) = norm(d_jk,2) / (sqrt(nk)*Ak_norm + 1e-12);
        elseif best_fi == 2; sr(kk) = norm(d_jk,2) / (norm(dj,2) + 1e-12);
        elseif best_fi == 3; sr(kk) = norm(d_jk,2);
        elseif best_fi == 4; sr(kk) = sum(d_jk.^2);
        elseif best_fi == 6; sr(kk) = norm(d_jk,2) / (sqrt(nk) + 1e-12);
        elseif best_fi == 7; sr(kk) = norm(d_jk,1);
        end
    end
    if best_fi ~= 5
        sr = sr / (sum(sr) + 1e-12);
    end
    sr_s = sort(sr, 'descend');
    rev_margin(j) = sr_s(1) - sr_s(2);
end

% Strategy C: PF margin gating
acc_margin_gate = 0; n_fwd_gate = 0; n_rev_gate = 0;
for T_gate = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    correct = 0; n_fwd = 0; n_rev = 0;
    for j = 1:M
        if pf_margin(j) > T_gate
            pred = pred_fwd(j);
            n_fwd = n_fwd + 1;
        else
            pred = best_pred_rev(j);
            n_rev = n_rev + 1;
        end
        if pred == T_label(j); correct = correct + 1; end
    end
    fprintf('Gate T=%.2f: acc=%.4f (fwd=%.0f%%, rev=%.0f%%)\n', ...
        T_gate, correct/M, n_fwd/M*100, n_rev/M*100);
end

%% ============================================================
% Part 5: Analyze D matrix sparsity per class
%% ============================================================
fprintf('\n--- D matrix structure analysis ---\n');
fprintf('Class activations for first 3 test samples:\n');
for j = 1:min(3, M)
    dj = D(j, :);
    fprintf('Sample %d (true=%d, fwd=%d): ', j, T_label(j), pred_fwd(j));
    for kk = 1:K
        idx = (S_label == kk);
        fprintf('c%d=%.2f ', kk, norm(dj(idx),2));
    end
    fprintf('\n');
end

%% ============================================================
% Helper
%% ============================================================
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

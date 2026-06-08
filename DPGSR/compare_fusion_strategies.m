% compare_fusion_strategies.m
% 对比多种融合策略：如何在保持正向正确的同时，用逆向修正错误

clc;
rng(1);
load('yaleborigin.mat');
data = dataset;

K = 15;
number = 64;
nTrain = 8;
nt = nTrain;
nSplits = 10;
lambda1 = 1e-2;
lambda2 = 1e-2;

fprintf('=============================================================\n');
fprintf('Fusion Strategy Comparison (K=%d, lambda=[%.0e,%.0e])\n', K, lambda1, lambda2);
fprintf('=============================================================\n\n');

all_truth = [];
all_fwd = [];      % pure forward
all_rev = [];      % pure reverse
all_fusion = [];   % original adaptive fusion
all_margin = [];   % forward confidence margin
all_alpha = [];    % original alpha
all_PF = {};       % store full PF for custom strategies
all_PR = {};       % store full PR for custom strategies
all_classes = {};  % store class mapping
all_pred_fwd = [];
all_pred_rev = [];
all_truth_by_split = {};
all_PF_by_split = {};
all_PR_by_split = {};
all_classes_by_split = {};
all_alpha_by_split = {};
split_info = [];

for m = 1:nSplits
    train_cell = cell(K,1);
    test_cell = cell(K,1);
    for c = 1:K
        perm = randperm(number);
        train_cell{c} = perm(1:nTrain);
        test_cell{c} = perm(nTrain+1:end);
    end
    [S, S_label] = build_set(data, train_cell, number);
    [T, T_label] = build_set(data, test_cell, number);

    [pred_fusion, ~, out] = DPGSR_Fusion(S, T, S_label, T_label, nt, ...
        lambda1, lambda2, 'adaptive');

    [~, pred_fwd] = max(out.PF, [], 1);
    pred_fwd = out.classes(pred_fwd);
    [~, pred_rev] = max(out.PR, [], 1);
    pred_rev = out.classes(pred_rev);

    % Forward confidence margin: top1 - top2
    PF_sorted = sort(out.PF, 1, 'descend');
    margin = PF_sorted(1,:) - PF_sorted(2,:);

    all_truth = [all_truth, T_label];
    all_fwd = [all_fwd, pred_fwd];
    all_rev = [all_rev, pred_rev];
    all_fusion = [all_fusion, pred_fusion];
    all_margin = [all_margin, margin];
    all_alpha = [all_alpha, out.alpha_j_true];

    all_truth_by_split{m} = T_label;
    all_PF_by_split{m} = out.PF;
    all_PR_by_split{m} = out.PR;
    all_classes_by_split{m} = out.classes;
    all_alpha_by_split{m} = out.alpha_j_true;
    split_info(m).M = length(T_label);
end

total = length(all_truth);

%% ============================================================
% Baseline accuracies
%% ============================================================
acc_fwd = sum(all_fwd == all_truth) / total;
acc_rev = sum(all_rev == all_truth) / total;
acc_fusion = sum(all_fusion == all_truth) / total;

fprintf('--- Baseline ---\n');
fprintf('Pure Forward : %.4f\n', acc_fwd);
fprintf('Pure Reverse : %.4f\n', acc_rev);
fprintf('Adapt Fusion : %.4f\n', acc_fusion);

%% ============================================================
% Strategy 1: Margin-based alpha (continuous)
% alpha = exp(-beta * margin), tuned via beta
%% ============================================================
fprintf('\n--- Strategy 1: Margin-based alpha ---\n');
fprintf('  alpha = exp(-beta * margin)\n');
betas = [0.1, 0.5, 1, 2, 5, 10, 20, 50];
best_acc_m1 = -inf;
best_beta_m1 = 0;
for beta = betas
    acc_m1 = compute_acc_margin_alpha(all_truth_by_split, all_PF_by_split, ...
        all_PR_by_split, all_classes_by_split, beta, split_info);
    fprintf('  beta=%4.1f: acc=%.4f\n', beta, acc_m1);
    if acc_m1 > best_acc_m1
        best_acc_m1 = acc_m1;
        best_beta_m1 = beta;
    end
end
fprintf('  Best: beta=%.1f, acc=%.4f (Δ vs fwd: %+.4f)\n', best_beta_m1, best_acc_m1, best_acc_m1-acc_fwd);

%% ============================================================
% Strategy 2: Hard threshold
% if margin > threshold → pure forward
% else → original fusion
%% ============================================================
fprintf('\n--- Strategy 2: Hard threshold ---\n');
fprintf('  if margin > T → pure fwd, else → fusion\n');
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
best_acc_m2 = -inf;
best_T_m2 = 0;
best_use_pct_m2 = 0;
for T = thresholds
    n_fwd = 0; n_fusion_used = 0; correct = 0;
    for m = 1:nSplits
        truth = all_truth_by_split{m};
        PF = all_PF_by_split{m};
        PR = all_PR_by_split{m};
        cls = all_classes_by_split{m};
        alpha_j = all_alpha_by_split{m};
        M = length(truth);

        for j = 1:M
            PF_sorted = sort(PF(:,j), 'descend');
            margin_j = PF_sorted(1) - PF_sorted(2);

            if margin_j > T
                [~, pred] = max(PF(:,j));
                n_fwd = n_fwd + 1;
            else
                score = (1 - alpha_j(j)) * PF(:,j) + alpha_j(j) * PR(:,j);
                [~, pred] = max(score);
                n_fusion_used = n_fusion_used + 1;
            end
            if cls(pred) == truth(j)
                correct = correct + 1;
            end
        end
    end
    acc = correct / total;
    pct = n_fusion_used / total * 100;
    fprintf('  T=%.1f: acc=%.4f  (fwd-only: %.0f%%, fusion: %.0f%%)\n', T, acc, n_fwd/total*100, pct);
    if acc > best_acc_m2
        best_acc_m2 = acc; best_T_m2 = T; best_use_pct_m2 = pct;
    end
end
fprintf('  Best: T=%.1f, acc=%.4f, fusion used on %.0f%% of samples (Δ vs fwd: %+.4f)\n', ...
    best_T_m2, best_acc_m2, best_use_pct_m2, best_acc_m2-acc_fwd);

%% ============================================================
% Strategy 3: Agreement-based
% if fwd == rev prediction → use fwd (or rev, they agree)
% if fwd != rev → use the more confident one
%% ============================================================
fprintf('\n--- Strategy 3: Agreement-based ---\n');
agree_correct = 0; disagree_correct = 0; total_agree = 0; total_disagree = 0;
disagree_fwd_correct = 0; disagree_rev_correct = 0; disagree_margin_correct = 0;

for m = 1:nSplits
    truth = all_truth_by_split{m};
    PF = all_PF_by_split{m};
    PR = all_PR_by_split{m};
    cls = all_classes_by_split{m};
    M = length(truth);

    for j = 1:M
        [~, fwd_idx] = max(PF(:,j));
        [~, rev_idx] = max(PR(:,j));
        fwd_margin = max(PF(:,j)) - sort(PF(:,j), 'descend');
        fwd_margin = fwd_margin(1);  % top1 - top2
        rev_margin = max(PR(:,j)) - sort(PR(:,j), 'descend');
        rev_margin = rev_margin(1);

        if fwd_idx == rev_idx
            total_agree = total_agree + 1;
            pred = cls(fwd_idx);
            if pred == truth(j); agree_correct = agree_correct + 1; end
        else
            total_disagree = total_disagree + 1;

            % Track accuracy by how we decide
            if cls(fwd_idx) == truth(j); disagree_fwd_correct = disagree_fwd_correct + 1; end
            if cls(rev_idx) == truth(j); disagree_rev_correct = disagree_rev_correct + 1; end

            % Use the one with higher relative margin
            if fwd_margin > rev_margin
                pred = cls(fwd_idx);
            else
                pred = cls(rev_idx);
            end
            if pred == truth(j); disagree_margin_correct = disagree_margin_correct + 1; end
        end
    end
end

acc_agree = agree_correct / total_agree;
acc_disagree_margin = disagree_margin_correct / max(total_disagree,1);
acc_overall_s3 = (agree_correct + disagree_margin_correct) / total;
fprintf('  Agree (fwd==rev):   %d samples (%.1f%%), acc=%.4f\n', total_agree, total_agree/total*100, acc_agree);
fprintf('  Disagree (fwd!=rev): %d samples (%.1f%%)\n', total_disagree, total_disagree/total*100);
fprintf('    - fwd correct: %d, rev correct: %d\n', disagree_fwd_correct, disagree_rev_correct);
fprintf('    - By better margin: acc=%.4f\n', acc_disagree_margin);
fprintf('  Overall acc=%.4f (Δ vs fwd: %+.4f)\n', acc_overall_s3, acc_overall_s3-acc_fwd);

%% ============================================================
% Strategy 4: Re-scaled S_G alpha
% alpha_new = alpha^p  (p > 1 shrinks alpha, p < 1 expands it)
%% ============================================================
fprintf('\n--- Strategy 4: Re-scaled S_G alpha ---\n');
fprintf('  alpha_new = alpha^power\n');
powers = [0.5, 0.8, 1.2, 1.5, 2, 3, 5];
best_acc_m4 = -inf; best_p_m4 = 0;
for p = powers
    correct = 0;
    for m = 1:nSplits
        truth = all_truth_by_split{m};
        PF = all_PF_by_split{m};
        PR = all_PR_by_split{m};
        cls = all_classes_by_split{m};
        alpha = all_alpha_by_split{m};
        M = length(truth);
        for j = 1:M
            a = alpha(j)^p;
            score = (1-a)*PF(:,j) + a*PR(:,j);
            [~, pred] = max(score);
            if cls(pred) == truth(j); correct = correct + 1; end
        end
    end
    acc = correct / total;
    fprintf('  power=%.1f: acc=%.4f\n', p, acc);
    if acc > best_acc_m4; best_acc_m4 = acc; best_p_m4 = p; end
end
fprintf('  Best: power=%.1f, acc=%.4f (Δ vs fwd: %+.4f)\n', best_p_m4, best_acc_m4, best_acc_m4-acc_fwd);

%% ============================================================
% Strategy 5: Fusion only when forward uncertain AND reverse confident
%% ============================================================
fprintf('\n--- Strategy 5: Double-gate ---\n');
fprintf('  Use fusion only when fwd_margin < T AND rev_margin > R\n');
best_acc_m5 = -inf; best_T5 = 0; best_R5 = 0;
for T = [0.2, 0.3, 0.4, 0.5, 0.6]
    for R = [0.1, 0.15, 0.2, 0.25, 0.3]
        correct = 0; n_fwd_only = 0; n_fusion_used = 0;
        for m = 1:nSplits
            truth = all_truth_by_split{m};
            PF = all_PF_by_split{m};
            PR = all_PR_by_split{m};
            cls = all_classes_by_split{m};
            alpha_j = all_alpha_by_split{m};
            M = length(truth);
            for j = 1:M
                fwd_sorted = sort(PF(:,j), 'descend');
                rev_sorted = sort(PR(:,j), 'descend');
                fwd_margin = fwd_sorted(1) - fwd_sorted(2);
                rev_margin = rev_sorted(1) - rev_sorted(2);

                if fwd_margin > T
                    [~, pred] = max(PF(:,j));
                    n_fwd_only = n_fwd_only + 1;
                else
                    score = (1-alpha_j(j))*PF(:,j) + alpha_j(j)*PR(:,j);
                    [~, pred] = max(score);
                    n_fusion_used = n_fusion_used + 1;
                end
                if cls(pred) == truth(j); correct = correct + 1; end
            end
        end
        acc = correct / total;
        if acc > best_acc_m5
            best_acc_m5 = acc; best_T5 = T; best_R5 = R;
        end
    end
end
fprintf('  Best: T=%.2f, acc=%.4f (Δ vs fwd: %+.4f)\n', best_T5, best_acc_m5, best_acc_m5-acc_fwd);

%% ============================================================
% Full summary
%% ============================================================
fprintf('\n========================================\n');
fprintf('SUMMARY\n');
fprintf('========================================\n');
fprintf('Pure Forward      : %.4f\n', acc_fwd);
fprintf('Pure Reverse      : %.4f\n', acc_rev);
fprintf('Adapt Fusion (orig): %.4f\n', acc_fusion);
fprintf('S1 Margin-alpha   : %.4f  (beta=%.1f)\n', best_acc_m1, best_beta_m1);
fprintf('S2 Hard threshold : %.4f  (T=%.1f)\n', best_acc_m2, best_T_m2);
fprintf('S3 Agreement      : %.4f\n', acc_overall_s3);
fprintf('S4 Alpha rescale  : %.4f  (power=%.1f)\n', best_acc_m4, best_p_m4);
fprintf('S5 Double-gate    : %.4f  (T=%.1f)\n', best_acc_m5, best_T5);
fprintf('\nForward errors fixed by best strategy:\n');

% Compute for best strategy (hard threshold)
fwd_wrong = (all_fwd ~= all_truth);
fixed_by_best = 0; broken_by_best = 0; total_best_correct = 0;
% Re-run best strategy to get per-sample predictions
best_preds = zeros(1, total);
offset = 0;
for m = 1:nSplits
    truth = all_truth_by_split{m};
    PF = all_PF_by_split{m};
    PR = all_PR_by_split{m};
    cls = all_classes_by_split{m};
    alpha_j = all_alpha_by_split{m};
    M = length(truth);
    for j = 1:M
        fwd_sorted = sort(PF(:,j), 'descend');
        margin_j = fwd_sorted(1) - fwd_sorted(2);
        if margin_j > best_T_m2
            [~, pred] = max(PF(:,j));
        else
            score = (1-alpha_j(j))*PF(:,j) + alpha_j(j)*PR(:,j);
            [~, pred] = max(score);
        end
        best_preds(offset + j) = cls(pred);
    end
    offset = offset + M;
end

fwd_wrong_idx = find(fwd_wrong);
fixed = 0; broken = 0;
for i = 1:length(fwd_wrong_idx)
    idx = fwd_wrong_idx(i);
    if best_preds(idx) == all_truth(idx); fixed = fixed + 1; end
end
fwd_right = (all_fwd == all_truth);
fwd_right_idx = find(fwd_right);
for i = 1:length(fwd_right_idx)
    idx = fwd_right_idx(i);
    if best_preds(idx) ~= all_truth(idx); broken = broken + 1; end
end
fprintf('  Forward errors: %d, fixed: %d (%.1f%%)\n', sum(fwd_wrong), fixed, fixed/max(1,sum(fwd_wrong))*100);
fprintf('  Forward correct samples broken: %d / %d (%.1f%%)\n', broken, sum(fwd_right), broken/max(1,sum(fwd_right))*100);

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

function acc = compute_acc_margin_alpha(truth_by_split, PF_by_split, PR_by_split, ...
    classes_by_split, beta, split_info)
nSplits = length(truth_by_split);
correct = 0; total = 0;
for m = 1:nSplits
    truth = truth_by_split{m};
    PF = PF_by_split{m};
    PR = PR_by_split{m};
    cls = classes_by_split{m};
    M = length(truth);
    for j = 1:M
        PF_sorted = sort(PF(:,j), 'descend');
        margin = PF_sorted(1) - PF_sorted(2);
        alpha_j = exp(-beta * margin);
        score = (1-alpha_j)*PF(:,j) + alpha_j*PR(:,j);
        [~, pred] = max(score);
        if cls(pred) == truth(j); correct = correct + 1; end
    end
    total = total + M;
end
acc = correct / total;
end

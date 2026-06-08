% sweep_sigmoid.m — Pareto sweep of sigmoid parameters (k, r0)
clc;
rng(1);
load('yaleborigin.mat');
data = dataset;

K = 15;
number = 64;
nTrain = 8;
nt = nTrain;
nSplits = 3;  % fewer splits for speed
lambda1 = 1e-2;
lambda2 = 1e-2;

% Pre-generate splits
train_cells = cell(nSplits, 1);
test_cells = cell(nSplits, 1);
for m = 1:nSplits
    tc = cell(K,1); sc = cell(K,1);
    for c = 1:K
        perm = randperm(number);
        tc{c} = perm(1:nTrain);
        sc{c} = perm(nTrain+1:end);
    end
    train_cells{m} = tc;
    test_cells{m} = sc;
end

% Quick test: PR margin gate with default params
k_vals  = 15;
r0_vals = [1.45, 1.50, 1.55];

nK = length(k_vals);
nR = length(r0_vals);

results_acc   = zeros(nK, nR);
results_broken = zeros(nK, nR);
results_fixed  = zeros(nK, nR);
results_net    = zeros(nK, nR);
results_alpha_hi = zeros(nK, nR);  % alpha for fwdX+revOK
results_alpha_lo = zeros(nK, nR);  % alpha for fwdOK+revX

fprintf('Sweeping %d x %d = %d parameter combinations...\n', nK, nR, nK*nR);
fprintf('%-6s %-6s %8s %8s %8s %8s %8s %8s\n', ...
    'k', 'r0', 'Acc', 'Broken', 'Fixed', 'Net', 'aHi', 'aLo');
fprintf('%-6s %-6s %8s %8s %8s %8s %8s %8s\n', ...
    '--', '--', '---', '------', '-----', '---', '---', '---');

for ki = 1:nK
    k = k_vals(ki);
    for ri = 1:nR
        r0 = r0_vals(ri);

        all_alpha = [];
        all_pred_fwd = [];
        all_pred_rev = [];
        all_pred_fusion = [];
        all_truth = [];
        all_SG = [];
        all_qF = [];
        all_resRatio = [];
        all_SG_R = [];

        for m = 1:nSplits
            [S, S_label] = build_set(data, train_cells{m}, number);
            [T, T_label] = build_set(data, test_cells{m}, number);

            % Run DPGSR with current sigmoid + PR gate params
            [pred_fusion, ~, out] = DPGSR_Fusion(S, T, S_label, T_label, nt, ...
                lambda1, lambda2, 'adaptive', [], 'separate', k, r0, 100, 0.055);

            [~, pred_fwd] = max(out.PF, [], 1);
            [~, pred_rev] = max(out.PR, [], 1);

            all_pred_fwd = [all_pred_fwd, out.classes(pred_fwd)];
            all_pred_rev = [all_pred_rev, out.classes(pred_rev)];
            all_pred_fusion = [all_pred_fusion, pred_fusion];
            all_alpha = [all_alpha, out.alpha_j_true];
            all_truth = [all_truth, T_label];
            all_SG = [all_SG, out.SG];
            all_qF = [all_qF, out.quality_F];
        end

        total = length(all_truth);

        % Compute metrics
        fwd_wrong = (all_pred_fwd ~= all_truth);
        rev_wrong = (all_pred_rev ~= all_truth);
        fusion_right = (all_pred_fusion == all_truth);
        fusion_wrong = (all_pred_fusion ~= all_truth);

        % Broken: forward right but fusion wrong
        broken = sum((all_pred_fwd == all_truth) & fusion_wrong);
        % Fixed: forward wrong but fusion right
        fixed = sum(fwd_wrong & fusion_right);
        % Accuracy
        acc = sum(fusion_right) / total;

        % Alpha for sub-groups
        fwd_wrong_rev_right = fwd_wrong & (all_pred_rev == all_truth);
        fwd_right_rev_wrong_fusion_wrong = (all_pred_fwd == all_truth) & rev_wrong & fusion_wrong;

        a_hi = mean(all_alpha(fwd_wrong_rev_right));
        a_lo = mean(all_alpha(fwd_right_rev_wrong_fusion_wrong));

        results_acc(ki, ri)   = acc;
        results_broken(ki, ri) = broken;
        results_fixed(ki, ri)  = fixed;
        results_net(ki, ri)    = fixed - broken;
        results_alpha_hi(ki, ri) = a_hi;
        results_alpha_lo(ki, ri) = a_lo;

        fprintf('%-6d %-6.2f %8.4f %8d %8d %8d %8.3f %8.3f\n', ...
            k, r0, acc, broken, fixed, fixed-broken, a_hi, a_lo);
    end
end

% Find best by accuracy
[best_acc, idx] = max(results_acc(:));
[best_ki, best_ri] = ind2sub(size(results_acc), idx);
fprintf('\n=== Best by Accuracy ===\n');
fprintf('k=%d, r0=%.2f: acc=%.4f, broken=%d, fixed=%d, net=%d\n', ...
    k_vals(best_ki), r0_vals(best_ri), best_acc, ...
    results_broken(best_ki, best_ri), results_fixed(best_ki, best_ri), ...
    results_net(best_ki, best_ri));

% Find best by net (fixed - broken)
[best_net, idx] = max(results_net(:));
[best_ki, best_ri] = ind2sub(size(results_net), idx);
fprintf('\n=== Best by Net (Fixed-Broken) ===\n');
fprintf('k=%d, r0=%.2f: acc=%.4f, broken=%d, fixed=%d, net=%d\n', ...
    k_vals(best_ki), r0_vals(best_ri), results_acc(best_ki, best_ri), ...
    results_broken(best_ki, best_ri), results_fixed(best_ki, best_ri), best_net);

% Show top-5 by accuracy
fprintf('\n=== Top-5 by Accuracy ===\n');
[~, sort_idx] = sort(results_acc(:), 'descend');
for ii = 1:min(5, length(sort_idx))
    [ki, ri] = ind2sub(size(results_acc), sort_idx(ii));
    fprintf('  #%d: k=%d, r0=%.2f, acc=%.4f, broken=%d, fixed=%d, net=%d\n', ...
        ii, k_vals(ki), r0_vals(ri), results_acc(ki, ri), ...
        results_broken(ki, ri), results_fixed(ki, ri), results_net(ki, ri));
end

% Show top-5 by net
fprintf('\n=== Top-5 by Net ===\n');
[~, sort_idx] = sort(results_net(:), 'descend');
for ii = 1:min(5, length(sort_idx))
    [ki, ri] = ind2sub(size(results_acc), sort_idx(ii));
    fprintf('  #%d: k=%d, r0=%.2f, acc=%.4f, broken=%d, fixed=%d, net=%d\n', ...
        ii, k_vals(ki), r0_vals(ri), results_acc(ki, ri), ...
        results_broken(ki, ri), results_fixed(ki, ri), results_net(ki, ri));
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

% analyze_alpha_signal.m
% 深入分析：alpha 的各种可能来源 vs 正向实际准确率

clc;
rng(1);
load('yaleborigin.mat');
data = dataset;

K = 15;
number = 64;
nTrain = 8;
nt = nTrain;
nSplits = 5;
lambda1 = 1e-2;
lambda2 = 1e-2;

fprintf('=============================================================\n');
fprintf('Alpha Signal Quality Analysis (K=%d, splits=%d)\n', K, nSplits);
fprintf('=============================================================\n\n');

all_truth = [];
all_alpha_SG = [];
all_RF_margin = [];
all_RF_ratio = [];
all_PF_margin = [];
all_PF_entropy = [];
all_fwd_correct = [];

for m = 1:nSplits
    train_cell = cell(K,1); test_cell = cell(K,1);
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

    M = size(T,2);
    for j = 1:M
        rf = out.RF(:, j);
        rf_s = sort(rf);
        rf_margin = rf_s(2) - rf_s(1);
        rf_ratio = rf_s(1) / (rf_s(2) + 1e-12);

        pf = out.PF(:, j);
        pf_s = sort(pf, 'descend');
        pf_margin = pf_s(1) - pf_s(2);
        pf_norm = pf / (sum(pf) + 1e-12);
        pf_ent = -sum(pf_norm .* log(pf_norm + 1e-12));

        all_truth(end+1) = T_label(j);
        all_alpha_SG(end+1) = out.alpha_j_true(j);
        all_RF_margin(end+1) = rf_margin;
        all_RF_ratio(end+1) = rf_ratio;
        all_PF_margin(end+1) = pf_margin;
        all_PF_entropy(end+1) = pf_ent;
        all_fwd_correct(end+1) = pred_fwd(j) == T_label(j);
    end
end

%% Correlation analysis
fprintf('--- Correlation with forward correctness ---\n');
fprintf('  (sign indicates direction; |r| = strength)\n\n');
fprintf('%-28s %10s\n', 'Signal', 'Corr w/ Fwd OK');
fprintf('%-28s %10s\n', '------', '--------------');

sig_names = {'S_G alpha (current)', 'RF margin', 'RF ratio', 'PF margin', 'PF entropy'};
sig_vals = {all_alpha_SG, all_RF_margin, all_RF_ratio, all_PF_margin, all_PF_entropy};

for i = 1:length(sig_names)
    r = corrcoef(sig_vals{i}, all_fwd_correct);
    fprintf('%-28s %10.4f\n', sig_names{i}, r(1,2));
end

%% Binned accuracy
fprintf('\n--- Fwd accuracy by signal quintile ---\n');
fprintf('  Q1 = lowest signal, Q5 = highest\n\n');
fprintf('%-18s', 'Signal');
for q = 1:5; fprintf('    Q%d   ', q); end
fprintf('\n%-18s', '------');
for q = 1:5; fprintf('  ------'); end
fprintf('\n');

for i = 1:length(sig_names)
    sig = sig_vals{i};
    n_bins = 5;
    edges = quantile(sig, 0:1/n_bins:1);
    edges(1) = edges(1) - 1e-10; edges(end) = edges(end) + 1e-10;
    [~, ~, bin] = histcounts(sig, edges);

    fprintf('%-18s', sig_names{i});
    for b = 1:n_bins
        mask = (bin == b);
        if sum(mask) > 0
            acc = sum(all_fwd_correct(mask)) / sum(mask);
            fprintf('  %.4f', acc);
        else
            fprintf('     N/A');
        end
    end
    fprintf('\n');
end

%% AUROC
fprintf('\n--- AUROC for predicting fwd correctness ---\n\n');
for i = 1:length(sig_names)
    scores = sig_vals{i};
    labels = all_fwd_correct;

    if i == 1 || i == 5  % alpha and entropy: lower = better
        scores = -scores;
    end

    auroc = calc_auroc(scores, labels);
    fprintf('%-28s AUROC=%.4f\n', sig_names{i}, auroc);
end

%% ============================================================
% Test alternative alpha formulas with fusion
%% ============================================================
fprintf('\n========================================\n');
fprintf('Testing Alternative Alpha Formulas\n');
fprintf('========================================\n\n');

% Re-run to get fresh out structs
rng(1);
all_RF = {}; all_PR = {}; all_PF = {}; all_cls = {}; all_truth2 = {}; all_alpha = {};
total2 = 0;

for m = 1:nSplits
    train_cell = cell(K,1); test_cell = cell(K,1);
    for c = 1:K
        perm = randperm(number);
        train_cell{c} = perm(1:nTrain);
        test_cell{c} = perm(nTrain+1:end);
    end
    [S, S_label] = build_set(data, train_cell, number);
    [T, T_label] = build_set(data, test_cell, number);
    [~, ~, out] = DPGSR_Fusion(S, T, S_label, T_label, nt, lambda1, lambda2, 'adaptive');

    all_RF{m} = out.RF; all_PR{m} = out.PR; all_PF{m} = out.PF;
    all_cls{m} = out.classes; all_truth2{m} = T_label; all_alpha{m} = out.alpha_j_true;
    total2 = total2 + length(T_label);
end

% Method names and alpha functions
n_methods = 6;
method_names = cell(1, n_methods);
acc = zeros(1, n_methods);

for m = 1:nSplits
    RF = all_RF{m}; PR = all_PR{m}; PF = all_PF{m};
    cls = all_cls{m}; truth = all_truth2{m}; alpha_sg = all_alpha{m};
    [Kk, M] = size(RF);

    for j = 1:M
        rf = RF(:,j); rf_s = sort(rf);
        rf_ratio = rf_s(1) / (rf_s(2) + 1e-12);
        rf_margin = rf_s(2) - rf_s(1);

        pf = PF(:,j);
        pf_norm = pf / sum(pf);

        % Method 1: Alpha = 1 - RF_ratio
        a1 = 1 - rf_ratio;

        % Method 2: Alpha = exp(-RF_margin / mean(RF))
        a2 = exp(-rf_margin / (mean(rf) + 1e-12));

        % Method 3: Alpha directly = 1 - rf_ratio (clipped differently)
        a3 = (1 - rf_ratio)^2;

        % Method 4: Alpha = original SG^2  (shrink SG alpha more)
        a4 = alpha_sg(j)^2;

        % Method 5: Alpha = original SG^3
        a5 = alpha_sg(j)^3;

        % Method 6: Alpha = 1 - RF_ratio^2
        a6 = 1 - rf_ratio^2;

        alphas = [a1, a2, a3, a4, a5, a6];
        names = {'1-RF_ratio', 'exp(-RF_margin)', '(1-RF_ratio)^2', 'SG_alpha^2', 'SG_alpha^3', '1-RF_ratio^2'};

        for ai = 1:n_methods
            a = min(1, max(0, alphas(ai)));
            score = (1-a)*PF(:,j) + a*PR(:,j);
            [~, pred] = max(score);
            if cls(pred) == truth(j)
                acc(ai) = acc(ai) + 1;
            end
        end
    end
end

% Pure forward/reverse accuracy
acc_fwd = 0; acc_rev = 0;
for m = 1:nSplits
    PF = all_PF{m}; PR = all_PR{m}; cls = all_cls{m}; truth = all_truth2{m};
    [~, fwd_idx] = max(PF, [], 1);
    [~, rev_idx] = max(PR, [], 1);
    acc_fwd = acc_fwd + sum(cls(fwd_idx) == truth);
    acc_rev = acc_rev + sum(cls(rev_idx) == truth);
end

fprintf('%-28s %10s %10s\n', 'Alpha Source', 'Accuracy', 'vs Fwd');
fprintf('%-28s %10s %10s\n', '-----------', '--------', '------');
for ai = 1:n_methods
    fprintf('%-28s %10.4f %+10.4f\n', names{ai}, acc(ai)/total2, (acc(ai)-acc_fwd)/total2);
end
fprintf('%-28s %10.4f\n', 'Original SG alpha fusion', acc(1)/total2);  % placeholder — need original
fprintf('%-28s %10.4f\n', 'Pure Forward (baseline)', acc_fwd/total2);
fprintf('%-28s %10.4f\n', 'Pure Reverse (baseline)', acc_rev/total2);

% Also compute original SG alpha fusion
acc_orig = 0;
for m = 1:nSplits
    PF = all_PF{m}; PR = all_PR{m}; cls = all_cls{m};
    truth = all_truth2{m}; alpha_sg = all_alpha{m};
    M = length(truth);
    for j = 1:M
        a = alpha_sg(j);
        score = (1-a)*PF(:,j) + a*PR(:,j);
        [~, pred] = max(score);
        if cls(pred) == truth(j); acc_orig = acc_orig + 1; end
    end
end
fprintf('%-28s %10.4f %+10.4f\n', 'Original SG fusion', acc_orig/total2, (acc_orig-acc_fwd)/total2);

%% ============================================================
% Helper functions
%% ============================================================
function auroc = calc_auroc(scores, labels)
    [~, idx] = sort(scores, 'descend');
    labels = labels(idx);
    n_pos = sum(labels);
    n_neg = sum(~labels);
    if n_pos == 0 || n_neg == 0; auroc = NaN; return; end
    tpr = cumsum(labels) / n_pos;
    fpr = cumsum(~labels) / n_neg;
    auroc = 0;
    for i = 2:length(tpr)
        auroc = auroc + (fpr(i)-fpr(i-1)) * (tpr(i)+tpr(i-1))/2;
    end
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

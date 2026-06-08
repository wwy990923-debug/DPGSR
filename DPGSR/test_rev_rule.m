% test_rev_rule.m — 替代逆向分类规则：Z行匹配检索 vs argmax激活
clc; rng(1);
load('yaleborigin.mat');
data = double(dataset);

K = 15; number = 64; nTrain = 8; nt = nTrain; nSplits = 5;
lambda1 = 1e-2; lambda2 = 1e-2;

fprintf('逆向分类规则对比 (K=%d, nt=%d)\n', K, nt);
fprintf('  Rule A (current): argmax_k ||Z(j,Omega_k)|| / (sqrt(nk)*||A_k||)\n');
fprintf('  Rule B (new):     top-K training samples by |Z(j,i)| vote\n\n');

acc_A = zeros(nSplits,1);  % current PR
acc_B = zeros(nSplits,1);  % Z-row matching
acc_fwd = zeros(nSplits,1);
topK_vals = [1, 3, 5, 10, 15];

for tk = 1:length(topK_vals)
    topK = topK_vals(tk);
    for m = 1:nSplits
        tc = cell(K,1); sc = cell(K,1);
        for c = 1:K
            perm = randperm(number);
            tc{c} = perm(1:nTrain);
            sc{c} = perm(nTrain+1:end);
        end
        [S, S_label] = build_set(data, tc, number);
        [T, T_label] = build_set(data, sc, number);

        [~, ~, out] = DPGSR_Fusion(S, T, S_label, T_label, nt, lambda1, lambda2, 'adaptive');
        Z = out.D; M = size(T,2); N = size(S,2);

        [~, pf] = max(out.PF,[],1);
        [~, pr] = max(out.PR,[],1);
        acc_A(m) = sum(out.classes(pr) == T_label) / M;
        acc_fwd(m) = sum(out.classes(pf) == T_label) / M;

        % Rule B: Z-row top-K training sample voting
        pred_B = zeros(1,M);
        for j = 1:M
            [~, sorted_idx] = sort(abs(Z(j,:)), 'descend');
            top_labels = S_label(sorted_idx(1:topK));
            pred_B(j) = mode(top_labels);
        end
        acc_B(m) = sum(pred_B == T_label) / M;
    end

    fprintf('  Z top-%2d voting:  acc=%.4f  (fwd=%.4f,  PR=%.4f)\n', ...
        topK, mean(acc_B), mean(acc_fwd), mean(acc_A));
end

% Best: weighted voting (by Z magnitude)
fprintf('\n--- Weighted voting ---\n');
for tk = 1:length(topK_vals)
    topK = topK_vals(tk);
    for m = 1:nSplits
        tc = cell(K,1); sc = cell(K,1);
        for c = 1:K
            perm = randperm(number);
            tc{c} = perm(1:nTrain);
            sc{c} = perm(nTrain+1:end);
        end
        [S, S_label] = build_set(data, tc, number);
        [T, T_label] = build_set(data, sc, number);
        [~, ~, out] = DPGSR_Fusion(S, T, S_label, T_label, nt, lambda1, lambda2, 'adaptive');
        Z = out.D; M = size(T,2);

        pred_B = zeros(1,M);
        for j = 1:M
            [sorted_val, sorted_idx] = sort(abs(Z(j,:)), 'descend');
            top_labels = S_label(sorted_idx(1:topK));
            top_vals = sorted_val(1:topK);
            % weighted vote
            votes = zeros(1,K);
            for kk = 1:K
                votes(kk) = sum(top_vals(top_labels == out.classes(kk)));
            end
            [~, pred_B(j)] = max(votes);
            pred_B(j) = out.classes(pred_B(j));
        end
        acc_B(m) = sum(pred_B == T_label) / M;
    end
    fprintf('  Z top-%2d weighted: acc=%.4f  (fwd=%.4f)\n', topK, mean(acc_B), mean(acc_fwd));
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

% quick_test_reverse.m - Quick test of fixed Reverse.m
clc;
rng(1);
load('yaleborigin.mat');
data = dataset;

K = 15;
number = 64;
nTrain = 8;
nt = nTrain;
nSplits = 3;

l2s = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 10];
n_formulas = 7;

best_acc = -inf(1, n_formulas);
best_l2 = zeros(1, n_formulas);

sp = cell(nSplits, 1);
for m = 1:nSplits
    tc = cell(K,1); sc = cell(K,1);
    for c = 1:K
        p = randperm(number);
        tc{c} = p(1:nTrain);
        sc{c} = p(nTrain+1:end);
    end
    sp{m} = {tc, sc};
end

for li = 1:length(l2s)
    l2 = l2s(li);
    fprintf('lambda2=%.0e: ', l2);
    acc_f = zeros(1, n_formulas);
    total = 0;

    for m = 1:nSplits
        tc = sp{m}{1}; sc = sp{m}{2};
        [S, Sl] = build_set(data, tc, number);
        [T, Tl] = build_set(data, sc, number);
        [~, ~, out] = DPGSR_Fusion(S, T, Sl, Tl, nt, 1e-2, l2, 'adaptive');
        D = out.D;
        M = size(T,2);
        cls = out.classes;
        total = total + M;

        for j = 1:M
            dj = D(j,:);
            for kk = 1:K
                idx = (Sl == cls(kk));
                nk = sum(idx);
                Ak_norm = norm(S(:,idx), 'fro');
                d_jk = dj(idx);
                SR(1,kk) = norm(d_jk,2) / (sqrt(nk)*Ak_norm + 1e-12);
                SR(2,kk) = norm(d_jk,2) / (norm(dj,2) + 1e-12);
                SR(3,kk) = norm(d_jk,2);
                SR(4,kk) = sum(d_jk.^2);
                SR(6,kk) = norm(d_jk,2) / (sqrt(nk) + 1e-12);
                SR(7,kk) = norm(d_jk,1);
            end
            for kk = 1:K
                idx = (Sl == cls(kk));
                res = norm(S(:,idx) - T*D(:,idx), 'fro')^2;
                SR(5,kk) = -res;
            end
            for fi = 1:n_formulas
                sr = SR(fi,:);
                if fi == 5
                    [~, pred] = max(sr);
                else
                    pr_f = sr / (sum(sr) + 1e-12);
                    [~, pred] = max(pr_f);
                end
                if cls(pred) == Tl(j)
                    acc_f(fi) = acc_f(fi) + 1;
                end
            end
        end
    end

    for fi = 1:n_formulas
        acc = acc_f(fi) / total;
        fprintf('F%d=%.3f ', fi, acc);
        if acc > best_acc(fi)
            best_acc(fi) = acc;
            best_l2(fi) = l2;
        end
    end
    fprintf('\n');
end

fprintf('\n=== Best per formula ===\n');
names = {'||D(j,idx)||/(sqrt(nk)*||Sk||) [orig]', ...
         '||D(j,idx)||/||D(j,:)|| [relative]', ...
         '||D(j,idx)|| [raw]', ...
         'sum(D.^2) [energy]', ...
         'reverse residual', ...
         '||D(j,idx)||/sqrt(nk)', ...
         '||D(j,idx)||_1'};
for fi = 1:n_formulas
    fprintf('F%d: %.4f (l2=%.0e)  %s\n', fi, best_acc(fi), best_l2(fi), names{fi});
end

% Forward/Reverse baseline
acc_fwd = 0; acc_rev_orig = 0;
for m = 1:nSplits
    tc = sp{m}{1}; sc = sp{m}{2};
    [S, Sl] = build_set(data, tc, number);
    [T, Tl] = build_set(data, sc, number);
    [~, ~, out] = DPGSR_Fusion(S, T, Sl, Tl, nt, 1e-2, best_l2(1), 'adaptive');
    [~, pfwd] = max(out.PF, [], 1);
    [~, prev] = max(out.PR, [], 1);
    acc_fwd = acc_fwd + sum(out.classes(pfwd) == Tl);
    acc_rev_orig = acc_rev_orig + sum(out.classes(prev) == Tl);
end
total_all = nSplits * (K * (number-nTrain));
fprintf('\nPure Forward: %.4f\n', acc_fwd / total_all);
fprintf('Orig PR: %.4f\n', acc_rev_orig / total_all);

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

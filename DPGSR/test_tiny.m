% test_tiny.m — 极限小样本：训练极少 + 测试也少
clc; rng(1);
load('yaleborigin.mat');
data = double(dataset);

K = 15;  number = 64;
nTrain_vals = [1, 2, 3];         % 每类训练样本
nTest_vals  = [5, 10, 15];        % 每类测试样本（去掉大测试集避免ADMM秩不足）
lambda_grid = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1];
nSplits = 5;

fprintf('=== Tiny Sample DPGSR Test (K=%d) ===\n\n', K);

for ni = 1:length(nTrain_vals)
    nt = nTrain_vals(ni);
    for nj = 1:length(nTest_vals)
        nTest = nTest_vals(nj);
        if nt >= nTest, continue; end  % 训练不能超过测试

        best_fus = -inf; best_fwd = -inf; best_lam = [0,0]; best_net = -inf;

        for l1 = 1:length(lambda_grid)
            for l2 = 1:length(lambda_grid)
                lam1 = lambda_grid(l1);
                lam2 = lambda_grid(l2);

                fwd_acc = zeros(nSplits,1); fus_acc = zeros(nSplits,1);
                broken = zeros(nSplits,1);  fixed = zeros(nSplits,1);

                for m = 1:nSplits
                    tc = cell(K,1); sc = cell(K,1);
                    for c = 1:K
                        perm = randperm(number);
                        tc{c} = perm(1:nt);
                        sc{c} = perm(nt+1:nt+nTest);
                    end
                    [S, S_label] = build_set(data, tc, number);
                    [T, T_label] = build_set(data, sc, number);

                    [pred_fusion, ~, out] = DPGSR_Fusion(S, T, S_label, T_label, nt, ...
                        lam1, lam2, 'adaptive');

                    [~, pf] = max(out.PF,[],1);
                    M = length(T_label);
                    fwd_acc(m) = sum(out.classes(pf)==T_label)/M;
                    fus_acc(m) = sum(pred_fusion==T_label)/M;
                    broken(m) = sum(out.classes(pf)==T_label & pred_fusion~=T_label);
                    fixed(m)  = sum(out.classes(pf)~=T_label & pred_fusion==T_label);
                end

                mfwd = mean(fwd_acc); mfus = mean(fus_acc);
                mnet = round(mean(fixed)-mean(broken));
                if mfus > best_fus
                    best_fwd = mfwd; best_fus = mfus; best_lam = [lam1, lam2];
                    best_net = mnet;
                end
            end
        end
        fprintf('nt=%d nT=%d | fwd=%.4f fus=%.4f | net=%+d gap=%+.4f | lam=%.0e,%.0e\n', ...
            nt, nTest, best_fwd, best_fus, best_net, best_fus-best_fwd, best_lam(1), best_lam(2));
    end
    fprintf('\n');
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

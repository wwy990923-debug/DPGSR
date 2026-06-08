% test_joint.m - Quick test of JointConstrained solver
rng(1);

load('yaleborigin.mat');

% Small subset: 3 classes, 4 training per class
K = 3;
number = 64;
nTrain = 4;
nt = nTrain;
dataset = dataset(:, 1:K*number);

train_cell = cell(K,1);
test_cell = cell(K,1);
for c = 1:K
    perm = randperm(number);
    train_cell{c} = perm(1:nTrain);
    test_cell{c} = perm(nTrain+1:end);
end

[S, S_label] = build_set_by_class(dataset, train_cell, number);
[T, T_label] = build_set_by_class(dataset, test_cell, number);

fprintf('S size: %d x %d\n', size(S,1), size(S,2));
fprintf('T size: %d x %d\n', size(T,1), size(T,2));
fprintf('nt = %d\n\n', nt);

% --- Test 1: Joint solver only ---
fprintf('=== Test 1: JointConstrained ===\n');
tic;
[W_j, D_j, out_j] = JointConstrained(S, T, nt, 1, 1);
t_j = toc;
fprintf('Time: %.2f sec, iterations: %d, final stopC: %.3e\n', t_j, out_j.iter, out_j.stopC);

% --- Test 2: Separate solvers ---
fprintf('\n=== Test 2: Separate Forward/Reverse ===\n');
tic;
W_s = Forward(S, T, nt, 1e-2);
t_fwd = toc;
tic;
D_s = Reverse(T, S, nt, 1e-2);
t_rev = toc;
fprintf('Forward: %.2f sec, Reverse: %.2f sec\n', t_fwd, t_rev);

% --- Test 3: Fusion with joint solver ---
fprintf('\n=== Test 3: DPGSR_Fusion (joint) ===\n');
[pred_j, acc_j, ~] = DPGSR_Fusion(S, T, S_label, T_label, nt, 1, 1, 'adaptive', [], 'joint');
fprintf('Joint fusion accuracy: %.4f\n', acc_j);

% --- Test 4: Fusion with separate solvers ---
fprintf('\n=== Test 4: DPGSR_Fusion (separate) ===\n');
[pred_s, acc_s, ~] = DPGSR_Fusion(S, T, S_label, T_label, nt, 1e-2, 1e-2, 'adaptive', [], 'separate');
fprintf('Separate fusion accuracy: %.4f\n', acc_s);

fprintf('\n=== Summary ===\n');
fprintf('Joint  (constrained)  : acc=%.4f, time=%.2fs, iter=%d\n', acc_j, t_j, out_j.iter);
fprintf('Separate (penalty)    : acc=%.4f, time=%.2fs\n', acc_s, t_fwd+t_rev);

% Also test with different gamma values
fprintf('\n=== Test 5: Joint with different gammas ===\n');
gammas = [1e-2, 1e-1, 1, 10];
for g = gammas
    [W_g, D_g, ~] = JointConstrained(S, T, nt, g, g);
    [pred_g, acc_g, ~] = DPGSR_Fusion(S, T, S_label, T_label, nt, g, g, 'adaptive', [], 'joint');
    fprintf('gamma=%.0e: acc=%.4f\n', g, acc_g);
end

fprintf('\nDone.\n');

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

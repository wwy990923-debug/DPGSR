% demo_vgdpgsr.m — Minimal working example of VG-DP-GSR
% Required files: Forward.m, Reverse.m, DPGSR_Fusion.m, pcayaleb50.mat
clc; rng(1);

% Load Yale B PCA-50 data (38 classes, 64 images per class)
load('pcayaleb50.mat');
D = double(dataset);

% Sub1 training, test on Subset 2-5 (cross-illumination)
K = 38; N = 64; nt = 7;
tr = 1:7; te = 8:64;
S = []; Sl = []; T = []; Tl = [];
for c = 1:K
    for i = tr; S = [S D(:, (c-1)*N + i)]; Sl = [Sl c]; end
    for i = te; T = [T D(:, (c-1)*N + i)]; Tl = [Tl c]; end
end
M = size(T, 2);

% Parameters (VG-DP-GSR optimal)
lambda1 = 1.0; lambda2 = 1e-2; theta = 0.70; delta = 0.05;

fprintf('Yale B Sub1: %d train, %d test, %d classes\n', size(S,2), M, K);
fprintf('Running VG-DP-GSR...\n');

% Run Dual-Path Group Sparse Representation
[~, ~, out] = DPGSR_Fusion(S, T, Sl, Tl, nt, lambda1, lambda2, ...
    'adaptive', [], [], theta, delta);

% Forward prediction
[~, pf] = max(out.PF, [], 1);
fwd = sum(out.classes(pf) == Tl) / M * 100;

% Reverse prediction
[~, pr] = max(out.PR, [], 1);
rev = sum(out.classes(pr) == Tl) / M * 100;

% Hard-switch fusion
Z = out.D;
Sh = zeros(K, M);
for j = 1:M
    vf = norm(Z(j, Sl == out.classes(pf(j))), 2) / (norm(Z(j,:), 2) + 1e-12);
    PRm = sort(out.PR(:, j), 'descend');
    PRm = PRm(1) - PRm(2);
    if vf > theta; alfa = 0; elseif PRm > delta; alfa = 1; else alfa = 0; end
    Sh(:, j) = (1 - alfa) * out.PF(:, j) + alfa * out.PR(:, j);
end
[~, ph] = max(Sh, [], 1);
fus = sum(out.classes(ph) == Tl) / M * 100;

fprintf('Forward: %.2f%%  Reverse: %.2f%%  VG-DP-GSR: %.2f%%\n', fwd, rev, fus);

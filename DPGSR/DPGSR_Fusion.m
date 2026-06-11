function [pred_label, acc, out] = DPGSR_Fusion(S, T, S_label, T_label, nt, lambda1, lambda2, mode, fixedAlpha, solver, verif_thresh, pr_margin_thresh)
% DPGSR_Fusion
%
% Dual-Path Group Sparse Representation classifier.
%
% Forward path:
%   T ≈ S * W
%
% Reverse path:
%   S ≈ T * D
%
% Inputs:
%   S          - training data, d x N
%   T          - test data, d x M
%   S_label    - training labels, 1 x N or N x 1
%   T_label    - test labels, 1 x M or M x 1, can be [] if unknown
%   nt         - number of training samples per class
%   lambda1    - forward regularization parameter
%   lambda2    - reverse regularization parameter
%   mode       - 'adaptive' or 'fixed'
%   fixedAlpha - fixed alpha value, only used when mode = 'fixed'
%   solver     - 'separate' (default, penalty form) or
%                'joint' (constrained form, jointly optimized)
%   k_sigmoid  - (optional) sigmoid steepness, default 15
%   r0_sigmoid - (optional) sigmoid center, default 1.55
%
% Outputs:
%   pred_label - predicted labels, 1 x M
%   acc        - classification accuracy, if T_label is given
%   out        - intermediate results

if nargin < 8 || isempty(mode)
    mode = 'adaptive';
end

if nargin < 9 || isempty(fixedAlpha)
    fixedAlpha = 0.5;
end

if nargin < 10 || isempty(solver)
    solver = 'separate';
end

if nargin < 11 || isempty(verif_thresh)
    verif_thresh = 0.65;      % verif > 阈值 → 信正向
end

if nargin < 12 || isempty(pr_margin_thresh)
    pr_margin_thresh = 0.05;  % PR_margin > 阈值 → 逆向值得考虑
end

eps0 = 1e-12;

S_label = S_label(:)';

if ~isempty(T_label)
    T_label = T_label(:)';
end

classes = unique(S_label, 'stable');
K = length(classes);
M = size(T, 2);
N = size(S, 2);

if length(S_label) ~= N
    error('Length of S_label must equal the number of columns in S.');
end

if ~isempty(T_label) && length(T_label) ~= M
    error('Length of T_label must equal the number of columns in T.');
end

%% ============================================================
% 1. Forward path: T ≈ S * W
% 2. Reverse path: S ≈ T * D
%% ============================================================

if strcmpi(solver, 'joint')
    % Joint constrained ADMM solver
    % lambda1/lambda2 serve as gamma1/gamma2 (group-norm weights)
    [W, D, joint_out] = JointConstrained(S, T, nt, lambda1, lambda2);
else
    % Original separate penalty-form solvers
    W = Forward(S, T, S_label, lambda1);     % W size: N x M

    if size(W,1) ~= N || size(W,2) ~= M
        error('Forward output W has wrong size. Expected N x M.');
    end

    D = Reverse(T, S, S_label, lambda2);

    if size(D,1) ~= M || size(D,2) ~= N
        error('Reverse output D has wrong size.');
    end
end

%% ============================================================
% 3. Forward residual evidence P_F
%% ============================================================

RF = zeros(K, M);

for j = 1:M
    for kk = 1:K
        cls = classes(kk);
        idx = (S_label == cls);

        x_kj = W(idx, j);
        RF(kk, j) = norm(T(:, j) - S(:, idx) * x_kj, 2)^2;
    end
end

% Temperature-controlled Softmax over negative residuals
tauF = mean(RF, 1) + eps0;
logPF = -bsxfun(@rdivide, RF, tauF);

% Numerical stability
logPF = bsxfun(@minus, logPF, max(logPF, [], 1));

PF = exp(logPF);
PF = bsxfun(@rdivide, PF, sum(PF, 1) + eps0);

%% ============================================================
% 4. Reverse activation evidence P_R
%% ============================================================

SR = zeros(K, M);

for j = 1:M
    for kk = 1:K
        cls = classes(kk);
        idx = (S_label == cls);

        nk = sum(idx);
        Ak_norm = norm(S(:, idx), 'fro');

        % D(j, idx): contribution of test sample j to class-k training atoms
        a_kj = norm(D(j, idx), 2);

        % Normalized reverse activation (class-balance calibration)
        SR(kk, j) = a_kj / (sqrt(nk) * Ak_norm + eps0);
    end
end

% Normalize reverse activation scores into posterior proxy
PR = bsxfun(@rdivide, SR + eps0, sum(SR + eps0, 1) + eps0);

%% ============================================================
% 5. Verification-Gated Fusion
%    只用逆向验证分 verif = ||D(j,Omega_predF)||_2 / ||D(j,:)||_2
%    作为门控信号，取代复杂多层 sigmoid
%% ============================================================

G = zeros(K, M);
SG = zeros(1, M);
alpha_j = zeros(1, M);

% --- Predictions ---
[~, pred_F] = max(PF, [], 1);   % 1 x M, forward prediction
[~, pred_R] = max(PR, [], 1);   % 1 x M, reverse prediction

for j = 1:M
    for kk = 1:K
        cls = classes(kk);
        idx = (S_label == cls);
        nk = sum(idx);
        G(kk, j) = norm(W(idx, j), 2) / (sqrt(nk) + eps0);
    end

    gj = G(:, j);
    g2 = gj.^2;
    total_energy = sum(g2) + eps0;
    max_ratio = max(g2) / total_energy;
    SG(j) = (max_ratio - 1/K) / (1 - 1/K);
    SG(j) = max(0, min(1, SG(j)));

    % Verification score: does D(j,:) agree with forward's prediction?
    idx_F = (S_label == classes(pred_F(j)));
    verif_score = norm(D(j, idx_F), 2) / (norm(D(j, :), 2) + eps0);

    % PR margin for reverse confidence check
    PR_sorted = sort(PR(:, j), 'descend');
    PR_margin = PR_sorted(1) - PR_sorted(2);

    if verif_score > verif_thresh
        % High verification: forward is consistent with reverse → trust forward
        alpha_j(j) = 0;
    elseif PR_margin > pr_margin_thresh
        % Low verification + reverse is confident → forward likely wrong → trust reverse
        alpha_j(j) = 1;
    else
        % Neither confident → conservative, trust forward (more accurate overall)
        alpha_j(j) = 0;
    end
end

alpha_j_true = alpha_j;  % save before fixed mode overwrites

%% ============================================================
% 6. Final fusion decision
%% ============================================================

Score = zeros(K, M);

switch lower(mode)

    case 'adaptive'
        for j = 1:M
            Score(:, j) = (1 - alpha_j(j)) * PF(:, j) + alpha_j(j) * PR(:, j);
        end

    case 'fixed'
        for j = 1:M
            Score(:, j) = (1 - fixedAlpha) * PF(:, j) + fixedAlpha * PR(:, j);
        end
        alpha_j(:) = fixedAlpha;

    otherwise
        error('mode must be either adaptive or fixed.');
end

[~, pred_idx] = max(Score, [], 1);
pred_label = classes(pred_idx);

%% ============================================================
% 7. Accuracy
%% ============================================================

if isempty(T_label)
    acc = NaN;
else
    acc = sum(pred_label == T_label) / M;
end

%% ============================================================
% 8. Return intermediate variables
%% ============================================================

out.W = W;
out.D = D;
out.RF = RF;
out.SR = SR;
out.PF = PF;
out.PR = PR;
out.G = G;
out.SG = SG;
out.pred_F = pred_F;
out.pred_R = pred_R;
out.alpha_j = alpha_j;
out.alpha_j_true = alpha_j_true;
out.Score = Score;
out.classes = classes;
out.mode = mode;
out.fixedAlpha = fixedAlpha;
out.solver = solver;

if strcmpi(solver, 'joint')
    out.joint_iter = joint_out.iter;
    out.joint_stopC = joint_out.stopC;
    out.joint_obj = joint_out.obj;
end

end

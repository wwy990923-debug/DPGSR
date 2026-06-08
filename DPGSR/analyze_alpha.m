% analyze_alpha.m
% 分析 alpha_j / SG / quality_F 与正向/逆向/融合准确率的关系
% 验证 Plan C 残差质量门控是否有效

clc;
rng(1);
load('yaleborigin.mat');
data = dataset;

K = 15;
number = 64;
nTrain = 8;
nt = nTrain;
nSplits = 10;

% 可调参数
lambda1 = 1e-2;
lambda2 = 1e-2;

fprintf('=============================================================\n');
fprintf('Plan C: Residual Quality Gate Analysis\n');
fprintf('K=%d, train/class=%d, splits=%d, lambda=[%.0e, %.0e]\n', ...
    K, nTrain, nSplits, lambda1, lambda2);
fprintf('=============================================================\n\n');

all_pred_fwd = [];
all_pred_rev = [];
all_pred_fusion = [];
all_alpha = [];
all_SG = [];
all_SG_R = [];
all_res_ratio = [];
all_PF_margin = [];
all_PR_margin = [];
all_nn_fwd = [];
all_nn_rev = [];
all_nn_ent = [];
all_PF_fwd = [];
all_PF_rev = [];
all_PR_fwd = [];
all_PR_bi  = [];
all_quality_F = [];
all_quality_contrast = [];
all_quality_abs = [];
all_truth = [];

for m = 1:nSplits
    % 随机划分
    train_cell = cell(K,1);
    test_cell = cell(K,1);
    for c = 1:K
        perm = randperm(number);
        train_cell{c} = perm(1:nTrain);
        test_cell{c} = perm(nTrain+1:end);
    end
    [S, S_label] = build_set(data, train_cell, number);
    [T, T_label] = build_set(data, test_cell, number);

    % 跑 DPGSR
    [pred_fusion, ~, out] = DPGSR_Fusion(S, T, S_label, T_label, nt, ...
        lambda1, lambda2, 'adaptive');

    % 提取正向/逆向单独预测
    [~, pred_fwd] = max(out.PF, [], 1);  % 纯正向
    [~, pred_rev] = max(out.PR, [], 1);  % 纯逆向

    all_pred_fwd = [all_pred_fwd, out.classes(pred_fwd)];
    all_pred_rev = [all_pred_rev, out.classes(pred_rev)];
    all_pred_fusion = [all_pred_fusion, pred_fusion];
    all_alpha = [all_alpha, out.alpha_j_true];
    all_SG = [all_SG, out.SG];
    all_quality_F = [all_quality_F, out.quality_F];
    all_quality_contrast = [all_quality_contrast, out.quality_contrast];
    all_quality_abs = [all_quality_abs, out.quality_abs];
    all_truth = [all_truth, T_label];

    % 残差比: RF(pred_R) / RF(pred_F) — <1 说明逆向类重构更好
    RF_mat = out.RF;
    [~, pf_idx] = max(out.PF, [], 1);
    [~, pr_idx] = max(out.PR, [], 1);
    for j = 1:size(T,2)
        r_fwd_j = RF_mat(pf_idx(j), j);
        r_rev_j = RF_mat(pr_idx(j), j);
        all_res_ratio = [all_res_ratio, r_rev_j / (r_fwd_j + eps)];
    end

    % 计算逆向集中度 SG_R（从 D 矩阵）
    D = out.D;
    S_lab = S_label;
    cls_u = out.classes;
    Kk = length(cls_u);
    SG_R_split = zeros(1, size(T,2));
    for j = 1:size(T,2)
        G_R = zeros(Kk, 1);
        for kk = 1:Kk
            idx = (S_lab == cls_u(kk));
            nk = sum(idx);
            G_R(kk) = norm(D(j, idx), 2) / (sqrt(nk) + eps);
        end
        g2_R = G_R.^2;
        max_ratio_R = max(g2_R) / (sum(g2_R) + eps);
        SG_R_split(j) = (max_ratio_R - 1/Kk) / (1 - 1/Kk);
        SG_R_split(j) = max(0, min(1, SG_R_split(j)));
    end
    all_SG_R = [all_SG_R, SG_R_split];

    % k-NN 局部邻域一致性（独立于 W/D 的新信息源）
    k_nn = 5;
    nn_fwd_agree = zeros(1, size(T,2));
    nn_rev_agree = zeros(1, size(T,2));
    nn_entropy   = zeros(1, size(T,2));
    for j = 1:size(T,2)
        diff = S - T(:,j);
        dists = sqrt(sum(diff.^2, 1));
        [~, sorted_idx] = sort(dists);
        nn_labels = S_label(sorted_idx(1:k_nn));

        nn_fwd_agree(j) = sum(nn_labels == out.classes(pf_idx(j))) / k_nn;
        nn_rev_agree(j) = sum(nn_labels == out.classes(pr_idx(j))) / k_nn;

        cnt = zeros(1, Kk);
        for kk = 1:Kk
            cnt(kk) = sum(nn_labels == cls_u(kk));
        end
        p = cnt / k_nn;
        p = p(p > 0);
        nn_entropy(j) = -sum(p .* log(p + eps));
    end
    all_nn_fwd = [all_nn_fwd, nn_fwd_agree];
    all_nn_rev = [all_nn_rev, nn_rev_agree];
    all_nn_ent  = [all_nn_ent, nn_entropy];

    % 交叉概率：PF给逆向类的分，PR给正向类的分
    PF_fwd_val = zeros(1, size(T,2));
    PF_rev_val = zeros(1, size(T,2));
    PR_fwd_val = zeros(1, size(T,2));
    PR_top2_gap = zeros(1, size(T,2));
    for j = 1:size(T,2)
        PF_fwd_val(j) = out.PF(pf_idx(j), j);
        PF_rev_val(j) = out.PF(pr_idx(j), j);
        PR_fwd_val(j) = out.PR(pf_idx(j), j);
        pr_s = sort(out.PR(:,j), 'descend');
        PR_top2_gap(j) = (pr_s(1) - pr_s(2)) / (pr_s(2) - pr_s(3) + eps);  % bimodality
    end
    all_PF_fwd = [all_PF_fwd, PF_fwd_val];
    all_PF_rev = [all_PF_rev, PF_rev_val];
    all_PR_fwd = [all_PR_fwd, PR_fwd_val];
    all_PR_bi   = [all_PR_bi, PR_top2_gap];

    % PF/PR margins
    PF_margins = zeros(1, size(T,2));
    PR_margins = zeros(1, size(T,2));
    for j = 1:size(T,2)
        pf_s = sort(out.PF(:,j), 'descend');
        pr_s = sort(out.PR(:,j), 'descend');
        PF_margins(j) = pf_s(1) - pf_s(2);
        PR_margins(j) = pr_s(1) - pr_s(2);
    end
    all_PF_margin = [all_PF_margin, PF_margins];
    all_PR_margin = [all_PR_margin, PR_margins];
end

total = length(all_alpha);

%% =============================================================
% 1. 整体统计
%% =============================================================
fprintf('--- Overall Statistics ---\n');
fprintf('SG:            min=%.4f, max=%.4f, mean=%.4f, std=%.4f\n', ...
    min(all_SG), max(all_SG), mean(all_SG), std(all_SG));
fprintf('quality_contr: min=%.4f, max=%.4f, mean=%.4f, std=%.4f\n', ...
    min(all_quality_contrast), max(all_quality_contrast), mean(all_quality_contrast), std(all_quality_contrast));
fprintf('quality_abs:   min=%.4f, max=%.4f, mean=%.4f, std=%.4f\n', ...
    min(all_quality_abs), max(all_quality_abs), mean(all_quality_abs), std(all_quality_abs));
fprintf('quality_F:     min=%.4f, max=%.4f, mean=%.4f, std=%.4f\n', ...
    min(all_quality_F), max(all_quality_F), mean(all_quality_F), std(all_quality_F));
fprintf('alpha (Plan C): min=%.4f, max=%.4f, mean=%.4f, std=%.4f\n\n', ...
    min(all_alpha), max(all_alpha), mean(all_alpha), std(all_alpha));

%% =============================================================
% 2. 按 alpha 分段统计准确率
%% =============================================================
alpha_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
nBins = length(alpha_bins) - 1;

fprintf('%-12s %8s %10s %10s %10s %10s\n', ...
    'Alpha Range', 'Count', 'Fwd Acc', 'Rev Acc', 'Fusion Acc', '%Samples');
fprintf('%-12s %8s %10s %10s %10s %10s\n', ...
    '------------', '--------', '----------', '----------', '----------', '----------');

for b = 1:nBins
    lo = alpha_bins(b);
    hi = alpha_bins(b + 1);
    mask = (all_alpha >= lo & all_alpha < hi);
    if b == nBins
        mask = (all_alpha >= lo & all_alpha <= hi);
    end
    n_in_bin = sum(mask);
    pct = n_in_bin / total * 100;

    if n_in_bin > 0
        acc_fwd = sum(all_pred_fwd(mask) == all_truth(mask)) / n_in_bin;
        acc_rev = sum(all_pred_rev(mask) == all_truth(mask)) / n_in_bin;
        acc_fusion = sum(all_pred_fusion(mask) == all_truth(mask)) / n_in_bin;
        avg_qf = mean(all_quality_F(mask));
    else
        acc_fwd = NaN; acc_rev = NaN; acc_fusion = NaN; avg_qf = NaN;
    end

    fprintf('[%3.1f, %3.1f)  %8d %10.4f %10.4f %10.4f %9.1f%%  qF=%.3f\n', ...
        lo, hi, n_in_bin, acc_fwd, acc_rev, acc_fusion, pct, avg_qf);
end

%% =============================================================
% 3. 按 quality_F 分段统计（验证门控是否有效）
%% =============================================================
fprintf('\n--- By Quality_F Bins ---\n');
q_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
nQBins = length(q_bins) - 1;
fprintf('%-14s %8s %10s %10s %10s %10s\n', ...
    'Qual_F Range', 'Count', 'Fwd Acc', 'Rev Acc', 'Fusion Acc', 'Mean Alpha');
fprintf('%-14s %8s %10s %10s %10s %10s\n', ...
    '------------', '--------', '----------', '----------', '----------', '----------');

for b = 1:nQBins
    lo = q_bins(b);
    hi = q_bins(b+1);
    mask = (all_quality_F >= lo & all_quality_F < hi);
    if b == nQBins
        mask = (all_quality_F >= lo & all_quality_F <= hi);
    end
    n_in_bin = sum(mask);
    if n_in_bin > 0
        acc_fwd = sum(all_pred_fwd(mask) == all_truth(mask)) / n_in_bin;
        acc_rev = sum(all_pred_rev(mask) == all_truth(mask)) / n_in_bin;
        acc_fusion = sum(all_pred_fusion(mask) == all_truth(mask)) / n_in_bin;
        avg_alpha = mean(all_alpha(mask));
    else
        acc_fwd = NaN; acc_rev = NaN; acc_fusion = NaN; avg_alpha = NaN;
    end
    fprintf('[%3.1f, %3.1f)  %8d %10.4f %10.4f %10.4f %10.4f\n', ...
        lo, hi, n_in_bin, acc_fwd, acc_rev, acc_fusion, avg_alpha);
end

%% =============================================================
% 4. 关键分析：正向错误但被融合纠正的样本
%% =============================================================
fwd_wrong = (all_pred_fwd ~= all_truth);
fusion_right = (all_pred_fusion == all_truth);
fixed_by_reverse = fwd_wrong & fusion_right;
fwd_wrong_count = sum(fwd_wrong);
fixed_count = sum(fixed_by_reverse);

fprintf('\n--- When forward is WRONG ---\n');
fprintf('Total forward errors: %d / %d (%.2f%%)\n', fwd_wrong_count, total, fwd_wrong_count/total*100);

if fwd_wrong_count > 0
    fprintf('Fixed by fusion (trusted reverse): %d (%.1f%% of errors)\n', ...
        fixed_count, fixed_count/fwd_wrong_count*100);
    fprintf('  Fixed samples:    alpha=%.3f, SG=%.3f, qF=%.3f\n', ...
        mean(all_alpha(fixed_by_reverse)), mean(all_SG(fixed_by_reverse)), ...
        mean(all_quality_F(fixed_by_reverse)));
    not_fixed = fwd_wrong & ~fusion_right;
    if sum(not_fixed) > 0
        fprintf('  NOT fixed samples: alpha=%.3f, SG=%.3f, qF=%.3f\n', ...
            mean(all_alpha(not_fixed)), mean(all_SG(not_fixed)), ...
            mean(all_quality_F(not_fixed)));
    end
end

%% =============================================================
% 5. 正向正确但被融合搞错的样本（alpha 过高导致）
%% =============================================================
fwd_right = (all_pred_fwd == all_truth);
fusion_wrong = (all_pred_fusion ~= all_truth);
broken_by_reverse = fwd_right & fusion_wrong;
broken_count = sum(broken_by_reverse);

fprintf('\n--- When forward is RIGHT but fusion is WRONG ---\n');
fprintf('Broken by fusion: %d (alpha=%.3f, SG=%.3f, qF=%.3f)\n', ...
    broken_count, ...
    nanmean(all_alpha(broken_by_reverse)), ...
    nanmean(all_SG(broken_by_reverse)), ...
    nanmean(all_quality_F(broken_by_reverse)));

%% =============================================================
% 6. 反向同理
%% =============================================================
rev_wrong = (all_pred_rev ~= all_truth);
fixed_by_forward = rev_wrong & fusion_right;
rev_wrong_count = sum(rev_wrong);
fixed_fwd_count = sum(fixed_by_forward);

fprintf('\n--- When reverse is WRONG ---\n');
fprintf('Total reverse errors: %d / %d (%.2f%%)\n', rev_wrong_count, total, rev_wrong_count/total*100);
if rev_wrong_count > 0
    fprintf('Fixed by fusion (trusted forward): %d (%.1f%% of errors)\n', ...
        fixed_fwd_count, fixed_fwd_count/rev_wrong_count*100);
    fprintf('  Fixed samples:    alpha=%.3f, SG=%.3f, qF=%.3f\n', ...
        mean(all_alpha(fixed_by_forward)), mean(all_SG(fixed_by_forward)), ...
        mean(all_quality_F(fixed_by_forward)));
end

%% =============================================================
% 7. 正向错误时，逆向对不对？【核心问题】
%% =============================================================
fwd_wrong_rev_right = fwd_wrong & (all_pred_rev == all_truth);
fwd_wrong_rev_wrong = fwd_wrong & (all_pred_rev ~= all_truth);
n_fwd_wrong_rev_right = sum(fwd_wrong_rev_right);
n_fwd_wrong_rev_wrong = sum(fwd_wrong_rev_wrong);

fprintf('\n=== KEY: When forward is WRONG, is reverse RIGHT? ===\n');
fprintf('Forward errors total: %d\n', fwd_wrong_count);
fprintf('  Reverse CORRECT:  %5d (%.1f%% of fwd errors)\n', ...
    n_fwd_wrong_rev_right, n_fwd_wrong_rev_right/fwd_wrong_count*100);
fprintf('  Reverse WRONG:    %5d (%.1f%% of fwd errors)\n', ...
    n_fwd_wrong_rev_wrong, n_fwd_wrong_rev_wrong/fwd_wrong_count*100);

% Of those where reverse IS right: did fusion use it?
fwd_wrong_rev_right_fusion_right = fwd_wrong_rev_right & fusion_right;
fwd_wrong_rev_right_fusion_wrong = fwd_wrong_rev_right & ~fusion_right;
n_used = sum(fwd_wrong_rev_right_fusion_right);
n_wasted = sum(fwd_wrong_rev_right_fusion_wrong);

fprintf('\n--- When forward WRONG but reverse RIGHT ---\n');
fprintf('Total: %d\n', n_fwd_wrong_rev_right);
if n_fwd_wrong_rev_right > 0
    fprintf('  Fusion CORRECT (alpha high enough): %5d (%.1f%%)\n', ...
        n_used, n_used/n_fwd_wrong_rev_right*100);
    fprintf('    -> alpha=%.3f, SG=%.3f, qF=%.3f\n', ...
        mean(all_alpha(fwd_wrong_rev_right_fusion_right)), ...
        mean(all_SG(fwd_wrong_rev_right_fusion_right)), ...
        mean(all_quality_F(fwd_wrong_rev_right_fusion_right)));
    fprintf('  Fusion WRONG   (alpha too low):     %5d (%.1f%%)  <-- 融合浪费了逆向的正确判断\n', ...
        n_wasted, n_wasted/n_fwd_wrong_rev_right*100);
    fprintf('    -> alpha=%.3f, SG=%.3f, qF=%.3f\n', ...
        mean(all_alpha(fwd_wrong_rev_right_fusion_wrong)), ...
        mean(all_SG(fwd_wrong_rev_right_fusion_wrong)), ...
        mean(all_quality_F(fwd_wrong_rev_right_fusion_wrong)));
end

% Also check: when forward WRONG and reverse WRONG (both wrong), what does fusion do?
fwd_wrong_rev_wrong_fusion_right = fwd_wrong_rev_wrong & fusion_right;
n_both_wrong_fusion_right = sum(fwd_wrong_rev_wrong_fusion_right);
fprintf('\n--- When BOTH forward and reverse are WRONG ---\n');
fprintf('Total: %d\n', n_fwd_wrong_rev_wrong);
fprintf('  Fusion still CORRECT (lucky): %d (%.1f%%)\n', ...
    n_both_wrong_fusion_right, n_both_wrong_fusion_right/n_fwd_wrong_rev_wrong*100);

% 正向错+逆向对 vs 正向对+逆向错 -> alpha 应该分别高/低
fprintf('\n--- Separability: FwdWrong+RevRight vs FwdRight+RevWrong ---\n');
mask_need_high = fwd_wrong_rev_right;          % need alpha HIGH (139 cases)
mask_need_low  = fwd_right & rev_wrong & fusion_right;  % already OK
mask_broken    = fwd_right & rev_wrong & fusion_wrong;   % need alpha LOW (548 cases)

fprintf('%-30s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n', 'Group', 'N', 'SG', 'SG_R', 'qF', 'alpha', 'resRatio', 'nnFwd', 'nnRev', 'nnEnt');
fprintf('%-30s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n', '-----', '-', '--', '----', '--', '-----', '-------', '-----', '-----', '-----');
if sum(mask_need_high) > 0
    fprintf('%-30s %8d %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n', 'Need HIGH alpha (FwdX,RevOK)', ...
        sum(mask_need_high), mean(all_SG(mask_need_high)), mean(all_SG_R(mask_need_high)), ...
        mean(all_quality_F(mask_need_high)), mean(all_alpha(mask_need_high)), ...
        mean(all_res_ratio(mask_need_high)), mean(all_nn_fwd(mask_need_high)), ...
        mean(all_nn_rev(mask_need_high)), mean(all_nn_ent(mask_need_high)));
end
if sum(mask_broken) > 0
    fprintf('%-30s %8d %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n', 'Need LOW alpha (FwdOK,RevX)', ...
        sum(mask_broken), mean(all_SG(mask_broken)), mean(all_SG_R(mask_broken)), ...
        mean(all_quality_F(mask_broken)), mean(all_alpha(mask_broken)), ...
        mean(all_res_ratio(mask_broken)), mean(all_nn_fwd(mask_broken)), ...
        mean(all_nn_rev(mask_broken)), mean(all_nn_ent(mask_broken)));
end
rev_right_all = (all_pred_rev == all_truth);
rev_wrong_all = (all_pred_rev ~= all_truth);
fprintf('%-30s %8d %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n', 'Reverse RIGHT (all)', ...
    sum(rev_right_all), mean(all_SG(rev_right_all)), mean(all_SG_R(rev_right_all)), ...
    mean(all_quality_F(rev_right_all)), mean(all_alpha(rev_right_all)), ...
    mean(all_res_ratio(rev_right_all)), mean(all_nn_fwd(rev_right_all)), ...
    mean(all_nn_rev(rev_right_all)), mean(all_nn_ent(rev_right_all)));
fprintf('%-30s %8d %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n', 'Reverse WRONG (all)', ...
    sum(rev_wrong_all), mean(all_SG(rev_wrong_all)), mean(all_SG_R(rev_wrong_all)), ...
    mean(all_quality_F(rev_wrong_all)), mean(all_alpha(rev_wrong_all)), ...
    mean(all_res_ratio(rev_wrong_all)), mean(all_nn_fwd(rev_wrong_all)), ...
    mean(all_nn_rev(rev_wrong_all)), mean(all_nn_ent(rev_wrong_all)));

% Residual ratio distribution analysis
fprintf('\n--- Residual Ratio: RF(pred_R)/RF(pred_F) ---\n');
fprintf('  Ratio < 1 = reverse class reconstructs BETTER than forward class\n');
fprintf('  Ratio > 1 = forward class reconstructs BETTER than reverse class\n');
fprintf('  FwdWrong+RevRight (need high alpha): mean=%.3f, frac<1=%.1f%%\n', ...
    mean(all_res_ratio(mask_need_high)), ...
    sum(all_res_ratio(mask_need_high) < 1) / sum(mask_need_high) * 100);
fprintf('  FwdRight+RevWrong (need low alpha):  mean=%.3f, frac<1=%.1f%%\n', ...
    mean(all_res_ratio(mask_broken)), ...
    sum(all_res_ratio(mask_broken) < 1) / sum(mask_broken) * 100);
fprintf('  Disagree, FwdRight+RevWrong+FusionOK (alpha fine): mean=%.3f, frac<1=%.1f%%\n', ...
    mean(all_res_ratio(mask_need_low)), ...
    sum(all_res_ratio(mask_need_low) < 1) / max(1,sum(mask_need_low)) * 100);

% k-NN neighborhood analysis for disagreement cases
disagree = (all_pred_fwd ~= all_pred_rev);
fprintf('\n--- k-NN (k=5) Neighborhood Consistency ---\n');
fprintf('  All disagree cases (N=%d):\n', sum(disagree));
fprintf('    nnFwd=%.3f, nnRev=%.3f, nnEnt=%.3f\n', ...
    mean(all_nn_fwd(disagree)), mean(all_nn_rev(disagree)), mean(all_nn_ent(disagree)));
fprintf('  Need HIGH alpha (FwdX,RevOK): nnFwd=%.3f, nnRev=%.3f, nnRev>nnFwd=%.1f%%\n', ...
    mean(all_nn_fwd(mask_need_high)), mean(all_nn_rev(mask_need_high)), ...
    sum(all_nn_rev(mask_need_high) > all_nn_fwd(mask_need_high)) / sum(mask_need_high) * 100);
fprintf('  Need LOW alpha (FwdOK,RevX):   nnFwd=%.3f, nnRev=%.3f, nnFwd>nnRev=%.1f%%\n', ...
    mean(all_nn_fwd(mask_broken)), mean(all_nn_rev(mask_broken)), ...
    sum(all_nn_fwd(mask_broken) > all_nn_rev(mask_broken)) / sum(mask_broken) * 100);

%% =============================================================
% 交叉概率分析 — PF给逆向类多少分，PR给正向类多少分
%% =============================================================
fprintf('\n--- Cross-Probability Analysis ---\n');
fprintf('  PFfwd = PF at forward-predicted class\n');
fprintf('  PFrev = PF at reverse-predicted class (forward considers it)\n');
fprintf('  PRfwd = PR at forward-predicted class (reverse considers it)\n');
fprintf('  PRbi  = PR bimodality (gap12/gap23, high=2-class competition)\n\n');
fprintf('%-26s %6s %6s %6s %6s\n', 'Group', 'PFfwd', 'PFrev', 'PRfwd', 'PRbi');
fprintf('%-26s %6s %6s %6s %6s\n', '----', '-----', '-----', '-----', '----');
if sum(mask_need_high) > 0
    fprintf('%-26s %6.3f %6.3f %6.3f %6.3f\n', 'Need HIGH (FwdX,RevOK)', ...
        mean(all_PF_fwd(mask_need_high)), mean(all_PF_rev(mask_need_high)), ...
        mean(all_PR_fwd(mask_need_high)), mean(all_PR_bi(mask_need_high)));
end
if sum(mask_broken) > 0
    fprintf('%-26s %6.3f %6.3f %6.3f %6.3f\n', 'Need LOW (FwdOK,RevX)', ...
        mean(all_PF_fwd(mask_broken)), mean(all_PF_rev(mask_broken)), ...
        mean(all_PR_fwd(mask_broken)), mean(all_PR_bi(mask_broken)));
end
fprintf('%-26s %6.3f %6.3f %6.3f %6.3f\n', 'Reverse RIGHT (all)', ...
    mean(all_PF_fwd(rev_right_all)), mean(all_PF_rev(rev_right_all)), ...
    mean(all_PR_fwd(rev_right_all)), mean(all_PR_bi(rev_right_all)));
fprintf('%-26s %6.3f %6.3f %6.3f %6.3f\n', 'Reverse WRONG (all)', ...
    mean(all_PF_fwd(rev_wrong_all)), mean(all_PF_rev(rev_wrong_all)), ...
    mean(all_PR_fwd(rev_wrong_all)), mean(all_PR_bi(rev_wrong_all)));

%% =============================================================
% 8. 相关性分析
fprintf('\n--- Correlations ---\n');
fprintf('corr(SG, quality_F)      = %+.4f\n', corr(all_SG(:), all_quality_F(:)));
fprintf('corr(SG, quality_contrast)= %+.4f\n', corr(all_SG(:), all_quality_contrast(:)));
fprintf('corr(SG, quality_abs)    = %+.4f\n', corr(all_SG(:), all_quality_abs(:)));

%% =============================================================
% 9. 总体准确率
%% =============================================================
fprintf('\n%-12s %8d %10.4f %10.4f %10.4f\n', ...
    'Overall', total, ...
    sum(all_pred_fwd == all_truth) / total, ...
    sum(all_pred_rev == all_truth) / total, ...
    sum(all_pred_fusion == all_truth) / total);

%% =============================================================
% 10. 前20个样本详细展示
%% =============================================================
fprintf('\n--- Per-sample detail (first 20) ---\n');
fprintf('Sample  Alpha    SG   qF  qContrast  qAbs  Fwd  Rev  Fusion  Truth\n');
fprintf('------  -----  ----  ---  ---------  ----  ---  ---  ------  -----\n');
for i = 1:min(20, total)
    fprintf('%6d  %5.3f  %4.2f %4.2f   %6.3f   %4.2f  %3s  %3s   %3s     %3d\n', ...
        i, all_alpha(i), all_SG(i), all_quality_F(i), ...
        all_quality_contrast(i), all_quality_abs(i), ...
        check(all_pred_fwd(i), all_truth(i)), ...
        check(all_pred_rev(i), all_truth(i)), ...
        check(all_pred_fusion(i), all_truth(i)), ...
        all_truth(i));
end

%% =============================================================
% 11. 验证标签校准分析 — alpha 还能不能更好？
%% =============================================================
disagree = (all_pred_fwd ~= all_pred_rev);
% 用门控前的 raw alpha（去掉后两个 sigmoid 之后的纯双低 alpha）
alpha_raw = (1 - all_SG) .* (1 - all_quality_F);

fprintf('\n=== Calibration: alpha vs actual reverse accuracy ===\n');
fprintf('Only prediction-disagree samples (N=%d)\n\n', sum(disagree));

nBins = 10;
bin_edges = linspace(0, 1, nBins+1);
fprintf('%-16s %8s %8s %10s %10s %10s\n', ...
    'alpha_raw bin', 'Count', 'RevAcc', 'optAlpha', 'curAlpha', 'gap');
fprintf('%-16s %8s %8s %10s %10s %10s\n', ...
    '-------------', '------', '------', '--------', '--------', '----');

for b = 1:nBins
    lo = bin_edges(b);
    hi = bin_edges(b+1);
    mask = disagree & (alpha_raw >= lo & alpha_raw < hi);
    if b == nBins, mask = disagree & (alpha_raw >= lo & alpha_raw <= hi); end
    n_bin = sum(mask);
    if n_bin >= 5
        rev_acc = sum(all_pred_rev(mask) == all_truth(mask)) / n_bin;
        cur_alpha = mean(all_alpha(mask));
        fprintf('[%4.2f, %4.2f)   %8d %8.3f %10.3f %10.3f %10.3f\n', ...
            lo, hi, n_bin, rev_acc, rev_acc, cur_alpha, rev_acc - cur_alpha);
    else
        fprintf('[%4.2f, %4.2f)   %8d %8s %10s %10s %10s\n', ...
            lo, hi, n_bin, '(skip)', '-', '-', '-');
    end
end

% 整体校准曲线：最优 alpha vs 实际 alpha
fprintf('\n--- Summary ---\n');
fprintf('Perfect calibration: curAlpha ≈ RevAcc in every bin.\n');
fprintf('If curAlpha << RevAcc → alpha too low,  losing fixes.\n');
fprintf('If curAlpha >> RevAcc → alpha too high, causing broken.\n\n');

%% =============================================================
% 11b. 逻辑回归 — 多维信号联合校准
%% =============================================================
if sum(disagree) > 50
    % 特征矩阵: 预测不一致样本的信号
    X_feat = [alpha_raw(disagree)', all_res_ratio(disagree)', ...
              all_PR_margin(disagree)', all_PR_bi(disagree)', ...
              all_SG_R(disagree)', all_PF_rev(disagree)'];
    y_true = (all_pred_rev(disagree) == all_truth(disagree))';  % 逆向是否正确

    % 标准化
    X_mean = mean(X_feat, 1);
    X_std  = std(X_feat, 0, 1);
    X_norm = (X_feat - X_mean) ./ (X_std + eps);

    % 5-fold 交叉验证逻辑回归
    n_dis = sum(disagree);
    cv_idx = crossvalind('Kfold', n_dis, 5);
    pred_prob = zeros(n_dis, 1);

    fprintf('\n=== Logistic Regression: 5-fold CV on disagree samples ===\n');
    fprintf('Features: alpha_raw, resRatio, PR_margin, PR_bi, SG_R, PF_rev\n\n');
    fprintf('Fold  Ntrain  Ntest   AUC     Acc(median)  Acc(optThr)\n');
    fprintf('----  ------  -----   ---     -----------  ----------\n');

    aucs = zeros(5,1);
    for f = 1:5
        tr_idx = (cv_idx ~= f);
        te_idx = (cv_idx == f);
        X_tr = X_norm(tr_idx, :);
        y_tr = y_true(tr_idx);
        X_te = X_norm(te_idx, :);
        y_te = y_true(te_idx);

        % 逻辑回归
        b = glmfit(X_tr, y_tr, 'binomial', 'link', 'logit');
        pred_prob(te_idx) = glmval(b, X_te, 'logit');

        % AUC
        [~,~,~,auc] = perfcurve(y_te, pred_prob(te_idx), 1);
        aucs(f) = auc;

        % 用中位数做阈值
        med_acc = mean((pred_prob(te_idx) > median(pred_prob(te_idx))) == y_te);
        % 最优阈值 (0.5)
        opt_acc = mean((pred_prob(te_idx) > 0.5) == y_te);

        fprintf('  %d    %5d  %5d   %.3f    %.4f       %.4f\n', ...
            f, sum(tr_idx), sum(te_idx), auc, med_acc, opt_acc);
    end
    fprintf('\n  Mean AUC: %.4f\n', mean(aucs));

    % 全量拟合看系数
    b_all = glmfit(X_norm, y_true, 'binomial', 'link', 'logit');
    fprintf('\n  Logistic coefficients (normalized features):\n');
    feat_names = {'Intercept','alpha_raw','resRatio','PR_margin','PR_bi','SG_R','PF_rev'};
    for i = 1:length(b_all)
        fprintf('    %-12s: %+.4f\n', feat_names{i}, b_all(i));
    end

    % 用逻辑回归预测概率 vs 实际逆向准确率
    fprintf('\n  LR predicted prob vs actual RevAcc (5 equal bins):\n');
    [~, sort_idx] = sort(pred_prob);
    nBin = 5;
    bin_size = floor(n_dis / nBin);
    fprintf('  %-10s %8s %8s\n', 'PredProb', 'RevAcc', 'Count');
    for b = 1:nBin
        lo = (b-1)*bin_size + 1;
        hi = min(b*bin_size, n_dis);
        idx = sort_idx(lo:hi);
        fprintf('  [%5.3f,%5.3f] %8.3f %8d\n', ...
            min(pred_prob(idx)), max(pred_prob(idx)), mean(y_true(idx)), length(idx));
    end

    % 与手工 alpha 对比
    fprintf('\n  --- LR alpha vs Handcrafted alpha ---\n');
    fprintf('  Corr(LR_alpha, hand_alpha) = %.4f\n', corr(pred_prob, all_alpha(disagree)'));
    fprintf('  Mean LR_alpha = %.4f, Mean hand_alpha = %.4f\n', mean(pred_prob), mean(all_alpha(disagree)));
end

% 按信号维度分箱 — 看各信号的独立区分力
fprintf('=== Per-signal calibration ===\n');
signals = {'1-SG', '1-qF', 'resRatio', 'PR_margin'};
sig_vals = {1-all_SG, 1-all_quality_F, all_res_ratio, all_PR_margin};
sig_sign = {'lo=bad', 'lo=bad', 'hi=bad', 'lo=bad'};  % which end means unreliable

for s = 1:length(signals)
    fprintf('\n--- %s (%s) ---\n', signals{s}, sig_sign{s});
    vals = sig_vals{s};
    v_bins = linspace(prctile(vals(disagree), 5), prctile(vals(disagree), 95), 6);
    v_bins(1) = min(vals(disagree)); v_bins(end) = max(vals(disagree));
    fprintf('  %-14s %8s %8s\n', 'Bin', 'Count', 'RevAcc');
    for b = 1:length(v_bins)-1
        mask = disagree & (vals >= v_bins(b) & vals < v_bins(b+1));
        if b == length(v_bins)-1, mask = disagree & (vals >= v_bins(b) & vals <= v_bins(b+1)); end
        n_bin = sum(mask);
        if n_bin >= 10
            rev_acc = sum(all_pred_rev(mask) == all_truth(mask)) / n_bin;
            fprintf('  [%6.3f,%6.3f] %8d %8.3f\n', v_bins(b), v_bins(b+1), n_bin, rev_acc);
        end
    end
end

%% =============================================================
% 12. 从 D 矩阵挖掘更好的分类特征
%% =============================================================
fprintf('\n=== Alternative D-based classifiers ===\n');
S_saved = []; T_saved = []; S_label_saved = []; T_label_saved = [];
D_saved = []; classes_saved = []; nt_saved = 0;

rng(1);
load('yaleborigin.mat');
data = dataset;
Kd = 15; number_d = 64; nTrain_d = 8; nt_d = nTrain_d;

train_cell = cell(Kd,1); test_cell = cell(Kd,1);
for c = 1:Kd
    perm = randperm(number_d);
    train_cell{c} = perm(1:nTrain_d);
    test_cell{c} = perm(nTrain_d+1:end);
end
[S1, S_label1] = build_set(data, train_cell, number_d);
[T1, T_label1] = build_set(data, test_cell, number_d);
[~, ~, out1] = DPGSR_Fusion(S1, T1, S_label1, T_label1, nt_d, 1e-2, 1e-2, 'adaptive');
D1 = out1.D; classes1 = out1.classes; K1 = length(classes1);
M1 = size(T1,2); N1 = size(S1,2);

% Current PR accuracy (baseline)
[~, pred_pr] = max(out1.PR, [], 1);
acc_pr = sum(classes1(pred_pr) == T_label1) / M1;
fprintf('  Current PR (L2-norm, softmax):    acc=%.4f\n', acc_pr);

% Alt 1: L1-norm per class (no softmax, raw activation)
score_L1 = zeros(K1, M1);
for j = 1:M1
    for kk = 1:K1
        idx = (S_label1 == classes1(kk));
        score_L1(kk,j) = norm(D1(j, idx), 1) / sqrt(sum(idx));
    end
end
[~, pred_L1] = max(score_L1, [], 1);
acc_L1 = sum(classes1(pred_L1) == T_label1) / M1;
fprintf('  D-L1 norm per class:              acc=%.4f\n', acc_L1);

% Alt 2: Raw sum (not norm) per class
score_sum = zeros(K1, M1);
for j = 1:M1
    for kk = 1:K1
        idx = (S_label1 == classes1(kk));
        score_sum(kk,j) = sum(abs(D1(j, idx))) / sqrt(sum(idx));
    end
end
[~, pred_sum] = max(score_sum, [], 1);
acc_sum = sum(classes1(pred_sum) == T_label1) / M1;
fprintf('  D-abs-sum per class:              acc=%.4f\n', acc_sum);

% Alt 3: Top-K D coefficients voting (sparse nearest-neighbor in D-space)
% For each test sample j, find the top-10 training samples by D(j,:), vote
topK = 10;
pred_topK = zeros(1, M1);
for j = 1:M1
    [~, sorted_idx] = sort(abs(D1(j,:)), 'descend');
    top_labels = S_label1(sorted_idx(1:topK));
    pred_topK(j) = mode(top_labels);
end
acc_topK = sum(pred_topK == T_label1) / M1;
fprintf('  D top-%d training samples voting:  acc=%.4f\n', topK, acc_topK);

% Alt 4: D-based k-NN among test samples
% For each test sample j, find k nearest neighbors in D-row space, vote by forward prediction
[~, pred_fwd1] = max(out1.PF, [], 1);
fwd_labels = classes1(pred_fwd1);
k_nn_d = 10;
pred_dnn = zeros(1, M1);
for j = 1:M1
    dists = zeros(1, M1);
    for jj = 1:M1
        dists(jj) = norm(D1(j,:) - D1(jj,:));
    end
    dists(j) = inf;  % exclude self
    [~, sorted_idx] = sort(dists);
    nn_labels = fwd_labels(sorted_idx(1:k_nn_d));
    pred_dnn(j) = mode(nn_labels);
end
acc_dnn = sum(pred_dnn == T_label1) / M1;
fprintf('  D-space k-NN (k=%d, fwd labels):   acc=%.4f\n', k_nn_d, acc_dnn);

% Alt 5: D + Forward ensemble (stacking)
% Combine PF with D-L1 score, simple average
score_ens = out1.PF + score_L1 ./ (max(score_L1,[],1) + eps);
[~, pred_ens] = max(score_ens, [], 1);
acc_ens = sum(classes1(pred_ens) == T_label1) / M1;
fprintf('  PF + D-L1 ensemble:               acc=%.4f\n', acc_ens);

% Summary
fprintf('\n  --- Summary ---\n');
acc_fwd1 = sum(classes1(pred_fwd1)==T_label1)/M1;
fprintf('  Forward (PF):      %.4f\n', acc_fwd1);
fprintf('  Reverse (PR):      %.4f  (current)\n', acc_pr);
fprintf('  Best D-alt:        %.4f  (PF+D-L1 ensemble)\n', acc_ens);
fprintf('  D-space k-NN:      %.4f  (most independent signal)\n', acc_dnn);

% === Key analysis: ensemble vs PR on forward errors ===
fwd_wrong_1 = (classes1(pred_fwd1) ~= T_label1);
n_fwd_wrong_1 = sum(fwd_wrong_1);
fprintf('\n  --- On %d forward errors ---\n', n_fwd_wrong_1);
fprintf('  PR correct:      %d (%.1f%%)\n', ...
    sum(fwd_wrong_1 & (classes1(pred_pr)==T_label1)), ...
    sum(fwd_wrong_1 & (classes1(pred_pr)==T_label1))/n_fwd_wrong_1*100);
fprintf('  D-L1 correct:    %d (%.1f%%)\n', ...
    sum(fwd_wrong_1 & (classes1(pred_L1)==T_label1)), ...
    sum(fwd_wrong_1 & (classes1(pred_L1)==T_label1))/n_fwd_wrong_1*100);
fprintf('  Ensemble correct:%d (%.1f%%)\n', ...
    sum(fwd_wrong_1 & (classes1(pred_ens)==T_label1)), ...
    sum(fwd_wrong_1 & (classes1(pred_ens)==T_label1))/n_fwd_wrong_1*100);

% Check: does ensemble agree with PR on correct forward samples?
fwd_right_1 = (classes1(pred_fwd1) == T_label1);
fprintf('\n  --- On %d forward correct samples ---\n', sum(fwd_right_1));
fprintf('  PR wrong:        %d\n', sum(fwd_right_1 & (classes1(pred_pr)~=T_label1)));
fprintf('  D-L1 wrong:      %d\n', sum(fwd_right_1 & (classes1(pred_L1)~=T_label1)));
fprintf('  Ensemble wrong:  %d\n', sum(fwd_right_1 & (classes1(pred_ens)~=T_label1)));

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

function s = check(pred, truth)
if pred == truth; s = 'OK'; else; s = 'X'; end
end

function [pred_label, acc] = DPGSR_Pipeline(S, T, S_label, T_label, lambda1, lambda2, n_iter, verif_thresh, pr_margin_thresh)
% DPGSR_Pipeline — 统一管道：单次/自训练/融合收尾
%
% n_iter = 1:  单次 DP-GSR + 硬开关融合 (无自训练)
% n_iter > 1:  迭代自训练最多 n_iter 轮, 收敛后融合收尾
% n_iter = Inf: 自训练至收敛

if nargin < 7 || isempty(n_iter); n_iter = 1; end
if nargin < 8 || isempty(verif_thresh); verif_thresh = 0.65; end
if nargin < 9 || isempty(pr_margin_thresh); pr_margin_thresh = 0.05; end

S_label = S_label(:)'; T_label = T_label(:)';
classes = unique(S_label, 'stable'); K = length(classes);
M = size(T,2);
unresolved = true(1,M);
pseudo_label = zeros(1,M);

% Expand training set
S_cur = S; Sl_cur = S_label;

% ---- Phase 1: Self-training (skip if n_iter==1) ----
if n_iter > 1
    for rnd = 1:n_iter
        [~,~,o] = DPGSR_Fusion(S_cur, T(:,unresolved), Sl_cur, T_label(unresolved), ...
            Sl_cur, lambda1, lambda2, 'adaptive', [], [], verif_thresh, pr_margin_thresh);
        Z = o.D; [~,pf] = max(o.PF,[],1);
        uri = find(unresolved); nt_u = sum(unresolved);

        verif = zeros(1,nt_u);
        for j = 1:nt_u
            verif(j) = norm(Z(j, Sl_cur==o.classes(pf(j))),2) / (norm(Z(j,:),2)+1e-12);
        end
        new_conf = (verif > verif_thresh);
        n_add = sum(new_conf);
        if n_add == 0; break; end  % converged early

        for jj = 1:nt_u
            if new_conf(jj)
                j = uri(jj);
                S_cur = [S_cur, T(:,j)];
                Sl_cur = [Sl_cur, o.classes(pf(jj))];
                unresolved(j) = false;
                pseudo_label(j) = o.classes(pf(jj));
            end
        end
    end
end

% ---- Phase 2: Final pass with ALL test as reverse dict ----
[~,~,o] = DPGSR_Fusion(S_cur, T, Sl_cur, T_label, Sl_cur, lambda1, lambda2, ...
    'adaptive', [], [], verif_thresh, pr_margin_thresh);
Z = o.D; [~,pf] = max(o.PF,[],1); [~,pr] = max(o.PR,[],1);

% Predictions for resolved samples
for j = 1:M
    if ~unresolved(j); continue; end
    uri = find(unresolved);
    for jj = 1:sum(unresolved)
        if unresolved(uri(jj))
            pseudo_label(uri(jj)) = o.classes(pf(jj));
        end
    end
    break;
end

% Hard-gate fusion on unresolved
uri = find(unresolved);
for jj = uri
    kF = o.classes(pf(jj));
    vf = norm(Z(jj, Sl_cur==kF),2) / (norm(Z(jj,:),2)+1e-12);
    PRs = sort(o.PR(:,jj), 'descend'); PRm = PRs(1)-PRs(2);
    if vf > verif_thresh
        pseudo_label(jj) = kF;
    elseif PRm > pr_margin_thresh
        pseudo_label(jj) = o.classes(pr(jj));
    else
        pseudo_label(jj) = kF;
    end
end

pred_label = pseudo_label;
if ~isempty(T_label)
    acc = sum(pred_label == T_label) / M;
else
    acc = NaN;
end
end

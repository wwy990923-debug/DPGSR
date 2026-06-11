function [W, D, out] = JointConstrained(S, T, nt, gamma1, gamma2)
% JointConstrained
%
% Jointly solves forward and reverse group-sparse optimization:
%
%   Forward:  min_W  ||S*W - T||_F^2 + gamma1 * ||W||_{group}
%   Reverse:  min_D  ||T*D - S||_F^2 + gamma2 * ||D||_{row-group}
%
% where:
%   group norm on W groups nt consecutive rows (same-class training samples)
%   row-group norm on D groups nt consecutive columns (same-class training samples)
%
% Solved via ADMM with auxiliary variables:
%   Forward:  min ||S*W-T||^2 + gamma1*||F||_{group}  s.t. W = F
%   Reverse:  min ||T*D-S||^2 + gamma2*||G||_{row-group}  s.t. D = G
%
% The two subproblems alternate within a single ADMM loop (joint optimization).
%
% Inputs:
%   S      - training data, d x N (N = total training samples)
%   T      - test data, d x M
%   nt     - number of training samples per class (must divide N)
%   gamma1 - forward regularization weight (same role as lambda1 in Forward.m)
%   gamma2 - reverse regularization weight (same role as lambda2 in Reverse.m)
%
% Outputs:
%   W      - forward coefficient matrix, N x M
%   D      - reverse coefficient matrix, M x N
%   out    - struct with convergence history

if nargin < 4 || isempty(gamma1), gamma1 = 1e-2; end
if nargin < 5 || isempty(gamma2), gamma2 = 1e-2; end

tol = 1e-4;
maxIter = 500;

[d, N] = size(S);  % N = number of training samples
[d, M] = size(T);  % M = number of test samples

if mod(N, nt) ~= 0
    error('Number of training samples N must be divisible by nt.');
end

rho = 1.2;
max_mu = 1e20;
mu = 1.1;

%% ============================================================
% Initialize forward variables
% W: NxM,  F: NxM,  Y1: NxM
%% ============================================================
W  = zeros(N, M);
F  = zeros(N, M);
Y1 = zeros(N, M);

%% ============================================================
% Initialize reverse variables
% D: MxN,  G: MxN,  Z1: MxN
%% ============================================================
D  = zeros(M, N);
G  = zeros(M, N);
Z1 = zeros(M, N);

%% ============================================================
% Precompute constant terms
%% ============================================================
STS2 = 2 * (S' * S);   % 2*S'S — constant part of W-update matrix
STT2 = 2 * (S' * T);   % 2*S'T — constant part of W-update RHS
TTS2 = 2 * (T' * T);   % 2*T'T — constant part of D-update matrix
TTSr = 2 * (T' * S);   % 2*T'S — constant part of D-update RHS

%% ============================================================
% Main ADMM loop
%% ============================================================
iter = 0;
stop_hist = zeros(maxIter, 1);
obj_hist = zeros(maxIter, 1);

while iter < maxIter
    iter = iter + 1;

    %% ==================== FORWARD ====================

    % Update F: group L21 shrinkage
    tempF = W - Y1 / mu;
    PF = reshape(tempF, nt, (N/nt) * M);
    JF = solve_L21(PF, gamma1 / mu);
    F = reshape(JF, N, M);

    % Update W: least squares with ADMM penalty
    M_fwd = STS2 + mu * eye(N);
    RHS_fwd = STT2 + mu * (F + Y1 / mu);
    W = M_fwd \ RHS_fwd;

    % Constraint violation and multiplier update
    leq1_F = F - W;
    Y1 = Y1 + mu * leq1_F;

    %% ==================== REVERSE ====================

    % Update G: row-group L21 shrinkage (column-group on D')
    tempR = D - Z1 / mu;
    tempRT = tempR';
    PR = reshape(tempRT, nt, (N/nt) * M);
    JR = solve_L21(PR, gamma2 / mu);
    G = reshape(JR, N, M)';

    % Update D: least squares with ADMM penalty
    M_rev = TTS2 + mu * eye(M);
    RHS_rev = TTSr + mu * (G + Z1 / mu);
    D = M_rev \ RHS_rev;

    % Constraint violation and multiplier update
    leq1_R = G - D;
    Z1 = Z1 + mu * leq1_R;

    %% ==================== CONVERGENCE ====================

    stopF = max(abs(leq1_F(:)));
    stopR = max(abs(leq1_R(:)));
    stopC = max(stopF, stopR);
    stop_hist(iter) = stopC;

    % Joint objective value
    recF_err = norm(S*W - T, 'fro')^2;
    QF = reshape(W, nt, (N/nt) * M);
    regF = gamma1 * sum(sqrt(sum(QF.^2, 1)));

    recR_err = norm(T*D - S, 'fro')^2;
    QR = reshape(D', nt, (N/nt) * M);
    regR = gamma2 * sum(sqrt(sum(QR.^2, 1)));

    obj_hist(iter) = recF_err + recR_err + regF + regR;

    if iter == 1 || mod(iter, 50) == 0 || stopC < tol
        fprintf('iter %d, mu=%.1e, |W-F|=%.3e, |D-G|=%.3e, obj=%.4e, rk(W)=%d\n', ...
            iter, mu, stopF, stopR, obj_hist(iter), ...
            rank(W, 1e-4*norm(W,2)));
    end

    if stopC < tol
        break;
    else
        mu = min(max_mu, mu * rho);
    end
end

%% ============================================================
% Output
%% ============================================================
out.obj = obj_hist(1:iter);
out.stop_hist = stop_hist(1:iter);
out.iter = iter;
out.stopC = stopC;

end


%% ============================================================
% Helper: column-wise L21 shrinkage (prox of group L21 norm)
%% ============================================================
function [E] = solve_L21(M, lambda)
n = size(M, 2);
E = M;
for i = 1:n
    E(:,i) = solve_l2(M(:,i), lambda);
end
end

function [x] = solve_l2(w, lambda)
nw = norm(w, 2);
if nw > lambda
    x = (1 - lambda/nw) * w;
else
    x = zeros(size(w));
end
end

clc
clear
close all
warning off

%% ============================================================
%  DP-GSR on YaleB: no validation set
%  Sweep lambda1 and lambda2 outside
%
%  Need files in the same path:
%    1) Forward.m
%    2) Reverse.m
%    3) DPGSR_Fusion.m
%% ============================================================

rng(1);   % for reproducibility

%% Load data
filename1 = 'yaleborigin';
load(strcat(filename1,'.mat'))

data = dataset;

%% Optional path
addpath('D:\my_SRC-main\fista_lasso.m')

%% Output TXT file
filename = 'dpgsr_yaleb_no_validation_lambda_sweep';
fid = fopen(strcat(filename,'.txt'),'a+');

dat = date;
fprintf(fid, '\r\n\r\n==============================\r\n');
fprintf(fid, 'Date: %s\r\n', dat);
fprintf(fid, 'DP-GSR no-validation lambda sweep\r\n');
fprintf(fid, '==============================\r\n');

%% Dataset settings
nSet_list = 38;             % number of classes
number = 64;                % samples per class
number_train_list = [8];    % training samples per class
nmax = 1;                  % repeated random splits

%% Lambda grid
lambda_grid = [1e-2, 1e-1, 1, 10];

%% Whether to use random split or fixed split
useRandomSplit = true;

%% Main loops
for k1 = 1:length(nSet_list)

    K = nSet_list(k1);
    dataset = data(:, 1:K*number);

    for n1 = 1:length(number_train_list)

        nTrain = number_train_list(n1);

        fprintf('\n==============================\n');
        fprintf('Classes = %d, train/class = %d\n', K, nTrain);
        fprintf('==============================\n');

        fprintf(fid, '\r\nClasses = %d, train/class = %d\r\n', K, nTrain);

        %% ============================================================
        % 1. Pre-generate splits
        %    这样所有 lambda1/lambda2 使用同一批随机划分，比较更公平
        %% ============================================================

        all_train_cell = cell(nmax, 1);
        all_test_cell  = cell(nmax, 1);

        for m5 = 1:nmax

            train_cell = cell(K,1);
            test_cell = cell(K,1);

            for c = 1:K

                if useRandomSplit
                    perm = randperm(number);

                    train_idx = perm(1:nTrain);
                    test_idx  = perm(nTrain+1:end);

                else
                    train_idx = 1:nTrain;
                    test_idx  = nTrain+1:number;
                end

                train_cell{c} = train_idx;
                test_cell{c} = test_idx;
            end

            all_train_cell{m5} = train_cell;
            all_test_cell{m5}  = test_cell;
        end

        %% ============================================================
        % 2. Sweep lambda1 and lambda2
        %% ============================================================

        result_table = [];
        cnt_result = 0;
        alpha_j_all_pairs = cell(length(lambda_grid), length(lambda_grid));

        best_avg_acc = -inf;
        best_lambda1 = lambda_grid(1);
        best_lambda2 = lambda_grid(1);

        for l1 = 1:length(lambda_grid)

            lambda1 = lambda_grid(l1);

            for l2 = 1:length(lambda_grid)

                lambda2 = lambda_grid(l2);

                Bcc = zeros(nmax,1);
                Tim = zeros(nmax,1);
                alpha_j_cell = cell(nmax, 1);

                fprintf('\n--------------------------------\n');
                fprintf('Testing lambda1 = %g, lambda2 = %g\n', lambda1, lambda2);
                fprintf('--------------------------------\n');

                fprintf(fid, '\r\nTesting lambda1 = %g, lambda2 = %g\r\n', lambda1, lambda2);

                for m5 = 1:nmax

                    tic;

                    train_cell = all_train_cell{m5};
                    test_cell  = all_test_cell{m5};

                    nt = length(train_cell{1});

                    %% Build training dictionary S and test set T
                    [S, S_label] = build_set_by_class(dataset, train_cell, number);
                    [T, T_label] = build_set_by_class(dataset, test_cell, number);

                    %% Run DP-GSR
                    [pred_label, bcc, out] = DPGSR_Fusion( ...
                        S, T, S_label, T_label, ...
                        nt, lambda1, lambda2, ...
                        'adaptive' ...
                        );

%                     [pred_label, bcc, out] = DPGSR_Fusion( ...
%                         S, T, S_label, T_label, ...
%                         nt, lambda1, lambda2, ...
%                         'fixed', 0 ...
%                         );


                    confMat = confusionmat(T_label, pred_label);

                    elapsed_time = toc;

                    Bcc(m5) = bcc;
                    Tim(m5) = elapsed_time;
                    alpha_j_cell{m5} = out.alpha_j_true;

                    fprintf('Repeat %d/%d, acc = %.4f, time = %.4f sec\n', ...
                        m5, nmax, bcc, elapsed_time);

                    fprintf(fid, ['Repeat:%d, class:%d, train/class:%d, ' ...
                        'test_acc:%8.6f, lambda1:%g, lambda2:%g, time:%8.6f\r\n'], ...
                        m5, K, nTrain, bcc, lambda1, lambda2, elapsed_time);

                end

                %% Summary for current lambda pair
                avgbcc = mean(Bcc);
                maxbcc = max(Bcc);
                minbcc = min(Bcc);
                stdbcc = std(Bcc);
                avgtim = mean(Tim);

                all_alpha = cell2mat(alpha_j_cell');
                avg_alpha = mean(all_alpha);
                std_alpha = std(all_alpha);

                fprintf('\nSummary for lambda1=%g, lambda2=%g\n', lambda1, lambda2);
                fprintf('avg acc = %.4f\n', avgbcc);
                fprintf('std acc = %.4f\n', stdbcc);
                fprintf('min acc = %.4f\n', minbcc);
                fprintf('max acc = %.4f\n', maxbcc);
                fprintf('avg time = %.4f sec\n', avgtim);
                fprintf('avg alpha_j = %.4f, std alpha_j = %.4f\n', avg_alpha, std_alpha);

                fprintf(fid, ['SUMMARY, class:%d, train/class:%d, ' ...
                    'lambda1:%g, lambda2:%g, ' ...
                    'avgbcc:%8.6f, stdbcc:%8.6f, minbcc:%8.6f, maxbcc:%8.6f, ' ...
                    'avgtim:%8.6f, avg_alpha:%8.6f, std_alpha:%8.6f\r\n'], ...
                    K, nTrain, lambda1, lambda2, ...
                    avgbcc, stdbcc, minbcc, maxbcc, avgtim, avg_alpha, std_alpha);

                cnt_result = cnt_result + 1;
                result_table(cnt_result, :) = [lambda1, lambda2, avgbcc, stdbcc, minbcc, maxbcc, avgtim, avg_alpha, std_alpha];

                if avgbcc > best_avg_acc
                    best_avg_acc = avgbcc;
                    best_lambda1 = lambda1;
                    best_lambda2 = lambda2;
                end

                alpha_j_all_pairs{l1, l2} = alpha_j_cell;

            end
        end

        %% ============================================================
        % 3. Print best lambda pair under no-validation sweep
        %% ============================================================

        fprintf('\n========================================\n');
        fprintf('Best parameter pair on test sweep:\n');
        fprintf('lambda1 = %g, lambda2 = %g, avg acc = %.4f\n', ...
            best_lambda1, best_lambda2, best_avg_acc);
        fprintf('========================================\n');

        fprintf(fid, '\r\nBEST TEST-SWEEP PARAMETER PAIR:\r\n');
        fprintf(fid, 'lambda1:%g, lambda2:%g, avg_acc:%8.6f\r\n', ...
            best_lambda1, best_lambda2, best_avg_acc);

        %% Save result table
        result_filename = sprintf('lambda_sweep_class%d_train%d.mat', K, nTrain);
        save(result_filename, 'result_table', 'lambda_grid', 'best_lambda1', 'best_lambda2', 'best_avg_acc', 'alpha_j_all_pairs');

    end
end

fclose(fid);


%% ============================================================
% Local helper function
%% ============================================================
function [X, label] = build_set_by_class(dataset, index_cell, number_per_class)
% Build data matrix and label vector from per-class indices.
%
% dataset: d x (K*number_per_class)
% index_cell{c}: sample indices selected from class c
%
% Output:
%   X     : d x total_samples
%   label : 1 x total_samples

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
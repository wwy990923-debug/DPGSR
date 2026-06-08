% extract_gabor_features.m — 用 Gabor 滤波器组替代原始像素
function features = extract_gabor_features(dataset, img_h, img_w)
% dataset: d x N, 每列是一张拉直的人脸图像
% img_h, img_w: 原始图像尺寸
% 返回: d_gabor x N, Gabor 特征

N = size(dataset, 2);
features = [];

% Gabor 参数
wavelengths = [4, 8, 14];          % 3 个尺度
orientations = [0, 45, 90, 135];   % 4 个方向
n_features = length(wavelengths) * length(orientations);

% 降采样因子
ds = 6;

for j = 1:N
    % 重塑为图像
    img = reshape(dataset(:,j), img_h, img_w);
    img = double(img);
    img = img / max(img(:));  % 归一化到 [0,1]

    feat_j = [];
    for w = 1:length(wavelengths)
        for o = 1:length(orientations)
            % 构建 Gabor 滤波器
            lambda = wavelengths(w);
            theta = orientations(o) * pi / 180;
            gamma = 0.5;
            sigma = 0.56 * lambda;

            % Gabor 核
            [g, ~] = gabor_fn(sigma, theta, lambda, gamma);
            g_odd = g{1};  % 奇数部分（边缘检测）

            % 卷积
            resp = conv2(img, g_odd, 'same');
            % 降采样
            resp_ds = resp(1:ds:end, 1:ds:end);
            feat_j = [feat_j; resp_ds(:)];
        end
    end
    features = [features, feat_j];
end

% PCA 降维
if size(features,2) > 1 && size(features,1) > 500
    feat_mean = mean(features, 2);
    features = features - feat_mean;
    [coeff, score, ~] = pca(features');
    features = score(:, 1:min(500, size(score,2)-1))';  % 500 x N
end

fprintf('  Gabor features: %d x %d\n', size(features,1), size(features,2));

function [g_odd, g_even] = gabor_fn(sigma, theta, lambda, gamma)
% 生成 Gabor 滤波器（奇偶部分）
sigma_x = sigma;
sigma_y = sigma / gamma;

nstds = 3;
xmax = max(abs(nstds * sigma_x * cos(theta)), abs(nstds * sigma_y * sin(theta)));
xmax = ceil(max(1, xmax));
ymax = max(abs(nstds * sigma_x * sin(theta)), abs(nstds * sigma_y * cos(theta)));
ymax = ceil(max(1, ymax));
xmin = -xmax; ymin = -ymax;
[x, y] = meshgrid(xmin:xmax, ymin:ymax);

x_theta = x * cos(theta) + y * sin(theta);
y_theta = -x * sin(theta) + y * cos(theta);

gb = exp(-0.5 * (x_theta.^2 / sigma_x^2 + y_theta.^2 / sigma_y^2)) .* cos(2 * pi * x_theta / lambda);
g_odd = {gb};
g_even = {exp(-0.5 * (x_theta.^2 / sigma_x^2 + y_theta.^2 / sigma_y^2)) .* sin(2 * pi * x_theta / lambda)};
end
end

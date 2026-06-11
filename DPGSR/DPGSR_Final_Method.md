# DP-GSR 最终方法整理

## 1. 核心定位

当前算法的最终定位是：

```text
正向路径负责分类，逆向路径负责验证。
```

也就是说，逆向路径不再被当作一个独立的强分类器，而是作为正向预测的置信度验证器。原因是逆向表示矩阵 `D` 来自一个全局协同重构问题：

```text
S ≈ T * D
```

其中 `S` 是训练集，`T` 是测试 batch，`D(j,:)` 表示第 `j` 个测试样本作为字典原子，在重构整个训练集时的参与程度。它优化的是全局训练流形重构，而不是单个测试样本的判别分类。因此，直接用 `argmax PR` 分类容易过度解释 `D(j,:)`。

最终使用方式是：

```text
Forward: 给出主预测 k_F
Reverse: 检查 D(j,:) 是否支持 k_F
Fusion: 只有当正向明显不可靠且逆向足够自信时，才允许逆向改判
```

## 2. 正向路径

正向路径用训练字典 `S` 重构测试样本 `T`：

```text
T ≈ S * W
```

优化目标为：

```text
min_W ||S*W - T||_F^2 + lambda1 * sum_j sum_k ||W(Omega_k, j)||_2
```

其中：

```text
W: N x M，正向表示矩阵
Omega_k: 第 k 类训练样本索引
W(Omega_k, j): 第 j 个测试样本在第 k 类训练样本上的系数块
```

正向路径的作用是得到每个测试样本的类别残差：

```text
RF(k,j) = ||T(:,j) - S(:,Omega_k) * W(Omega_k,j)||_2^2
```

然后通过 softmax 得到正向证据：

```text
PF(k|y_j) ∝ exp(-RF(k,j) / tau_F)
```

最终正向预测为：

```text
k_F = argmax_k PF(k|y_j)
```

代码对应：

```text
Forward.m
DPGSR_Fusion.m: forward residual evidence P_F
```

`Forward.m` 现在支持两种分组输入：

```matlab
W = Forward(S, T, nt, lambda1)       % 等类别样本数
W = Forward(S, T, S_label, lambda1)  % 变长类别样本数
```

当输入 `S_label` 时，代码按标签切分 `temp(idx,:)`，分别对每个类别块做 `solve_L21`，因此不再依赖 `reshape(temp, nt, ...)`。

## 3. 逆向路径

逆向路径用测试 batch `T` 重构训练集 `S`：

```text
S ≈ T * D
```

优化目标为：

```text
min_D ||T*D - S||_F^2 + lambda2 * sum_j sum_k ||D(j, Omega_k)||_2
```

其中：

```text
D: M x N，逆向表示矩阵
D(j, Omega_k): 第 j 个测试样本对第 k 类训练样本的逆向贡献
```

这里的组稀疏是行内类别组稀疏：

```text
对每个测试样本 j，按训练类别块 Omega_k 分组
每个 D(j, Omega_k) 做 l2 shrinkage
```

代码对应：

```text
Reverse.m
```

`Reverse.m` 现在也支持：

```matlab
D = Reverse(T, S, nt, lambda2)
D = Reverse(T, S, S_label, lambda2)
```

当输入 `S_label` 时，逆向正则项按真实训练标签分组，因此支持每类训练样本数不同的情况。

## 4. 逆向证据 PR

逆向路径仍然会计算一个按类别聚合的激活强度：

```text
SR(k,j) = ||D(j, Omega_k)||_2 / (sqrt(n_k) * ||S_k||_F)
```

并归一化为：

```text
PR(k|y_j) = SR(k,j) / sum_i SR(i,j)
```

但现在 `PR` 的角色是辅助证据，而不是主分类器。原因是：

```text
PR 反映的是测试样本参与重构训练流形的类别贡献；
它不是直接优化出来的 per-sample discriminative posterior。
```

因此，`argmax PR` 可以作为候选改判方向，但不能无条件覆盖正向。

## 5. 验证分 verif

最终方法中最关键的门控信号是验证分：

```text
verif(j) = ||D(j, Omega_{k_F})||_2 / ||D(j,:)||_2
```

其中 `k_F` 是正向预测类别。

物理含义：

```text
正向预测类在逆向系数 D(j,:) 中拿到了多少能量份额。
```

解释：

```text
verif 高：逆向也支持正向预测，正向可信
verif 低：逆向不支持正向预测，正向可能不可靠
```

这个信号比单独看 `SG`、残差比、PR margin、entropy 更符合当前方法的定位，因为它直接衡量正向和逆向是否一致。

## 6. 硬门控融合策略

当前 `DPGSR_Fusion.m` 采用的是 verification-gated hard fusion：

```text
1. 计算正向预测 k_F = argmax PF
2. 计算逆向预测 k_R = argmax PR
3. 计算 verif(j)
4. 计算 PR_margin = PR_top1 - PR_top2
5. 决策：

   if verif > theta:
       信任正向，alpha = 0
   elseif PR_margin > m0:
       正向验证失败，且逆向足够自信，alpha = 1
   else:
       两边都不够可靠，保守信任正向，alpha = 0
```

最终分数仍写成统一形式：

```text
Score(:,j) = (1 - alpha_j) * PF(:,j) + alpha_j * PR(:,j)
```

但由于 `alpha_j` 是硬开关，所以实际含义是：

```text
alpha = 0: 采用正向
alpha = 1: 采用逆向
```

推荐参数：

```text
theta = 0.65
m0 = 0.05
```

代码对应：

```text
DPGSR_Fusion.m: Verification-Gated Fusion
```

## 7. 自训练扩展

在 PCA-50 等低维空间中，`verif` 对正向是否可靠有明显区分能力。因此可以用它筛选高置信伪标签，做迭代自训练。

流程：

```text
Phase 1: 迭代自训练

repeat:
    1. 用当前训练集 S 跑 DP-GSR
    2. 对每个未解决测试样本计算正向预测 k_F 和 verif
    3. 选择 verif > theta 的样本
    4. 用其正向预测标签作为伪标签加入训练集
    5. 如果没有新样本加入，则停止

Phase 2: 剩余 hard samples 收尾

    1. 保留自训练阶段得到的正向预测
    2. 用全量测试样本作为逆向字典求 D
    3. 对剩余 hard samples 使用 verif + PR_margin 硬门控
```

自训练的核心不是让逆向直接分类，而是：

```text
用逆向验证分筛选正向高置信样本，逐步扩充训练字典。
```

## 8. 为什么不用继续堆 alpha 信号

实验中已经观察到：

```text
残差比、PRbi、PR margin、entropy、SG、kNN 邻域一致性
都能识别部分异常信号；
但在正常置信范围内，无法稳定区分正确和错误。
```

剩余 hard cases 的特点是：

```text
错误类不是异常获胜，而是在正常二类竞争中略胜。
```

因此继续调 alpha 或叠加更多同源 gate，通常只会在 fixed 和 broken 之间做 Pareto trade-off：

```text
减少 broken 的同时，也等比例减少 fixed。
```

这说明当前无监督几何信号已经接近上限。若要进一步提升，需要新的独立信息，例如：

```text
更强特征空间
多特征/多维度投票
监督校准 gate
metric learning 或 class-specific calibration
```

## 9. 特征空间结论

DP-GSR 对特征空间非常敏感。

高维原始像素空间中，正向路径往往已经很强，逆向路径作为验证器的 gap 较小，融合提升有限。

低维 PCA 空间中，正向和逆向更接近平衡：

```text
正向：训练样本少，容易欠完备
逆向：测试 batch 大，重构训练集更稳定
verif：正逆向一致性更明显
```

因此 PCA-50 一类低维空间更适合当前 verification-gated DP-GSR。

## 10. 与原始 DP-GSR 的区别

| 模块 | 原始思路 | 当前最终思路 |
|---|---|---|
| 正向路径 | 主分类证据 | 主分类证据 |
| 逆向路径 | 反向分类/激活证据 | 正向置信度验证器 |
| 融合方式 | 连续 alpha 加权 | verif 硬门控 |
| alpha 来源 | 正向稀疏度 SG | 正向-逆向一致性 verif |
| PR 角色 | 可直接参与分类 | 只在正向验证失败且 PR 自信时改判 |
| 类别不均衡 | 依赖固定 nt | Forward/Reverse 支持 S_label 分组 |
| 扩展策略 | 单次分类 | 可加入 verif 自训练 |

## 11. 当前代码文件

| 文件 | 作用 |
|---|---|
| `DPGSR_Fusion.m` | 单次 DP-GSR 分类与 verif 硬门控融合 |
| `Forward.m` | 正向组稀疏 ADMM，支持 `nt` 或 `S_label` 分组 |
| `Reverse.m` | 逆向行组稀疏 ADMM，支持 `nt` 或 `S_label` 分组 |
| `ReverseSparse.m` | 带额外稀疏项的逆向变体 |
| `test_pipeline.m` | 自训练加剩余样本融合的实验管道 |
| `test_pipeline2.m` | 自训练后只跑逆向收尾的实验管道 |
| `test_verif_final.m` | verif 阈值和特征空间对比实验 |
| `analyze_alpha.m` | alpha、broken/fixed、hard samples 诊断分析 |

## 12. 一句话总结

最终算法不是“正向分类器 + 逆向分类器”的简单加权，而是：

```text
以正向残差分类为主体，
用逆向全局协同重构产生的 verif 分数验证正向是否可信，
只在正向验证失败且逆向自信时允许改判，
并可利用高 verif 样本进行自训练扩充训练字典。
```

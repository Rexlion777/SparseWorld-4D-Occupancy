# MCQM × R8 Read-only Research Audit / 只读研究审计

> Snapshot date: 2026-07-14. The active Robot workspace was only read. No source, checkpoint, artifact, process, branch, or working-tree state was modified or copied into the portfolio branch.

> 快照日期：2026-07-14。对 Robot 正在进行的工作只进行了读取。未修改、执行、复制或提交其源码、checkpoint、产物、进程、分支或工作区状态。

## 1. Research question / 研究问题

R8 causally replaces a failed current camera's four FPN levels with the last valid same-camera FPN. The active research asks whether sparse temporal query memory can provide a more accurate correction while preserving R8's online constraints.

R8 使用上一个有效时刻的同相机 FPN，替换当前故障相机的四层 FPN。当前研究问题是：稀疏时序 Query 记忆能否在不破坏 R8 因果约束的前提下，给出更准确的修正。

## 2. End-to-end algorithm chain / 端到端算法链路

```text
t-1 camera images
  → Backbone + four FPN levels
  → SparseWorld query state {query feature, refine points, confidence, ego pose}
  → strict detached temporal memory
  → ego-motion compensation into the current ego frame
  → projection to failed cameras
  → confidence-weighted bilinear splat
  → four-level Q2F reconstruction or local residual/transport
  → overwrite failed cameras only
  → frozen Occupancy head
  → 0/2/4/6 s semantic occupancy and reliability metrics
```

中文链路：

```text
t-1 六相机图像
  → Backbone + 四层 FPN
  → SparseWorld Query 状态（特征、参考点、置信度、自车位姿）
  → 严格 detach 的历史记忆
  → 自车运动补偿到当前坐标系
  → 投影至故障相机
  → 置信度加权双线性 splat
  → 四层 Q2F 重建，或局部残差/传输
  → 仅写入故障相机
  → 冻结的 Occupancy Head
  → 0/2/4/6 s 语义占据与可靠性评测
```

## 3. Core equations / 核心公式

### 3.1 Ego-motion compensation / 自车运动补偿

For a historical query point \(p_{t-1}\):

\[
p_t = T_{E_t\leftarrow G}\,T_{G\leftarrow E_{t-1}}\,p_{t-1}
    = T_{G\leftarrow E_t}^{-1}T_{G\leftarrow E_{t-1}}p_{t-1}.
\]

这一步把历史 Query 的空间位置对齐到当前自车坐标系，避免把自车运动误认为环境运动。

### 3.2 Camera projection / 相机投影

\[
\tilde p_{i,c}=K_cT_{C_c\leftarrow E_t}p_{t,i},\qquad
u_{i,c}=\tilde x/\tilde z,\quad v_{i,c}=\tilde y/\tilde z.
\]

Only positive-depth, in-frame points enter the valid support mask. / 仅深度为正且落在图像内的点可以进入 support mask。

### 3.3 Confidence-weighted query-to-FPN splat / Query 到 FPN 的加权散射

For FPN level \(l\), query feature \(q_i\), confidence \(a_i\), and bilinear kernel \(k\):

\[
S_l(x)=
\frac{\sum_i a_i^{\gamma}k(x,\pi(p_{t,i}))W_lq_i}
{\sum_i a_i^{\gamma}k(x,\pi(p_{t,i}))+\epsilon}.
\]

The sparse field and its support/weight maps are then decoded into a dense FPN candidate. / 稀疏特征与 support/weight map 共同进入 decoder，生成稠密 FPN 候选。

### 3.4 R8 baseline / R8 基线

\[
F^{R8}_{t,l,c}=
\begin{cases}
F_{t,l,c}, & c\notin\mathcal F_t,\\
F_{t-1,l,c}, & c\in\mathcal F_t,
\end{cases}
\]

where \(\mathcal F_t\) is the declared failed-camera set. Healthy cameras remain bitwise unchanged. / \(\mathcal F_t\) 为当前明确声明的故障相机集合，健康相机保持 bitwise unchanged。

### 3.5 Full Q2F and R8 residual / 完整 Q2F 与 R8 残差

Full reconstruction:

\[
\hat F_{t,l,c}=D_l\!\left(S_l(P_l(Q^{comp}_{t-1}))\right),
\qquad c\in\mathcal F_t.
\]

R8-conditioned residual:

\[
\Delta F_{t,l,c}=D_l(S_l;F^{R8}_{t,l,c}),\qquad
F^{rec}_{t,l,c}=F^{R8}_{t,l,c}+M_{l,c}\odot\Delta F_{t,l,c}.
\]

The zero-initialized output head enforces \(\Delta F=0\) before training, so the new path starts exactly at R8 parity. / 输出头零初始化，保证训练前 \(\Delta F=0\)，新路径从与 R8 完全一致的起点开始。

### 3.6 Distillation and task objectives / 特征蒸馏与任务目标

\[
\mathcal L_{feat}=\operatorname{SmoothL1}(F^{rec},F^{clean})
+\lambda_{cos}\left(1-\cos(F^{rec},F^{clean})\right),
\quad \lambda_{cos}=0.1.
\]

\[
\mathcal L=\mathcal L_{feat}+\mathcal L_{occ},
\]

where \(\mathcal L_{occ}\) is the frozen SparseWorld Occupancy task loss propagated into the added branch. / \(\mathcal L_{occ}\) 是经过冻结 SparseWorld Occupancy Head 回传到新增分支的任务损失。

### 3.7 Gradient diagnosis and PCGrad / 梯度冲突诊断与 PCGrad

\[
\rho=\frac{g_{feat}^{\top}g_{task}}
{\|g_{feat}\|\,\|g_{task}\|}.
\]

If \(g_{feat}^{\top}g_{task}<0\), the implementation projects the task gradient away from the opposing feature component:

\[
g'_{task}=g_{task}-
\frac{g_{task}^{\top}g_{feat}}{\|g_{feat}\|^2}g_{feat},
\qquad g=g_{feat}+\operatorname{Rescale}(g'_{task}).
\]

若两个目标的梯度点积为负，则投影掉任务梯度中与特征梯度相反的分量，再合并更新。

## 4. Factorial design / 因子实验设计

| Mode | Mechanism | Purpose / 目的 |
|---|---|---|
| E0 | R8 only | Supported causal baseline / 可靠基线 |
| E1 | centered local Query residual | Isolate local learned correction / 验证局部学习残差 |
| E2 | deterministic Query transport or retrieval | Isolate temporal transport / retrieval / 验证时序传输或检索 |
| E3 | transport/retrieval + residual | Test composition / 验证组合路径 |

The base model, Backbone, FPN, and Occupancy head are frozen. Training uses window 40–49; 50–69 and 70–89 are independent evaluation windows. All paths check checkpoint SHA, camera order, tensor shape, native bypass, healthy-camera passthrough, history passthrough, zero-init parity, and teacher leakage.

基础模型、Backbone、FPN 与 Occupancy Head 全部冻结。训练窗口为 40–49，50–69 与 70–89 为独立评测窗口。每条路径都核查 checkpoint SHA、相机顺序、张量形状、native bypass、健康相机透传、历史透传、零初始化一致性和 teacher 泄漏。

## 5. Evidence and decision / 证据与决策

### Corrected four-FPN Q2F

- FPN shapes: `64×176`, `32×88`, `16×44`, `8×22`, all with 256 channels and six current cameras.
- Controlled window best gain: `+0.004975` IoU.
- Independent aggregate: `-0.001409` on 50–69 and `-0.016533` on 70–89.
- Removing more queries monotonically improved aggregate performance; therefore the decoder was query-sensitive, but the learned query contribution was harmful.

结论：四层 FPN 写入契约修正成功，但 Q2F 未在独立窗口泛化，不能代替 R8。

### R8 Query residual and transport

- Basic Query residual did not pass preregistered thresholds.
- Transport + residual improved recovered-feature L1 by `0.531158` and cosine by `0.036413`.
- It gained `+0.004677` IoU on 50–69 but regressed `-0.000561` on 70–89; stable-failure gain remained negative.

结论：特征空间更接近 clean teacher，不等于 Occupancy 任务更好。

### Gradient conflict and PCGrad

| Branch | Mean gradient cosine | Negative fraction | Task/feature norm ratio |
|---|---:|---:|---:|
| centered/local residual | `-0.487418` | `1.0` | `55.99` |
| transport residual | `-0.256316` | `1.0` | `44.76` |

PCGrad removed directly opposing components, but the evaluated E3 still regressed `-0.000922` on 70–89 and failed the preregistered generalization and false-free constraints.

结论：冲突不只是优化器层面的问题，更可能来自“逐像素拟合 clean FPN”与“提高稀疏 Occupancy 决策”之间的目标错位。

### Retrieval attention factorial

The query-layer retrieval branch has strict zero-initialization parity and uses historical dense FPN only as memory values. Current results remain below R8 on independent windows; the combined path amplifies errors. It is retained as an ablation, not a supported result.

Query 层检索分支满足零初始化一致性，历史稠密 FPN 仅作为 memory value。但当前独立窗口结果仍低于 R8，组合路径会放大误差，因此只作为消融证据。

## 6. Current solution / 当前解决方案

1. Keep R8 as the production-facing supported mainline. / 保留 R8 作为当前有证据支持的主线。
2. Keep Q2F, residual, transport, retrieval, and PCGrad as explicit negative/diagnostic branches. / 保留 Q2F、残差、传输、检索和 PCGrad 作为负结果与诊断证据。
3. Move the next correction target from dense FPN reconstruction toward Query/Occupancy-space objectives directly aligned with false-free and horizon metrics. / 后续应从稠密 FPN 重建转向 Query/Occupancy 空间，直接对齐 false-free 和未来时域指标。
4. Preserve the factorial and preregistered protocol: E0 parity, frozen base, independent windows, stable-success/failure groups, per-horizon metrics, and clean-scene safety. / 继续保留 E0 一致性、冻结基础模型、独立窗口、稳定成功/失败组、分时域指标和 clean-scene 安全约束。

## 7. Claim boundary / 结论边界

This audit documents an active, uncommitted research workspace. It does not claim an official nuScenes benchmark improvement, a production-ready replacement for R8, or completion of the Robot's current experiment. Numbers are internal frozen-window diagnostics and may be superseded by later work.

本文档记录的是正在进行、尚未提交的研究工作区。不宣称 nuScenes 官方榜单提升，不宣称已得到可替代 R8 的量产方案，也不代表 Robot 当前实验已完成。所有数值都是内部固定窗口诊断，可能被后续结果更新。

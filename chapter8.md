# 第 8 章：DeepSVG：学习式 SVG 表示与生成基线

## 8.1 开篇段落

在第 5 章中，我们探讨了如何通过传统算法（如 Potrace）将栅格图像追踪为矢量图；在第 6 章中，我们学习了如何将 SVG 解析为结构化数据。然而，传统算法缺乏语义理解能力——它们无法“凭空想象”一个图标，也无法理解“圆形”和“正方形”在潜在空间中的语义关系。

本章将深入探讨 **DeepSVG (NeurIPS 2020)**，这是深度学习领域处理 SVG 数据的里程碑式工作。DeepSVG 首次成功地将 SVG 建模为**层次化序列（Hierarchical Sequence）**，并利用 Transformer 和 VAE（变分自编码器）架构实现了高质量的矢量图标重建、生成和插值。

掌握 DeepSVG 不仅仅是学习一个特定的模型，更是为了掌握一套**通用的矢量建模方法论**：如何将连续的几何坐标离散化？如何处理长短不一的路径组合？如何平衡拓扑结构与几何精度？这些问题的解决方案将直接决定我们在后续章节构建的 SVG-MLLM 的底座质量。

---

## 8.2 文字论述

### 8.2.1 数据的“降维打击”：简化与标准化表示

SVG 标准（SVG 1.1/2.0）极其复杂，包含数十种指令和极其灵活的参数。直接将原始 XML 喂给神经网络是不可行的。DeepSVG 确立了一套标准的数据预处理流水线，这套 **Rule-of-Thumb** 被后来的大多数工作（如 IconShop, StarVector）所沿用。

#### 1. 图元统一化 (Primitive Unification)
模型不需要知道什么是 `<rect>`、`<circle>` 或 `<polygon>`。在几何上，它们都可以被数学表达能力更强的 **三次贝塞尔曲线 (Cubic Bézier Curves)** 完美模拟或近似。
*   **直线**：控制点共线的贝塞尔曲线。
*   **圆/椭圆**：通常用 4 段贝塞尔曲线拼接近似（误差极小）。
*   **转换优势**：将指令集词表（Vocabulary）从几十个压缩到极简集合：`Move (M)`, `Line (L)`, `Cubic (C)`, `Close (Z)`。DeepSVG 进一步将 L 也视为特殊的 C，极致简化输出空间。

#### 2. 坐标归一化与张量结构
SVG 的画布大小各异。为了让模型学习“形状”而非“位置”，必须进行 Canonicalization（规范化）：
*   **ViewBox 对齐**：解析 `viewBox`，将图形平移缩放到单位正方形 $[0, 1]$ 或 $[0, 255]$ 范围内。
*   **路径排序**：SVG 路径顺序在渲染时影响遮挡关系，但在语义上往往是无序集合。为了稳定训练，通常按路径的空间位置（如从上到下、从左到右）或长度对 `<path>` 进行重排。

**DeepSVG 的张量表示 $V_{tensor}$**：
一个 SVG 图标被表示为张量 $T$，维度为 $(N_P, N_C, N_F)$：
*   $N_P$：最大路径数（Max Paths），例如 8。
*   $N_C$：每条路径的最大指令数（Max Commands），例如 50。
*   $N_F$：特征维度，通常包含：
    1.  **Command Type**：One-hot 编码（M, L, C, Z, EOS, SOS）。
    2.  **Coordinates**：$(x, y)$ 坐标对。如果是贝塞尔曲线，则是 $(x_1, y_1, x_2, y_2, x, y)$。
    3.  **Visibility**：二值标记，指示该路径/指令是否有效（用于 Padding 掩码）。

### 8.2.2 核心架构：层次化 Transformer (Hierarchical Transformer)

SVG 天然具有 **“图 -> 路径 -> 指令”** 的两级层级结构。普通的 Flat Transformer（将 SVG 视为一长串 token）会丢失这种结构信息，且注意力机制的 $O(L^2)$ 复杂度会随点数增加而爆炸。

DeepSVG 设计了**非自回归编码（Encoder）**与**自回归解码（Decoder）**的混合架构：

```ascii
[输入 SVG 张量]
      |
(1) Path Encoder (处理路径内部)
      |  -> 得到每个 Path 的 Embedding (P1, P2... Pn)
      |
(2) Global Encoder (处理路径之间)
      |  -> 聚合得到 Global Latent Vector (z) <--- VAE 瓶颈在这里
      |
(3) Global Decoder (预测路径属性)
      |  -> 预测路径数量、起始点、样式
      |
(4) Path Decoder (生成具体指令)
      |  -> 针对每条路径，自回归生成：Cmd -> Coord -> Cmd -> Coord...
      |
[输出 SVG 张量]
```

*   **Intra-path attention**：只关注同一条路径内的点，捕捉局部形状（如圆角的曲率）。
*   **Inter-path attention**：关注路径之间的空间关系（如眼睛要在脸的内部）。

### 8.2.3 潜在空间 (Latent Space) 与 VAE 魔法

DeepSVG 不仅仅是一个 AutoEncoder，它是一个 **VAE**。引入 KL 散度损失项迫使潜在向量 $z$ 服从正态分布。这为 SVG 带来了**可计算性**：

1.  **平滑插值 (Morphing)**：
    给定两个图标 $A$ 和 $B$，通过 $z_{mid} = \alpha z_A + (1-\alpha) z_B$，解码 $z_{mid}$ 可以得到一个在几何上介于两者之间的“杂交”图标。
    *   *区别于像素插值*：像素插值是叠加淡入淡出；DeepSVG 插值是形状的动态形变（圆形逐渐变成方形）。
2.  **语义运算**：
    $z_{\text{伤心脸}} \approx z_{\text{笑脸}} - z_{\text{向上嘴角}} + z_{\text{向下嘴角}}$（理想情况下）。

### 8.2.4 训练目标与 Loss 设计 (Recipe for Training)

训练 DeepSVG 需要多任务损失函数的精细平衡：

$$ \mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{KL}} $$

其中重建损失 $\mathcal{L}_{\text{recon}}$ 细分为：

1.  **命令分类损失 (Classification Loss)**：交叉熵损失，判断当前是画直线、曲线还是闭合。
    *   *Gotcha*：类别不平衡。Move (M) 指令很少，Line/Curve 很多，需要加权。
2.  **坐标回归损失 (Coordinate Regression Loss)**：MSE 或 L1 Loss，预测控制点位置。
    *   *Rule-of-Thumb*：只对 Ground Truth 中存在的坐标计算 Loss。如果 GT 是“直线”，预测出的“控制点”坐标不应产生梯度（因为直线没有控制点）。
3.  **可见性损失 (Visibility Loss)**：二分类损失，判断生成的路径是否应该结束。

### 8.2.5 推理与生成策略

在训练好模型后，生成过程（Inference）有多种玩法：

*   **Greedy Search**：每一步取概率最大的指令和坐标。生成稳定，但可能陷入局部最优。
*   **Sampling (Temperature)**：按概率采样。增加多样性，但容易产生自交或畸形线条。
*   **Beam Search**：保留前 K 个最优序列。对于 SVG 这种对拓扑结构敏感的数据，Beam Search 能显著减少“画了一半突然闭合”的错误。

---

## 8.3 本章小结

*   **数据决定上限**：DeepSVG 的成功很大程度上归功于将 SVG 简化为 **Cubic Bézier 序列** 并进行严格的归一化。
*   **结构匹配数据**：层次化 Transformer（Path-level + Command-level）完美契合了 SVG 的 DOM 结构，比纯文本模型更高效。
*   **连续流形**：通过 VAE，我们将离散的 SVG 代码映射到了连续的流形空间，使得矢量图形的运算和演变得以实现。
*   **局限性**：DeepSVG 对于**拓扑极为复杂**（如包含孔洞、复杂的奇偶填充规则）的图形处理能力有限，且对**长序列**（超过 1000 个指令）会出现遗忘，这正是后续 StarVector 等工作试图解决的问题。

---

## 8.4 练习题

### 基础题（熟悉材料）

1.  **预处理逻辑**：给定一个 `<rect x="10" y="10" width="20" height="20"/>`，如果统一转换为 Cubic Bézier 格式，它将由几条指令组成？请写出大致的命令序列（M, C...）。
2.  **张量维度计算**：假设 $N_P=4, N_C=10$，每个指令包含 $(type, x, y)$，其中 type 有 6 种，坐标不量化。请问描述单张图片的张量最小需要多少个浮点数？
3.  **VAE 作用**：如果在训练 DeepSVG 时将 $\beta$（KL 散度的权重）设为 0，模型退化为普通 AutoEncoder。此时尝试对两个图标进行插值，预期会发生什么现象？
4.  **架构细节**：在 DeepSVG 的 Decoder 中，是先预测所有路径的起始点，还是预测完路径 1 的所有指令后再预测路径 2？（提示：回顾层次化结构）。

### 挑战题（深入思考）

5.  **填充规则的歧义**：SVG 的填充有 `nonzero` 和 `evenodd` 两种规则。DeepSVG 的标准实现主要关注轮廓（Stroke）。如果一个圆环是由两个同心圆路径组成的（大圆顺时针，小圆逆时针），DeepSVG 如何保证生成的一大一小两个圆能正确组成一个空心的圆环，而不是两个实心圆？
6.  **序列顺序不变性 (Permutation Invariance)**：SVG 文件中 `<path>` 的顺序打乱通常不影响视觉（除非重叠），但对 RNN/Transformer 来说是完全不同的序列。DeepSVG 采用了简单的排序策略。请设计一种更先进的 Loss 或架构，使模型对路径输入的顺序不敏感（Set Prediction）。
7.  **坐标的量化与回归**：DeepSVG 原文中对坐标直接回归浮点数，但后续有很多工作（如 Image generation 中的 VQ-VAE）主张将坐标离散化为 Token（例如 0-255 的整数）。请分析“回归浮点数”与“预测坐标Token”在 SVG 生成任务中各自的优劣。
8.  **从 SVG-MLLM 的角度**：如果我们要把 DeepSVG 作为一个 Visual Tokenizer 接入 LLM，它的 Latent Vector $z$ 适合直接作为 LLM 的输入 Token 吗？如果不适合，需要做什么转换？

<details>
<summary>点击查看练习题提示与答案方向</summary>

1.  **答案**：至少 5 个指令：`M` (移动到角), `L`/`C` (边1), `L`/`C` (边2), `L`/`C` (边3), `L`/`C` (边4), `Z` (闭合)。DeepSVG 可能会把 L 也变成 C。
2.  **答案**：$4 \times 10 \times (6 + 2) = 320$。实际实现中会有 Padding mask 和 Visibility 位。
3.  **答案**：插值会变得不平滑。中间状态可能不是一个合法的图形，或者在两张图之间生硬跳变，而不是渐变变形。
4.  **答案**：通常是分层的。Global Decoder 先确定路径的宏观属性（如起始点、Latent），然后 Path Decoder 逐个或并行地展开每条路径的具体指令。
5.  **提示**：这是一个痛点。模型需要隐式学习到“内嵌的路径通常是孔洞”这一规律。更强的做法是显式预测路径的方向（顺/逆时针）作为特征。
6.  **提示**：参考 DETR (Detection Transformer) 的匈牙利匹配损失 (Hungarian Matching Loss)。在计算 Loss 之前，先找到预测路径集合与真实路径集合之间的最佳二分图匹配。
7.  **提示**：回归浮点数精度高，但容易产生模糊的“平均值”结果；离散 Token 容易捕捉多峰分布（即边缘锐利），但不仅这就增加了词表大小，还需要处理量化误差。
8.  **提示**：不适合。$z$ 是连续向量，LLM 只能处理离散 Token。需要经过 Quantization (如 VQ-VAE 的 Codebook) 将 $z$ 映射为离散的 ID 序列，或者通过一个 Linear Projector 将 $z$ 投影到 LLM 的 Embedding 空间（类似 LLaVA）。

</details>

---

## 8.5 常见陷阱与错误 (Gotchas)

在复现或使用 DeepSVG 时，以下陷阱会让你的 Loss 居高不下或生成结果一团糟：

1.  **幽灵路径 (The Phantom Paths)**
    *   **现象**：模型生成的 SVG 代码看起来正常，但浏览器渲染出来一片空白或缺失部分。
    *   **原因**：训练数据中包含大量 `opacity="0"` 或 `display="none"` 的垃圾路径，或者路径的 `fill` 和 `stroke` 属性未被正确预测（默认为无色）。
    *   **对策**：在数据清洗阶段，**必须渲染**一遍以剔除不可见路径。在生成阶段，强制设置默认的 `stroke="black"` 和 `fill="none"` 用于调试。

2.  **贝塞尔曲线“飞线” (Exploding Control Points)**
    *   **现象**：图形大体正常，但偶尔有一两条线飞出屏幕几千像素远。
    *   **原因**：回归 Loss 对异常值不够敏感，或者模型未能学好控制点与端点的相对关系。
    *   **对策**：使用 `Smooth L1 Loss` 代替 `MSE`；或者将绝对坐标预测改为预测**相对偏移量 (Relative Offsets)**，限制控制点的活动范围。

3.  **Start/End Token 的混淆**
    *   **现象**：生成序列无法停止，或者路径之间粘连。
    *   **原因**：混淆了 `Global EOS`（整个图标结束）和 `Path EOS`（当前路径结束）。
    *   **对策**：在构建词表时，严格区分层级 Token。检查 Mask 矩阵是否正确屏蔽了 Padding 区域。

4.  **过拟合于简单的几何变换**
    *   **现象**：模型只会平移或缩放，无法改变形状。
    *   **原因**：数据集通过简单的 Affine Transformation 扩增太多，导致模型“偷懒”。
    *   **对策**：增加非线性扩增（如对控制点加高斯噪声），强迫模型学习形状的鲁棒表示。

5.  **ViewBox 的陷阱**
    *   **错误**：直接读取 `d` 属性中的坐标进行训练，忽略了父级 `<g transform="...">` 或根节点的 `viewBox`。
    *   **后果**：不同 SVG 的坐标量级相差巨大（有的 0-100，有的 0-10000），导致梯度爆炸或无法收敛。
    *   **对策**：必须实现一个完整的 **Flatten Transform** 算法，将所有变换矩阵应用到叶子节点的路径坐标上，统一转换到世界坐标系后再归一化。

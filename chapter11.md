# 第 11 章：SVG-MLLM 架构设计：理解与生成一体化

## 11.1 开篇段落：重新定义 SVG 模型

在多模态大模型（MLLM）的浪潮中，我们习惯了 GPT-4V 或 Claude 3 这种“看图说话”的模式。然而，构建一个 **SVG-MLLM** 远比处理普通自然图像复杂。普通的 MLLM 输出的是自然语言描述，容错率极高；而 SVG-MLLM 输出的是 **可执行代码**。少一个闭合标签 `/>`，或者坐标偏离了 5%，整个图像可能就会崩坏或完全改变语义。

因此，SVG-MLLM 的架构不能仅仅是简单的 `ViT + LLM`。我们需要构建一个 **“三位一体”** 的架构：
1.  **视觉感知 (Perception)**：看懂渲染后的像素（Raster）。
2.  **代码理解 (Code Reasoning)**：理解 XML 树状结构与属性（Symbolic）。
3.  **几何对齐 (Geometric Grounding)**：将像素空间与代码数值空间强绑定。

本章将带你深入设计这个系统的每一个组件：从处理长序列坐标的特殊 Tokenizer，到融合视觉特征的 Projector，再到保证输出合法的约束解码器。

---

## 11.2 核心任务抽象与架构概览

在设计架构前，我们必须明确模型需要支持的 **五大核心任务**，这决定了输入输出流的设计：

1.  **SVG Captioning (S2T)**: 输入 SVG 代码或图像 $\rightarrow$ 输出自然语言描述。
2.  **Text-to-SVG (T2S)**: 输入自然语言 $\rightarrow$ 输出 SVG 代码。
3.  **SVG Rendering/Prediction**: 输入部分 SVG 代码 $\rightarrow$ 预测渲染后的视觉特征（自监督学习用）。
4.  **SVG Editing (S2S)**: 输入原 SVG + 修改指令 $\rightarrow$ 输出新 SVG。
5.  **Visual Grounding**: 输入文本查询 $\rightarrow$ 输出对应图元的 ID 或 Bounding Box。

### 11.2.1 总体架构图 (ASCII Flow)

```text
[Input Modalities]           [The "Brain" (LLM)]             [Output]
------------------           -------------------             --------

1. Image Input (PNG)
       ⬇
  [Visual Encoder]  ──┐
  (e.g., SigLIP)      │
       ⬇              │
  [MLP Projector]     │
       ⬇              │
 [Visual Tokens] ─────┼───> [ Transformer Decoder ] ───> [Probability Distribution]
                      │     (Llama-3 / Qwen-2 Base)                ⬇
2. Text Instruction   │               ⬆                      [Next Token Prediction]
       ⬇              │        [LoRA Adapters]                     ⬇
 [Text Tokenizer] ────┤               │                     [Constraint Masking]
       ⬇              │               │                            ⬇
 [Text Tokens] ───────┘               │                  ┌───────────────────┐
                                      │                  │ 1. Text Response  │
3. SVG Code Input                     │                  │ 2. SVG Code Stream│
       ⬇                              │                  │    <path d="..."  │
 [SVG Special Tokenizer] ─────────────┘                  └───────────────────┘
 (Coord Discretization)
```

---

## 11.3 输入端设计 I：SVG 的序列化与 Embedding

这是 SVG-MLLM 最关键的“独门绝技”。直接用通用的 BPE（Byte Pair Encoding）处理 SVG 是灾难性的。

### 11.3.1 为什么通用 Tokenizer 会失败？
对于路径 `<path d="M 150.5 200.0 ..."/>`：
*   **LLM 视角**：它被切分为 `['<', 'path', 'd', '=', '"', 'M', '150', '.', '5', '200', ...]`。
*   **问题**：一个简单的圆可能消耗 500+ 个 Token。浮点数被切碎后，模型丢失了数值大小的概念（它不知道 `100` 比 `10` 大，它只当它们是字符）。

### 11.3.2 解决方案：专用 SVG Tokenizer

我们需要设计一种混合 Tokenizer：

1.  **结构词 (Structural Tokens)**：
    保留 XML 标签，如 `<svg>`, `<rect>`, `fill=`, `stroke=`。这部分沿用 LLM 词表。

2.  **坐标离散化 (Coordinate Binning)**：
    将连续的浮点坐标映射到离散的整数区间（Bins）。
    *   设定画布大小为 $H \times W$（例如 $1024 \times 1024$）。
    *   将所有坐标归一化到 $[0, 1024]$ 整数。
    *   **新增词表**：向 LLM 词表扩充 `<coord_0>` 到 `<coord_1024>` 共 1025 个特殊 Token。
    *   **效果**：`M 150.5 200.0` 变为 `M` `<coord_150>` `<coord_200>`。序列长度缩减 60% 以上，且模型能学到 `<coord_100>` 和 `<coord_101>` 在嵌入空间上的邻近性。

3.  **命令压缩 (Command Compression)**：
    可以将常见的命令组合合并，例如 `M_x_y` 作为一个整体事件，或者保留 SVG 的简写逻辑（`h`, `v` 相对坐标）。

> **Rule-of-Thumb 11.1**:
> **坐标即位置 embedding。**
> 不要训练模型去“阅读”数字字符串。要让模型“选择”位置索引。通过使用 `<coord_i>`，你实际上是将回归问题转化为了分类问题，这在 Transformer 中更稳定。

---

## 11.4 输入端设计 II：视觉编码器与多尺度感知

SVG 的特点是**无限分辨率**。线条在任何缩放级别下都是清晰的，但视觉编码器（Visual Encoder）输入通常是固定的（如 $336 \times 336$ 或 $448 \times 448$）。

### 11.4.1 编码器选择
*   **CLIP vs. SigLIP vs. InternViT**：
    *   CLIP 收敛快但分辨率低，对细线条（Stroke）捕捉能力差。
    *   **推荐**：**SigLIP** (Sigmoid Loss for Language Image Pre-training) 或 **InternViT**。它们在密集文本和几何形状上的表现通常优于原始 CLIP。

### 11.4.2 动态分辨率策略 (Dynamic High-Res)
为了处理包含大量细节的工程图或地图 SVG，我们需要引入 **AnyRes (Any Resolution)** 机制：
1.  **全局视图**：将图片 resize 到 $336 \times 336$，获取整体布局信息。
2.  **局部切片 (Crops)**：将原图切分为 $N$ 个 $336 \times 336$ 的图块（Tiles）。
3.  **融合**：将全局特征 + 局部图块特征拼接。
4.  **对于 SVG 的特殊意义**：SVG 往往包含极小的文字或图符，局部切片能防止这些细节在 resize 过程中由于抗锯齿而被“抹平”。

---

## 11.5 核心融合层：Projector 与模态对齐

视觉特征是 Dense 的（浮点向量），文本特征是 Discrete 的（Token ID）。连接它们的桥梁是 Projector。

### 11.5.1 Projector 类型
*   **Linear/MLP**：简单高效，适合数据量极大的预训练（LLaVA 方案）。
*   **Q-Former / Perceiver Resampler**：使用一组可学习的 Query 来“提取”视觉特征。适合压缩超长视觉序列（例如使用了动态分辨率切片后）。

### 11.5.2 特征对齐策略 (Grounding)
在架构层面，我们需要让模型知道 **SVG 代码中的 `<path id="5">`** 到底对应 **图像中的哪一块像素**。
*   **架构增强**：不只是简单的 Concat。可以引入 **Segment tokens**。
*   **输入构造**：`User: <image_embeddings> <coord_bbox_10_10_50_50> 这里面是什么？ Model: 这是一个 <rect>。`
*   通过这种显式的空间指令微调，迫使 Projector 学习到像素坐标与 SVG 坐标的映射关系。

---

## 11.6 输出端设计：约束解码器 (Constrained Decoder)

这是 SVG-MLLM 与普通 Chatbot 最大的区别。我们需要保证输出是“合法的 SVG”。

### 11.6.1 为什么需要约束？
LLM 本质是随机采样。它可能会生成：
*   `<rect x="10" r="5" />` （`rect` 没有 `r` 属性，只有 `rx/ry`）。
*   `<path d="M 10 10 L 20" />` （路径数据中途截断，缺少坐标）。
*   `fill="redd"` （颜色拼写错误）。

### 11.6.2 基于 Grammar 的采样 (Grammar-Guided Decoding)
我们不在训练时施加硬约束，而是在 **推理 (Inference)** 阶段施加 Logit 掩码。

1.  **定义 SVG 文法 (BNF/EBNF)**：
    定义合法的 XML 结构和 Path Data 序列规则。
2.  **构建有限状态机 (FSM)**：
    *   状态 0：期待 `<`。
    *   状态 1（收到 `<`）：期待标签名 `svg`, `path`, `rect`...
    *   状态 2（收到 `rect`）：期待属性名 `x`, `y`, `width`...
3.  **Logit Processor**：
    在模型生成下一个 Token 前，检查 FSM 的当前状态。将所有导致非法状态转换的 Token 的概率设为 $-\infty$。
    *   *例如*：如果当前正在生成 `x="` 之后，只有数字 Token `<coord_i>` 是允许的，字母 Token 被屏蔽。

> **Rule-of-Thumb 11.2**:
> **让规则不仅在心中，也在手中。**
> 仅仅依靠大量数据训练模型学会语法是昂贵的（且不保证 100% 正确）。加上一个轻量级的 Grammar Decoder（仅增加 <10ms 延迟）可以瞬间将语法正确率提升至 100%，让模型专注于语义和几何生成。

---

## 11.7 高级推理架构：思维链与工具回路 (System 2 Architecture)

为了处理复杂的 SVG 生成（如“画一个带渐变背景的流程图”），单次 Pass 生成往往不够。我们需要设计 **Agentic Workflow**。

### 11.7.1 渲染-修正回路 (Render-and-Refine)
这是一个多轮对话的架构模式：

1.  **Drafting**: 模型生成初版 SVG 代码 $C_0$。
2.  **Execution**: 架构中的 Python 执行器调用 `resvg` 将 $C_0$ 渲染为图像 $I_0$。
3.  **Perception**: 视觉编码器将 $I_0$ 编码回模型。
4.  **Critique**: 模型比较 $I_0$ 与用户指令（Prompt），生成修改建议（Feedback）。
5.  **Refinement**: 模型根据 Feedback 生成 $C_1$。

这实际上是在模拟人类设计师“画一笔，看一眼，改一笔”的过程。

### 11.7.2 布局规划 (Layout-First Generation)
模型可以先生成布局树（Layout Tree），再填充几何细节。
*   Step 1: 生成 `<bbox class="title" x="10" y="10" w="100" h="20" />`
*   Step 2: 根据 bbox 生成具体的 `<text>` 和 `<path>`。
这降低了长序列生成的难度。

---

## 11.8 本章小结

SVG-MLLM 的架构设计是一场在“灵活性”与“精确性”之间的走钢丝：
1.  **输入端**：必须采用 **混合 Tokenizer**（文本 + 离散坐标），并配合高分辨率或多尺度的 **Visual Encoder** 来捕捉细微的矢量特征。
2.  **模型主体**：利用 LLM 的预训练知识，通过 Projector 将视觉对齐到文本空间。
3.  **输出端**：不能只依赖概率，必须引入 **Grammar-Guided Decoding** 确保语法/几何合法性。
4.  **闭环**：通过渲染回路（Render Loop），让模型具备自我纠错的能力，这是通往高质量矢量生成的必经之路。

下一章，我们将探讨如何利用这个架构进行训练：从预训练数据的构造到指令微调的技巧。

---

## 11.9 练习题

### 基础题
1.  **Tokenizer 实验**：
    给定 SVG 片段 `<circle cx="50.5" cy="50.5" r="10"/>`。
    *   (a) 写出其被标准 GPT-4 Tokenizer 切分的大致结果。
    *   (b) 假设画布为 $100\times100$，量化精度为 1。写出将其转换为 `<cmd> <coord>` 格式后的序列。
    *   **Hint**: 注意 XML 属性名和数值的分离。

2.  **架构对比**：
    对比 DeepSVG（基于 VAE 和 LSTM/Transformer）与本章提出的 SVG-MLLM（基于 Decoder-only LLM）。列出至少三个维度的差异（输入模态、生成方式、通用性）。

3.  **约束逻辑**：
    如果要为一个只生成“矩形”的模型编写 Grammar Mask。请写出伪代码逻辑：当已经生成 `<rect` 后，下一个允许的 Token 集合是什么？
    *   **Hint**: 属性名 `x`, `y`, `width`, `height`, `fill`, `stroke` 以及闭合符 `/>`。

### 挑战题
4.  **多模态幻觉调试**：
    你训练的模型在 Text-to-SVG 任务中，经常生成代码是蓝色的，但 Caption 说是红色的。
    *   从架构角度分析，可能是 Vision Encoder、Projector 还是 LLM 的问题？
    *   设计一个消融实验（Ablation Study）来定位问题。

5.  **超长序列处理**：
    一张复杂的工程图 SVG可能有 10 万个 Token，超出了 Llama-3 的上下文窗口。
    *   请设计一种“分层生成”或“滑动窗口”架构来解决这个问题。
    *   **Hint**: 参考 Google Maps 的瓦片（Tile）加载机制，或者 DOM 树的层级展开（先生成 Group 结构，再进入 Group 生成内容）。

6.  **开放性思考：可微渲染的集成**：
    如果我们不使用外部工具 `resvg`，而是想把渲染器做成一个可微的神经网络层（Differentiable Rendering Layer）直接插入到 Decoder 后端。这将如何改变训练的 Loss 函数？
    *   **Hint**: 此时可以直接计算 Pixel Loss，而不仅仅是 Cross-Entropy Loss。

---

## 11.10 常见陷阱与错误 (Gotchas)

*   **陷阱 1：忽视 XML 的冗余性**
    *   **现象**：模型花费大量 Token 生成无意义的 `xml:space="preserve"` 或冗长的 `style` 字符串。
    *   **对策**：在数据预处理阶段（见第 4 章），务必进行 Canonicalization（规范化），剔除所有非必要的属性，将 style 属性展开为独立属性（如 `style="fill:red"` $\rightarrow$ `fill="red"`）。架构上，Tokenizer 甚至可以把常用属性名合并为一个 Token。

*   **陷阱 2：Projector 的初始化崩溃**
    *   **现象**：刚开始微调时，Loss 不降反升，或者模型开始胡言乱语。
    *   **对策**：如果 Vision Encoder 和 LLM 都是预训练好的，**必须冻结它们**，只训练 Projector（Warmup 阶段）。等待特征空间对齐后，再解冻 LLM 进行全量微调。

*   **陷阱 3：坐标归一化的坑**
    *   **现象**：模型生成的图形长宽比（Aspect Ratio）被拉伸了。
    *   **原因**：在 Tokenizer 中简单地将 $x, y$ 归一化到 $[0, 1024]$，忽略了原始 ViewBox 的长宽比。
    *   **对策**：始终保持长边对齐（Fit），短边留白（Pad），或者将 `viewBox` 的长宽比作为一个特殊的 Prompt Token 输入给模型（例如 `<ratio_1.5>`）。

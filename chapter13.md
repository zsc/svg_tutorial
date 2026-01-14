# 第 13 章：评测体系：像素、结构、语义与可编辑性

## 1. 开篇与学习目标

在多模态大模型（MLLM）的语境下，评价一个 SVG 生成模型的性能远比评价 Stable Diffusion 这种纯像素生成模型要复杂。因为 SVG 具有**双重属性**：
1.  **视觉属性**：它渲染后是一张图像，必须符合人类的视觉审美和语义要求。
2.  **代码属性**：它本质上是一段 XML 代码，必须符合语法规范，且具备拓扑结构上的合理性。

一个“视觉上完美”但“代码由一万个乱序微小线段组成”的 SVG，在工程上是不可用的垃圾数据（不可编辑、渲染卡顿、无限放大失真）。反之，代码极其精简但画得不像的 SVG 也是失败的。

本章将建立一个**全方位、分层级**的评测体系，从底层的语法解析率，到中间层的几何拓扑质量，再到顶层的语义对齐度，为你提供一套可复用、可量化的“体检报告”生成方案。

**本章学习目标：**
*   **掌握渲染层评测**：深入理解 LPIPS、FID 在矢量图领域的适用性与局限。
*   **构建结构层评测**：学会量化“可编辑性”，通过几何特征识别“伪矢量”（由像素描摹导致的过度拟合）。
*   **实施语义层评测**：利用 CLIP 和 VLM（如 GPT-4V）作为裁判进行自动化打分。
*   **数据卫生管理**：掌握基于渲染指纹（Rendered Fingerprint）的严格去重策略，防止测试集泄漏。
*   **搭建 Eval Harness**：设计一套自动化的评测流水线。

---

## 2. 核心论述

### 2.1 评测流水线设计：标准化（Canonicalization）

在计算任何指标之前，必须对模型生成的 SVG 和 Ground Truth（真实数据）进行“对齐”。直接比较两个 SVG 的文本字符串（BLEU/Rouge）是毫无意义的，因为 `path d="M 0 0 L 10 10"` 和 `path d="M 0 0 l 10 10"`（相对坐标）视觉一样但文本不同。

**标准评测流水线 (Evaluation Pipeline)**：

1.  **语法修复 (Syntax Repair)**：使用 `lxml` 或 `BeautifulSoup` 尝试解析输出。如果模型输出缺少闭合标签 `</svg>`，尝试自动补全。
2.  **视口归一化 (Viewport Normalization)**：
    *   读取 `viewBox`。如果缺失，计算 bounding box。
    *   将所有坐标通过矩阵变换映射到 $[0, 1] \times [0, 1]$ 或固定尺寸（如 $256 \times 256$）的统一坐标系中。
    *   这一步是为了消除位移（Translation）和缩放（Scale）带来的无关误差。
3.  **样式内联 (Style Inlining)**：将 CSS class 转换为内联样式，确保渲染引擎不会因为缺少 CSS 解析器而渲染成黑色方块。
4.  **栅格化 (Rasterization)**：使用 `resvg` 或 `Cairo` 将 SVG 渲染为 PNG。通常建议渲染两套分辨率：
    *   **低分（64x64）**：用于计算拓扑概览和颜色分布。
    *   **高分（512x512）**：用于计算细节 LPIPS 和边缘质量。

### 2.2 维度一：视觉与像素层评测 (Rendering Metrics)

这是最直观的评测，回答“**画得像不像**”的问题。

#### 2.2.1 像素级距离 (Pixel-wise Metrics)
*   **MSE (Mean Squared Error) / L1 / L2**：计算像素差值的平方和。
    *   *局限性*：对**位移**极其敏感。如果生成图只是向右平移了 1 个像素，MSE 会爆炸，但人类觉得这是一样的。仅适用于严格对齐的 Icon 重建任务。
*   **IoU (Intersection over Union)**：
    *   适用于单色/二值化图形。
    *   $IoU = \frac{Area(Gen \cap Ref)}{Area(Gen \cup Ref)}$
    *   *注意*：对于细线条图形，IoU 通常很低，建议先对线条进行膨胀（Dilation）操作再计算 IoU，这被称为 **Relaxed IoU**。

#### 2.2.2 感知级距离 (Perceptual Metrics)
*   **LPIPS (Learned Perceptual Image Patch Similarity)**：**（黄金标准）**
    *   利用预训练的 VGG/AlexNet 提取深层特征计算距离。
    *   它能容忍轻微的局部变形和位移，更符合人类直觉。SVG 生成任务中，LPIPS < 0.1 通常意味着非常好的重建。
*   **FID (Fréchet Inception Distance)**：
    *   用于评估**数据集整体分布**的质量，而非单张图。
    *   如果你做的是“无条件生成”或“类别条件生成”，需要计算生成集与真实集的 FID。FID 越低，生成的图像真实度和多样性越好。

### 2.3 维度二：结构与代码层评测 (Structural Metrics)

这是区分“SVG 模型”和“像素模型”的关键。我们追求的是**奥卡姆剃刀原则**：用最少的指令描述最复杂的图形。

#### 2.3.1 复杂度指标 (Complexity & Sparsity)
*   **路径数量 (Path Count, $N_{path}$)**：越少越好。
*   **指令数量 (Command Count, $N_{cmd}$)**：总的绘图指令（M, L, C, Z 等）数量。
*   **控制点密度 (Control Point Density)**：
    *   计算公式：$\rho = \frac{N_{points}}{Length_{path}}$
    *   **判断标准**：如果 $\rho$ 极高（例如每 2 个像素就有一个控制点），说明模型在进行“过拟合”或简单的“像素描摹（Tracing）”，这是**低质量**的矢量。高质量矢量应该用长贝塞尔曲线跨越长距离。

#### 2.3.2 图元使用率 (Primitive Usage)
*   **Circle/Rect Ratio**：
    *   检测模型是否学会了使用 `<circle>`, `<rect>`, `<ellipse>` 等高级图元，而不是全部用 `<path>` 拟合。
    *   *指标*：$R_{prim} = \frac{N_{shapes}}{N_{shapes} + N_{paths}}$。该值越高，说明模型对几何语义理解越深。

#### 2.3.3 语法合法性 (Code Validity)
*   **Parse Rate**：$\frac{Parsed}{Total}$。能否过 XML parser。
*   **Render Rate**：$\frac{Rendered}{Total}$。解析后能否不报错地渲染（排除死循环、内存溢出、非法属性值）。
*   **Winding Rule Consistency**：检测是否错误使用了 `fill-rule="evenodd"` 导致图形出现意外的空洞。

### 2.4 维度三：可编辑性与拓扑评测 (Editability & Topology)

可编辑性衡量的是 SVG 是否**对人类设计师友好**。

#### 2.4.1 闭合性 (Closure)
*   对于填充（Fill）的区域，路径必须闭合。
*   *检测*：检查 path 的最后指令是否为 `Z` 或 `z`。如果不是，计算起点 ($P_{start}$) 和终点 ($P_{end}$) 的欧氏距离。距离应小于 $\epsilon$。

#### 2.4.2 层级结构 (Hierarchy & Grouping)
*   **Group Utilization**：模型是否使用了 `<g>` 标签？
*   **Semantic Grouping**（高阶评测）：如果是生成“人脸”，理想情况下，“左眼”的所有路径应该在一个 `<g id="left_eye">` 中。可以通过计算 Group 内元素的空间聚集度（Spatial Clustering）来评估。

#### 2.4.3 冗余度 (Redundancy / Overdraw)
*   **看不见的路径**：检测被上层不透明形状完全遮挡的下层路径。
*   **零长度路径**：检测并惩罚 `d="M 10 10 L 10 10"` 这种无效指令。

### 2.5 维度四：语义一致性 (Semantic Alignment)

当没有参考图（Reference-free），只有文本提示词（Prompt）时，如何判断生成得对不对？

#### 2.5.1 CLIP-based Metrics
*   **CLIP Score**：计算 `Sim(Image_Encoder(Gen_SVG), Text_Encoder(Prompt))`。
*   **CLIP-R-Precision**：给定生成图和一组干扰文本，看 CLIP 能否检索到正确的 Prompt。
*   *陷阱*：CLIP 对几何形状（“五角星” vs “六角星”）和数量（“三个苹果”）并不敏感。它更擅长物体类别和艺术风格。

#### 2.5.2 LLM/VLM-as-a-Judge (GPT-4V / LLaVA)
这是目前最先进的评测方法。构建一个自动化裁判系统。

*   **VQA 准确率**：
    *   Prompt: "Draw a red triangle."
    *   Eval Prompt (to VLM): "Look at this image. What shape is it? What color is it?"
    *   计算 VLM 回答的准确率。
*   **综合打分**：
    *   Eval Prompt (to VLM): "Rate this SVG icon on a scale of 1-5 based on: 1. Visual clarity, 2. Relevance to prompt '{prompt}', 3. Aesthetic appeal."

### 2.6 数据集去重与防泄漏 (Data Hygiene)

由于 SVG 源码的易变性，基于文本的去重（如 MinHash）是不够的。

**推荐的去重策略**：
1.  **Rendered Hash**：将所有训练集和测试集数据渲染为 64x64 灰度图。
2.  **pHash (Perceptual Hash)**：计算图像指纹。
3.  **距离阈值**：如果 Hamming Distance < $T$（例如 5），则视为重复数据，从训练集中剔除。
4.  **理由**：这能防止模型仅仅是“记住了”训练集里的某个 SVG 代码，而不是学会了生成。

---

## 3. 本章小结

*   **多维视角**：SVG 评测 = **LPIPS** (看着像) + **Complexity** (代码简) + **CLIP/VQA** (语义对)。
*   **代码质量是隐形杀手**：不要只看缩略图。必须检查控制点密度和路径数量，防止模型学会“伪矢量化”。
*   **渲染一致性**：所有评测必须基于规范化的 ViewBox 和统一的渲染后端。
*   **大模型裁判**：在缺乏 Ground Truth 的生成任务中，使用 VLM 进行 VQA 测试是目前最鲁棒的语义评测手段。
*   **严谨去重**：基于渲染指纹的去重是保证 MLLM 泛化能力评估有效的底线。

---

## 4. 练习题

### 基础题

1.  **[IoU 变体]** 为什么对于细线条的 SVG 图标（Stroke-based icons），直接计算 IoU 分数通常极低且不具备参考价值？请提出一种改进方案。（Hint: 考虑线条宽度和对齐误差）。
2.  **[复杂度计算]** 编写一个伪代码函数 `calculate_complexity(svg_string)`，输入 SVG 字符串，返回其路径数、指令数和平均每条路径的控制点数。
3.  **[LPIPS vs MSE]** 给定一个正圆。情况 A：圆心向右偏移 2 像素。情况 B：圆变成了正方形，但位置完全重合。请预判 MSE 和 LPIPS 在这两种情况下的相对表现（谁对 A 惩罚大？谁对 B 惩罚大？）。
4.  **[视口归一化]** 一个 SVG 的 `viewBox="0 0 100 50"`，其中有一个点 `(50, 25)`。如果我们将其归一化到 `[0, 1] x [0, 1]` 的空间，这个点的新坐标是什么？注意保持纵横比（Aspect Ratio）的处理策略。

### 挑战题

5.  **[自动化裁判 Prompt 设计]** 你正在评测一个“Text-to-SVG”模型。请设计一段 System Prompt，发给 GPT-4V，让它不仅评估生成图的**内容正确性**，还要评估**矢量图特有的美学**（如线条流畅度、极简风格）。
6.  **[可编辑性检测算法]** 假设模型生成了一条由 100 个 `L` (Line) 指令组成的曲线，但这 100 个点其实都落在一条直线上。请设计一个算法来检测这种“冗余分段”，并计算“冗余率”。
7.  **[对抗性攻击评测]** 构造一个 SVG 案例（手动编写），使得它的 CLIP Score 极高（>32），但 Parse Rate 为 0 或者人类看起来完全是乱码。（Hint: CLIP 攻击通常涉及在图像中隐藏文字或纹理）。
8.  **[闭环评测系统设计]** 设计一个 Python 类结构 `SVGEvaluator`，包含 `evaluate_visual()`, `evaluate_code()`, `evaluate_semantic()` 三个方法。要求支持 batch 处理和 GPU 加速（针对 LPIPS/CLIP）。

---

### 练习题提示与答案

<details>
<summary><strong>点击查看提示与答案思路</strong></summary>

1.  **IoU 变体**
    *   **Hint**: Thin lines have minimal area intersection if shifted slightly.
    *   **Answer**: 细线条的交集对位移极度敏感。改进方案：**Relaxed IoU**。在计算 IoU 前，对生成图和参考图都进行形态学**膨胀（Dilation）**操作（例如膨胀 3-5 个像素），增加容错率，关注拓扑重叠而非像素重叠。

2.  **复杂度计算**
    *   **Hint**: Use a regex or SVG parser library like `svgpathtools`.
    *   **Answer**: 解析 `d` 属性，Count 'M','L','C'...。Count Points based on command type (C=3 points, L=1 point). Avg = Total Points / Total Paths.

3.  **LPIPS vs MSE**
    *   **Hint**: MSE hates misalignment. LPIPS looks at features (edges/shapes).
    *   **Answer**: MSE 对情况 A（平移）惩罚极大，对情况 B（形状变异但重叠面积大）可能惩罚较小。LPIPS 对情况 A 惩罚较小（特征图平移不变性），对情况 B（形状特征彻底改变）惩罚大。LPIPS 更符合人类判断：平移的圆还是圆，但正方形不是圆。

4.  **视口归一化**
    *   **Hint**: Map (x, y) to (x/w, y/h). Handling Aspect Ratio usually means centering or scaling by max dimension.
    *   **Answer**: 如果忽略纵横比拉伸：`(0.5, 0.5)`。如果保持纵横比（Fit in 1x1）：最大边长是 100，缩放因子 $s = 1/100$。新坐标 `(50*0.01, 25*0.01) = (0.5, 0.25)`，并在 Y 轴方向可能需要居中偏移。

5.  **自动化裁判 Prompt**
    *   **Hint**: Define criteria explicitly.
    *   **Answer**: "Role: You are a Senior Vector Graphic Designer. Task: Evaluate the attached image generated from the prompt '{prompt}'. Criteria: 1. Semantic Accuracy (Is the object correct?). 2. Vector Aesthetics (Are lines smooth? Is the composition balanced? Is it clean or messy?). Output strictly JSON: {score_semantic: 1-10, score_aesthetic: 1-10, reasoning: '...'}"

6.  **可编辑性检测算法**
    *   **Hint**: Collinearity check based on slopes or cross product.
    *   **Answer**: 遍历所有相邻线段 $P_{i-1}, P_i, P_{i+1}$。计算向量 $\vec{v_1} = P_i - P_{i-1}$ 和 $\vec{v_2} = P_{i+1} - P_i$。如果两个向量的夹角接近 0（或叉积接近 0），则点 $P_i$ 是冗余的。统计这类点的比例。

7.  **对抗性攻击评测**
    *   **Hint**: CLIP reads text inside images well.
    *   **Answer**: 可以在 SVG 中插入大量微小的文字（如写满 "apple"），或者用杂乱的线条拼凑出 "apple" 的字样。CLIP 会识别文字语义给高分，但这在图形上不是一个苹果。如果 SVG 代码里全是 `<text>` 标签而没有图形，Parse Rate 虽高但作为图形生成任务是失败的。

8.  **闭环评测系统设计**
    *   **Hint**: Structure involves dataloader, renderer, model interfaces.
    *   **Answer**:
        ```python
        class SVGEvaluator:
            def __init__(self, device='cuda'):
                self.lpips_model = ...
                self.clip_model = ...
                self.renderer = ResvgRenderer()
            def evaluate_visual(self, svg_code, gt_image):
                # render -> tensor -> lpips
            def evaluate_code(self, svg_code):
                # parse -> check paths/commands -> complexity score
            def evaluate_semantic(self, svg_code, prompt):
                # render -> tensor -> clip_score(image, prompt)
        ```

</details>

---

## 5. 常见陷阱与错误 (Gotchas)

### 5.1 颜色空间的“坑”：sRGB vs Linear
*   **问题**：计算 MSE 或 LPIPS 时，渲染出的 PNG 是 sRGB（经过 Gamma 校正），而很多 PyTorch 里的图像 Tensor 假设是线性的或归一化到 `[-1, 1]` 的。
*   **后果**：即使图片看起来一样，数值误差也很大。
*   **解决**：确保渲染引擎输出和 Ground Truth 图像处于相同的色彩空间。通常建议在计算 Loss 前将图像都转换到 Linear RGB 空间，或统一使用 sRGB 并归一化到 `[0, 1]`。

### 5.2 Stroke-Width 的“消失魔术”
*   **问题**：生成的 SVG 在浏览器里看正常，但喂给 CLIP 模型时得分极低。
*   **原因**：CLIP 的预处理通常会将图片 Resize 到 `224x224`。如果你的 SVG 线条非常细（例如 1px on 1024px canvas），Resize 之后线条会因为抗锯齿算法变得极其模糊甚至消失。
*   **解决**：在送入 CLIP 评测前，**人为加粗** SVG 的渲染线条（在渲染配置里设置最小 stroke-width），或者在渲染时使用更高的 DPI，然后进行高质量的 Downsampling。

### 5.3 默认值的诅咒
*   **问题**：模型生成的 SVG 没有写 `fill` 属性。
*   **后果**：SVG 标准规定默认 `fill="black"`。如果你的画布背景也是黑色，图形就“隐身”了。
*   **解决**：评测代码必须具备**鲁棒的默认值注入**机制。例如，强制设置背景为白色，若元素未指定 fill/stroke，则显式设为默认颜色（如黑色 fill，无 stroke），以防渲染引擎行为不一致。

### 5.4 幽灵路径 (Ghost Paths)
*   **问题**：代码评分显示 `complexity` 很高，但图看起来很简单。
*   **原因**：模型生成了大量 `opacity="0"`，`fill="none" stroke="none"`，或者位于视图外的路径。
*   **解决**：在计算结构指标前，必须进行 **Pruning（剪枝）**。移除所有不可见的节点，否则你的复杂度惩罚会错误地惩罚“不可见”的垃圾代码，而不是惩罚“画得复杂”的图形。

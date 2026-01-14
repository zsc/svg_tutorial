# SVG-MLLM：基于 SVG 的多模理解生成一体化大模型（中文教程）

本教程面向“**理解 + 生成一体**”的多模态大模型（MLLM），以 **SVG（可解析的文本矢量图形）**为核心媒介，系统讲解如何把 **SVG 字符串 ↔ SVG 渲染图像 ↔ 自然语言**三者打通：  
- 让模型不仅能“看懂 SVG 渲染出来的图”，还能“看懂 SVG 源码结构”；  
- 让模型不仅能“生成一张图”，还能“生成可编辑、可渲染、可动画的 SVG”；  
- 让模型能做“理解→生成”“生成→可控编辑”“跨模态对齐与检索”“动画与交互”等任务。

> 文件组织：`index.md` + `chapter1.md` ... `chapter16.md`（共 16 章）

---

## 你将完成的能力清单

- **SVG 作为可学习语言**：从 XML/DOM/几何语法入手，把 SVG 变成可建模的结构化 token/AST/图表示  
- **理解与生成一体化**：SVG→文本、文本→SVG、SVG 编辑、多模态对话与工具调用式生成  
- **渲染闭环**：结合 `resvg` 等渲染引擎 + 可微渲染/近似渲染思路，建立“生成→渲染→监督/评测”的闭环  
- **覆盖研究谱系**：从传统 image tracing 到 DeepSVG，再到 StarVector / OmniSVG / InternSVG 等（并扩展更多相关工作）  
- **应用落地**：SVG animation、字体/字形、以及 BEV 矢量地图（自动驾驶/地图场景）的理解与生成

## 目录（16 章）

> 每章文件：`chapterN.md`  
> 章节内小节编号采用 `N.M`（例如 3.2 表示第 3 章第 2 节）

---

### [第 1 章：从 SVG 到 SVG-MLLM：问题定义与路线图](chapter1.md)

- 1.1 为什么选择 SVG：文本可解析、可编辑、可结构化监督  
- 1.2 “理解-生成一体”的核心任务：caption、QA、编辑、规划式生成  
- 1.3 矢量 vs 栅格：表达能力、可控性、可扩展性与评测差异  
- 1.4 SVG 与网页数据：天然规模化语料、与上下文文本联动  
- 1.5 典型应用：图标/插画/流程图/仪表盘/地图/字体/动画  
- 1.6 总体技术栈：解析→规范化→建模→渲染→训练→评测  
- 1.7 训练闭环概念：生成 SVG → 渲染成图 → 多目标监督  
- 1.8 本教程的实验路线：从 baseline 到一体化 MLLM  
- 1.9 你将构建的最小系统（MVP）定义  
- 1.10 练习与项目建议（每章都有可落地任务）

---

### [第 2 章：SVG 核心语法：从 XML 到几何表达](chapter2.md)

- 2.1 SVG 文档结构：`<svg>`、命名空间、viewBox 与 viewport  
- 2.2 坐标系统与单位：px、mm、百分比、用户坐标与变换  
- 2.3 基本图元：rect/circle/ellipse/line/polyline/polygon  
- 2.4 `<path>` 语言：M/L/H/V/C/S/Q/T/A/Z 命令与参数语义  
- 2.5 变换系统：translate/scale/rotate/skew/matrix 与组合顺序  
- 2.6 样式系统：fill/stroke/opacity/linecap/linejoin/dasharray  
- 2.7 高级特性：gradient/pattern/mask/clipPath/filter  
- 2.8 文本相关：`<text>`、tspan、textPath、baseline、glyph  
- 2.9 元数据与可访问性：`<title>` / `<desc>` 与语义增强  
- 2.10 练习：手写一组可复用 SVG（组件化 + 参数化）

---

### [第 3 章：SVG 与 Web 联动：DOM、CSS、JS 与 three.js 协同](chapter3.md)

- 3.1 浏览器渲染管线：解析、布局、绘制与合成（SVG 的特殊性）  
- 3.2 SVG DOM：节点树、属性、事件绑定与动态修改  
- 3.3 CSS in SVG：样式继承、选择器、变量与主题切换  
- 3.4 JS 交互：事件模型、hit-testing 与交互式可视化  
- 3.5 外部资源引用：`<image>`、`<use>`、symbol/sprite 与安全  
- 3.6 SVG 与 Canvas/WebGL：何时栅格化，何时保留矢量  
- 3.7 three.js 与 SVG：SVGLoader、ShapeGeometry、路径挤出与渲染  
- 3.8 SVG 的“文字-图像联动”：源代码可定位到渲染区域  
- 3.9 性能与兼容：filter、复杂 path、字体与渐变的坑  
- 3.10 练习：做一个“选中元素→定位源码→可视化编辑”的 demo

---

### [第 4 章：Web SVG 数据工程：采集、清洗、规范化与对齐](chapter4.md)

- 4.1 SVG 在网页中的形态：inline / 外链文件 / icon system  
- 4.2 大规模采集：爬取策略、去重、版本与增量更新  
- 4.3 规范化（canonicalization）：属性排序、单位统一、viewBox 对齐  
- 4.4 清洗：脚本/外链、不可渲染特性、非法 XML 与异常修复  
- 4.5 结构抽取：DOM/AST、path 分解、层级与 group 语义  
- 4.6 文本对齐：标题/周边文案/aria-label 与多粒度 caption  
- 4.7 质量过滤：极端复杂度、异常尺寸、无意义噪声 SVG  
- 4.8 数据集切分：按站点/相似度/语义去泄漏  
- 4.9 标注策略：弱监督、合成指令、自动 QA 生成  
- 4.10 练习：搭建一条可复现实验的数据流水线（dataset card）

---

### [第 5 章：传统矢量化与图像追踪（Image Tracing）算法](chapter5.md)

- 5.1 任务定义：raster → vector 的目标与可编辑性要求  
- 5.2 预处理：去噪、平滑、二值化与多色量化  
- 5.3 边缘检测与轮廓提取：连通域、拓扑与层级轮廓  
- 5.4 轮廓简化：Ramer–Douglas–Peucker（RDP）等近似  
- 5.5 曲线拟合：样条、贝塞尔控制点估计与误差度量  
- 5.6 分层与填充：区域分割、叠放顺序与遮挡处理  
- 5.7 Potrace/AutoTrace 的典型思路与工程取舍  
- 5.8 传统方法的系统性误差：锯齿、拐点、细节丢失与过拟合  
- 5.9 学习式方法对照：可泛化、可控、可对齐语义  
- 5.10 练习：实现一个“轮廓→贝塞尔→SVG path”的最小追踪器

---

### [第 6 章：SVG 结构化表示：从文本到 Token / AST / 图](chapter6.md)

- 6.1 为什么“把 SVG 当普通文本”会困难：长程依赖 + 几何约束  
- 6.2 Tokenization 方案谱系：字符级 / 子词级 / 指令级 / 参数量化  
- 6.3 AST/DOM 表示：元素类型、属性、层级与作用域（defs/use）  
- 6.4 Path 表示：命令序列、控制点、相对/绝对坐标、闭合语义  
- 6.5 归一化：坐标规范、尺度归一、transform 展开与合并  
- 6.6 几何合法性：自交、退化段、fill-rule、方向性与拓扑一致  
- 6.7 图表示：节点（segment/point/object）与边（连接/层级/邻接）  
- 6.8 约束解码：语法约束（XML/grammar）+ 几何约束（range/topology）  
- 6.9 round-trip：解析→生成→再解析一致性与可维护性  
- 6.10 练习：写一个 SVG canonicalizer + validator（训练前必备）

---

### [第 7 章：渲染引擎与训练闭环：resvg 与 PyTorch-SVGRender](chapter7.md)

- 7.1 渲染在“理解-生成一体”中的地位：监督、评测与可视化  
- 7.2 SVG 栅格化基础：tessellation、抗锯齿、alpha compositing  
- 7.3 `resvg`：渲染一致性、支持特性、命令行/库集成  
- 7.4 浏览器 vs 离线渲染：差异来源与对齐策略  
- 7.5 可微渲染的需求：让 loss 反传到控制点/参数  
- 7.6 `PyTorch-SVGRender`：接口设计、batch 渲染与可训练管线  
- 7.7 视觉监督设计：像素级、感知特征（ViT/CLIP 特征）与多尺度  
- 7.8 特性支持：渐变、mask、clipPath、filter 的训练近似  
- 7.9 性能工程：缓存、并行、分辨率 curriculum  
- 7.10 练习：把“SVG→tensor”做成可复用 dataloader + renderer 模块

---

### [第 8 章：DeepSVG：学习式 SVG 表示与生成基线](chapter8.md)

- 8.1 DeepSVG 的问题设定：图标/插画的矢量序列建模  
- 8.2 数据表示：命令序列、参数化与量化策略  
- 8.3 Encoder：层级建模（segment/path/object）与 transformer  
- 8.4 Decoder：自回归生成与停止条件、层级展开  
- 8.5 VAE/latent：可插值、可控生成与语义聚类  
- 8.6 训练目标：重建损失、KL、辅助约束与后处理  
- 8.7 合法性与可渲染性：约束解码与修复策略  
- 8.8 复现实验：训练配置、数据规模与常见坑  
- 8.9 误差分析：几何漂移、路径断裂、层级错乱  
- 8.10 练习：复现 DeepSVG baseline 并输出可视化报告

---

### [第 9 章：从 Stroke 到 Path：Sketch 系列思想与贝塞尔生成](chapter9.md)

- 9.1 stroke-based 建模（SketchRNN 等）的直觉与优势  
- 9.2 stroke 与 SVG path 的差异：离散点 vs 连续曲线  
- 9.3 贝塞尔参数化：控制点空间、曲率与稳定性  
- 9.4 由 stroke 拟合到贝塞尔：分段、平滑与约束  
- 9.5 条件生成：布局、风格、粗细、颜色与多对象组合  
- 9.6 结构先验：对称性、重复、对齐与网格系统  
- 9.7 渲染监督 vs 结构监督：什么时候该用哪种 loss  
- 9.8 交互式编辑：局部重绘、局部约束与可解释控制  
- 9.9 练习：实现 stroke→SVG path 的拟合工具  
- 9.10 练习：做一个“文本条件→可控 SVG 图标生成”demo

---

### [第 10 章：现代 SVG 工作谱系综述：StarVector、OmniSVG、InternSVG 等](chapter10.md)

- 10.1 研究版图：icon / illustration / diagram / map / UI 元素  
- 10.2 StarVector：核心设定、数据与模型要点  
- 10.3 OmniSVG：统一表征、多任务训练与泛化讨论  
- 10.4 InternSVG：大模型化、跨模态对齐与指令化趋势  
- 10.5 相关方向补全 I：可微矢量（DiffVG 等）与端到端优化  
- 10.6 相关方向补全 II：CLIP 引导的矢量化/描摹（如 CLIPasso 思路）  
- 10.7 相关方向补全 III：图标生成、排版与设计工具链（Icon 类工作）  
- 10.8 相关方向补全 IV：文本到矢量（Transformer/扩散/混合范式）  
- 10.9 数据与评测：不同工作为何不可直接对比  
- 10.10 练习：选择 1 个工作做“复现/复述/消融 + 复用到你的系统”

---

### [第 11 章：SVG-MLLM 架构设计：理解与生成一体化](chapter11.md)

- 11.1 任务集合设计：SVG captioning、SVG QA、text-to-svg、svg-to-text、svg editing  
- 11.2 输入端一：SVG encoder（token/AST/graph）设计与对齐策略  
- 11.3 输入端二：渲染图像 encoder（ViT/Conv）与多分辨率  
- 11.4 输出端：受约束 SVG decoder（grammar + geometry constraints）  
- 11.5 融合机制：cross-attention、late fusion、工具式渲染回路  
- 11.6 多粒度对齐：token↔path↔object↔caption 的对齐监督  
- 11.7 指令化接口：把 SVG 生成/编辑包装成可对话能力  
- 11.8 推理与采样：beam / nucleus / constraint decoding / repair  
- 11.9 练习：实现一个“SVG + 图像 + 文本”的多模输入原型  
- 11.10 练习：实现“局部编辑指令：把圆角加大/换颜色/对齐到中心”等

---

### [第 12 章：训练流程：预训练、指令微调、偏好对齐与有效性保障](chapter12.md)

- 12.1 预训练目标 I：SVG 自监督（MLM/denoise/AR）  
- 12.2 预训练目标 II：跨模态对齐（对比学习、匹配、caption）  
- 12.3 指令数据构造：模板合成、程序化扰动、自动 QA 与自举  
- 12.4 有效性保障：合法 XML、可渲染性、参数范围与异常修复  
- 12.5 约束解码工程：CFG / grammar-guided decoding / schema 校验  
- 12.6 偏好对齐：pairwise preference、DPO/RLHF 风格（面向“更可用 SVG”）  
- 12.7 课程学习（curriculum）：从简单图元到复杂组合与动画  
- 12.8 训练稳定性：长序列、padding/mask、混合精度与梯度裁剪  
- 12.9 安全与合规：外链资源、脚本注入、潜在 XSS 风险  
- 12.10 练习：训练一个“文本→SVG”并加入“自动校验+修复”后处理

---

### [第 13 章：评测体系：像素、结构、语义与可编辑性](chapter13.md)

- 13.1 为什么不能只看像素相似度：结构可编辑性是关键  
- 13.2 渲染评测：PSNR/SSIM/LPIPS 与多尺度一致性  
- 13.3 结构评测：path 数、命令分布、控制点复杂度、层级深度  
- 13.4 语法与可渲染性：parse 成功率、渲染失败率、fallback 行为  
- 13.5 可编辑性评测：局部编辑代价、稳定性、可预测性  
- 13.6 语义评测：caption/QA 任务准确率与鲁棒性  
- 13.7 人评设计：pairwise、rubric、应用导向（可用性/美观/一致）  
- 13.8 鲁棒性：缩放、旋转、噪声、重排属性与 round-trip  
- 13.9 数据泄漏与去重：近重复 SVG 的识别与严格切分  
- 13.10 练习：搭建一个可复用的 eval harness（自动出报告 + 可视化）

---

### [第 14 章：SVG Animation：时间维度、交互与可控运动生成](chapter14.md)

- 14.1 SVG 动画生态：SMIL、CSS animations、JS timeline  
- 14.2 `<animate>`/`<animateTransform>`：关键帧、属性插值与 easing  
- 14.3 Path morphing：点数匹配、拓扑一致与形变约束  
- 14.4 分层动画：group/symbol/use 的复用与实例化  
- 14.5 交互式动画：事件驱动、状态机与可视化反馈  
- 14.6 生成任务：text-to-animated-svg、svg-animation editing  
- 14.7 约束：连续性、速度/加速度平滑、视觉稳定与闪烁控制  
- 14.8 评测：时序一致性、可读性与用户感知指标  
- 14.9 练习：生成一个 loading/hover/transition 动画组件  
- 14.10 练习：实现“指令编辑动画：加快/减慢/延迟/改变路径形变”

---

### [第 15 章：应用专题 I：字体、字形生成与排版（SVG × Typography）](chapter15.md)

- 15.1 字体与矢量：glyph 轮廓、曲线质量与可读性  
- 15.2 SVG 与字体生态：glyph 导出、OpenType/WOFF 与渲染差异  
- 15.3 字形数据集：字库构建、风格标签、字重与变体管理  
- 15.4 任务 I：字形补全（缺字生成）与风格一致性  
- 15.5 任务 II：风格迁移、字重插值与可控生成  
- 15.6 约束：笔画连贯、端点处理、曲率、闭合与自交  
- 15.7 排版系统：文本布局→SVG 版面生成（对齐、间距、层级）  
- 15.8 评测：可读性、风格一致、排版美学与应用可用性  
- 15.9 练习：生成一个小型 SVG 字库（含导出与浏览器预览）  
- 15.10 工程：从模型输出 SVG glyph 到字体文件/网页字体集成

---

### [第 16 章：应用专题 II：BEV 矢量地图与系统落地（SVG × Map/Driving）](chapter16.md)

- 16.1 BEV 与矢量地图表示：lane/boundary/crosswalk/polyline/polygon  
- 16.2 数据来源：HD Map、矢量标注、仿真与弱监督构造  
- 16.3 任务 I：感知→矢量地图（生成 polylines / polygons）  
- 16.4 任务 II：矢量地图补全与拓扑修复（连通性、车道关系）  
- 16.5 模型范式：set prediction、polyline decoder、transformer 结构先验  
- 16.6 SVG 映射：坐标系、尺度归一、分层语义（图层=类别）  
- 16.7 评测：几何误差、拓扑一致、可导航性与下游增益  
- 16.8 可视化与交互：前端渲染、three.js 联动、增量更新  
- 16.9 工程安全：输入校验、资源隔离、可渲染性与性能守护  
- 16.10 结语：统一“图形语言”视角下的研究机会与扩展阅读清单

---

## 建议阅读路线（可选）

- **工程优先**：第 2 → 6 → 7 → 11 → 12 → 13（先搭闭环）  
- **研究优先**：第 5 → 8 → 10 → 11 → 12（先读方法谱系）  
- **动画/字体/地图**：第 14 / 15 / 16 独立成专题路线（依赖第 2/6/7 的基础）

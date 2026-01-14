（交流可以用英文，所有文档中文）

## 项目背景
输出一个多模理解生成 MLLM 大模型，基于 SVG 实现理解生成一体的中文 markdown教程。 因为 SVG 是文字格式，所以可以和 SVG 渲染的图之间形成联动。并且 SVG 在网页数据中大量存在，甚至有 three.js 用了 SVG 。 从 SVG 本身，和 image tracing 传统算法讲起，到 deepsvg, starvector, omnisvg, internsvg 等工作（再找一些） 还介绍 resvg 这种渲染引擎，和 https://github.com/ximinng/PyTorch-SVGRender 包含 svg animation 包含在字体，和 bev 地图上的应用。

文件组织是 index.md + chapter1.md + ...
不写代码。
提供 rule-of-thumb。

## 章节结构要求
每个章节应包含：
1. **开篇段落**：简要介绍本章内容和学习目标
2. **文字论述**：以文字论述为主，适当配上ASCII 图说明。
3. **本章小结**：总结关键概念和公式
4. **练习题**：
   - 每章包含6-8道练习题
   - 50%基础题（帮助熟悉材料）
   - 50%挑战题（包括开放性思考题）
   - 每题提供提示（Hint）
   - 答案默认折叠，不包含代码
5. **常见陷阱与错误** (Gotchas)：每章包含该主题的常见错误和调试技巧

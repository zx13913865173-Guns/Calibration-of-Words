# 复现工作流（中文逐步指南）

对应论文：*The Calibration of Words: A Comparative Study of Three Prompt Weight
Syntaxes on Semantic Control Precision in Diffusion Models*（Zhang & Galkin）。

仓库里所有实验都被拆成四个可独立执行的阶段。**阶段 1 不需要 GPU**，只有阶段 2
（真正生成图像）需要 CUDA。

---

## 阶段 0：环境

```bash
conda env create -f environment.yml
conda activate calibration-of-words
```

论文报告的软件栈：Python 3.10、PyTorch 2.x、CUDA 12.x、Diffusers v0.25.0、
Automatic1111 WebUI v1.6.0、ComfyUI v0.2.2、SDXL base 1.0、CLIP-L、
DINOv2-large。基础模型权重需自行下载到 Hugging Face 缓存（SDXL 约 7 GB）。

---

## 阶段 1：构建清单并自检

```bash
bash scripts/01_build_manifests.sh
```

这一步做四件事：

1. 跑 `tests/test_syntaxes.py`（9 项）与 `tests/test_metrics.py`（16 项）——
   校验三种语法的渲染/解析往返、权重映射表、以及各项指标的实现口径。
2. 跑 `scripts/verify_reference.py`——把 `data/reference_results.json` 里论文
   报告的每一个数值，用其"可从汇总值确定性反算"的方式重算一遍，逐条 PASS/FAIL。
3. 把 `prompts/` 里的提示词列表展开成 `outputs/manifests/*.jsonl`，并核对图像
   数量（900/225/375/80/120/150/40，合计 1 890）。
4. 跑两条评价脚本的自检。

**任何图像都还没生成，但整个数据流已经验证过了。** 如果你只想确认代码正确，
只跑到这一步即可。

---

## 阶段 2：生成图像

### 2a 主后端（Diffusers + 受控解析器补丁）

```bash
python -m src.generation.generate_diffusers \
  --experiment main two_word ablation migration generalization nonlinear_mapping calibration \
  --manifests outputs/manifests --out outputs/images
```

这是论文的核心生成路径。三种语法**在同一条 SDXL 管线**上运行：语法只决定权重
系数，系数被施加到文本编码器内部的 token embedding 上（见
`src/prompt_weighting/embedding_patch.py`）。这样后端差异不会混进语法对比。

采样参数固定为 1024×1024、30 步、CFG 7.5、Euler、固定种子 42。

### 2b 工具内验证（Automatic1111）

```bash
# 先启动服务（需装 NovelAI prompt-weights 扩展 v1.3）
./webui.sh --api --nowebui
python -m src.generation.generate_a1111 --experiment main \
  --manifests outputs/manifests --out outputs/images
```

SD 原生与 NovelAI 两种语法在真实 WebUI 里生成，用于 `migration` 实验的
"迁移前"基准。

### 2c 工具内验证（ComfyUI）

```bash
python main.py --listen 127.0.0.1 --port 8188     # 在工作流目录下
python -m src.generation.generate_comfyui --experiment main nonlinear_mapping \
  --manifests outputs/manifests --out outputs/images
```

工作流模板：`workflows/comfyui_sdxl_weighted.json`（API 格式，节点编号不可改，
脚本按编号打补丁）。

---

## 阶段 3：打分

```bash
bash scripts/03_score.sh
```

对每张图计算两个指标：

* **CLIP-L 语义控制得分**：图像嵌入与"目标词文本嵌入"的余弦相似度
  （`src/evaluation/clip_score.py`）。这是全文的精度代理量，每个
  （语法 × 权重等级）单元的 60 张图取均值，得到每种语法 5 个点。
* **DINOv2-large 视觉身份距离**：同一 `concept_key`（同提示词同种子）下
  W5 与 W1 两图的余弦距离（`src/evaluation/dinov2_distance.py`），衡量视觉身份
  漂移程度，论文报告为 0.24 / 0.41 / 0.33。

---

## 阶段 4：统计与作图

```bash
bash scripts/04_analyze.sh
```

* `src/analysis/stats.py` 输出单元均值/标准差表与双因素方差分析表
  （语法 × 权重，含交互项，报告偏 η²）。
* `src/analysis/figures.py` 依据 `data/reference_results.json` 重绘论文
  图 2–图 5。

**没有 GPU 也想验证阶段 4**：先跑
`python scripts/demo_scores_from_reference.py`，它会按照论文公布的单元均值与
标准差生成一份**模拟**评分表，让统计与作图流程能空转跑通（单元均值会与论文
一致，方差分析的 F 值不会——F 值取决于单元间信号，不是 5 个均值点能完全
决定的；确定性校验请以 `scripts/verify_reference.py` 为准）。

论文报告的统计量：F(2, 885) = 0.21（语法，n.s.）、F(4, 885) = 745（权重）、
F(8, 885) = 13.4（交互）；偏 η² = 0.771（权重）、0.108（交互）、<0.001（语法）。

---

## 常见问题

**Q：一定要 7 张卡/多 GPU 吗？**
不需要。1 890 张 1024×1024 图，单张 24 GB 卡可分两三天跑完；也可以只跑
`main`（900 张）先复现主结论。

**Q：没有 GPU 能复现什么？**
阶段 1 的全部自检、阶段 3 的聚合、阶段 4 的统计与作图都能跑；论文里所有
"汇总值可反算"的数字都会被验证。

**Q：为什么 `verify_reference.py` 里"线性映射最优"只是 informational？**
因为 R² 与映射优劣是**图像层面**统计量，只用 5 个均值点无法还原；脚本据实
降级提示，单元测试改在理想线性数据上验证变换识别逻辑。要拿到该结论请跑完
阶段 3 用真实图像重算。

**Q：数字对不齐怎么办？**
论文的确定性统计量（方差分析、动态范围、压缩率、标定区间）应当完全一致；
图像层面统计量允许小幅偏差，取决于 GPU 与驱动。先核对
`configs/weight_mapping.yaml` 与 `configs/experiment.yaml` 未被改动。

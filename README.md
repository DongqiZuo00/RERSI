# RepRSI: minimal anonymous implementation

依据所提供论文的第 3 节及附录 B、C、D 重建。保留教师课程生成、学生分支训练、表征测量、教师更新、分支延续和基本评估；不包含绘图、网页、实验表格或批量消融框架。

**这是论文方法的可运行参考实现，不是原实验代码的恢复副本。** 原始诊断清单、校准轨迹、训练日志与最终权重没有随论文提供。本包生成新的确定性诊断数据，并明确记录必要的实现选择；没有把论文表格中的数值写成程序输出，也没有执行论文规模的 Gemma 训练。

## 1. 安装和最短运行

Python 3.11 或更高版本。在解压目录运行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m reprsi smoke --output runs/smoke
python -m unittest discover -s tests -v
```

Windows 激活命令为 `.venv\Scripts\activate`。GPU 训练应按本机 CUDA 环境安装匹配的 PyTorch 构建。

`smoke` 在 CPU 上训练一个随机初始化的小型因果 Transformer，运行两轮真实 RLOO 和分支选择。为缩短检查时间，它使用有限课程/答案词表、较高学习率和较小预算。**它只验证程序流程，不能作为论文实验。** 输出目录必须是新目录，避免覆盖已有训练结果。

## 2. 核心代码位置

| 文件 | 实现内容 |
|---|---|
| `reprsi/loop.py` | 共同起点、配对试验、教师更新、固定副本延续、计算成本记录 |
| `reprsi/policy.py` | 模型加载、约束采样、RLOO 梯度、KL、隐藏状态读取、模型和优化器保存/恢复 |
| `reprsi/metrics.py` | 状态平衡的 C+、C−、Φ、奖励归一化、RLOO 优势、pass@k |
| `reprsi/tasks.py` | 四类精确数学课程构造器与答案验证 |
| `reprsi/schema.py` | 固定教师指令和课程 JSON 结构 |
| `reprsi/diagnostics.py` | 诊断池、语义标签、共享诊断批次、读出位置 |
| `reprsi/calibrate.py` | 独立轨迹上的语义 patching 和层选择 |
| `reprsi/manufactoria.py` | 官方 DELTA 构造器/执行器适配、最小化 DFA 诊断 |
| `reprsi/benchmarks.py` | MATH/HARP 分区和官方 HARP 检查器适配 |
| `reprsi/__main__.py` | 运行入口、fail@128 筛选和基本评估 |

## 3. 与论文对齐的默认设置

`configs/math.json` 用于数学课程，分别启动 MATH 与 HARP 实验。`configs/manufactoria.json` 用于 Manufactoria。

| 项目 | 默认值 |
|---|---|
| 模型 | `google/gemma-4-E4B-it`，论文给出的固定 revision |
| 语言模型参数训练 | BF16 前向、FP32 主权重和 AdamW 状态、全参数更新、关闭 thinking；只加载语言部分 |
| 轮数 / 候选数 / 课程长度 / 副本数 | 100 / 4 / 32 / 2 |
| 每分支更新 | 8 步，每步 8 个题目，每题 4 个回答 |
| 训练顺序 | 有序课程完整遍历两次 |
| 教师 / 学生学习率 | 1e-6 / 2e-6，恒定 |
| AdamW | betas=(0.9,0.95)，eps=1e-8，weight_decay=0 |
| KL 系数 / 梯度范数裁剪 | 0.01 / 1.0 |
| 采样 | temperature=1，top_p=1，top_k=0 |
| 训练输入 / 输出上限 | 2048 / 4096 tokens |
| 评估输入 / 输出上限 | 16384 / 4096 tokens |
| 奖励诊断池 / 监控池 / 每轮诊断 | 4096 / 1024 / 256 |

重要细节：

- Φ 使用同类型下同状态和异状态的余弦相似度差。先对状态等权，再对类型/家族等权。对批内所有合法配对精确求平均，避免端点之间重新抽样带来的噪声。
- 同一轮起点与所有端点使用同一批诊断。读出在指定 decoder block 后、子计算输入最后一个 subtoken；中间答案不作为模型输入。
- 每次学生试验恢复同一份参数和优化器，随机种子仅由根种子、轮数和副本索引确定，不含候选索引。
- 有效候选奖励使用总体标准差归一化，然后给无效候选 −5。无效候选参与 RLOO 基线，不参与分支选择。全部无效时保留学生并跳过教师更新。
- 选择使用副本平均后的**原始奖励**；相同奖励选最小候选索引。始终继续选中课程的第一个副本，包含其优化器状态。
- 教师损失使用完整课程序列的 log-probability 之和；约束掩码用于采样、概率重算和参考策略。不会以无约束概率替代约束策略概率。
- 学生仅按最终答案正确性训练，不接收表征损失、诊断标签或表征向量。

## 4. 数学域：从诊断到完整训练

先生成诊断：

```bash
python -m reprsi prepare-diagnostics --domain math --output data/math_diagnostics
```

这里生成四类计算：有理数、多项式、模运算和线性系统。每类奖励池有 64 个 specification，每个含 4 个状态和每状态 4 个上下文；监控池每类 16 个 specification。独立校准池默认每类 4 个 specification。

论文没有给出实际校准层，也没有附原始校准轨迹。因此代码提供独立校准训练入口，并通过真实 patching 选择层，不预填猜测的层号：

```bash
python -m reprsi calibration-trajectory \
  --config configs/math.json --batches 2 --output runs/math_calibration

python -m reprsi calibrate --config configs/math.json \
  --diagnostics data/math_diagnostics/calibration.jsonl \
  --checkpoints runs/math_calibration/step_000.pt runs/math_calibration/step_008.pt runs/math_calibration/step_016.pt \
  --output runs/math_readout.json

python -m reprsi train --config configs/math.json \
  --diagnostics data/math_diagnostics/reward.jsonl \
  --calibration runs/math_readout.json --seed 0 --output runs/math_seed0
```

校准默认对所有 decoder blocks、每家族一个平衡 specification 的 64 个例子进行 patching，跨三个独立校准检查点平均，平局选择较浅层。`--specs-per-family` 可扩大校准集；`--layers` 仅用于检查指定层。校准后的层在整个递归运行中固定。

每轮保存完整候选 JSON、无效原因、各副本奖励、选中索引以及成本；最终 `student.pt`、`teacher.pt` 均含模型和优化器。`initial.pt` 是本次初始化。参考策略一直固定为该初始化。

分别使用根种子 0–4 启动五次训练。数学课程接口相同，但 MATH 与 HARP 应采用不同运行目录，分别评估。

## 5. MATH / HARP 数据与评估

安装评估依赖并获取官方 HARP 仓库：

```bash
python -m pip install -r requirements-benchmarks.txt
git clone https://github.com/aadityasingh/HARP.git third_party/HARP
git -C third_party/HARP checkout dac2734ff6443bcaf3bbdcb10f13cf21ae9729c2
python -m zipfile -e third_party/HARP/HARP.jsonl.zip data/harp_raw
python -m reprsi prepare-harp --source data/harp_raw/HARP.jsonl --output data/harp
```

对已下载并解压的原版 MATH（包含 `train/`、`test/` 目录）：

```bash
python -m reprsi prepare-math --source data/MATH --output data/math
```

MATH 分区为 6750/750/5000，HARP 为 3442/382/956。HARP 发布文件含 `2021_Fall` 形式的年份：本实现按前四位数值年份、contest、数值题号排序，等值键保持发布顺序，再用 `random.Random(2026)` 打乱。**论文没有提供季节年份的并列规则和原始分区清单，因此不能保证与原实验逐题一致。** 原始字符串年份保留在题目身份中。

全测试集 sampled evaluation：

```bash
python -m reprsi evaluate --config configs/math.json \
  --checkpoint runs/math_seed0/student.pt --data data/math/test.jsonl \
  --harp-root third_party/HARP --samples 32 --output runs/math_seed0/test32.jsonl
```

加 `--greedy` 则每题只用一个贪婪回答。MATH 和 HARP 均调用固定版本 HARP checker；生成课程继续使用构造器专用的精确验证器。只抽取最后一个 boxed answer，缺失答案记为错误。

fail@128 筛选与独立评估：

```bash
python -m reprsi screen --config configs/math.json --data data/math/test.jsonl \
  --harp-root third_party/HARP --output runs/math_screen_test.jsonl

python -m reprsi evaluate --config configs/math.json \
  --checkpoint runs/math_seed0/student.pt --data runs/math_screen_test.hard.jsonl \
  --harp-root third_party/HARP --samples 128 --output runs/math_seed0/hard128.jsonl
```

筛选固定使用初始模型，每题始终采样 128 次，不在首次成功后停止。正式评估使用另一随机流。训练/开发/测试分别筛选；`screen` 不接受训练后的检查点。所有汇总由实际 verifier 结果计算，分数为 [0,1]，不是百分数。

Target-grounded Teacher 的最小对照：复制配置，将 `feedback` 改为 `target`；对 `train.jsonl` 筛选后，以 `--target-train runs/math_screen_train.hard.jsonl --harp-root third_party/HARP` 启动。训练入口强制检查该数据来自训练集且为 128 次全失败。`teacher_updates=false` 可运行固定教师的 matched recursive search。

## 6. Manufactoria

```bash
git clone https://github.com/sunblaze-ucb/rl-grok-recipe.git third_party/DELTA
git -C third_party/DELTA checkout 8500bec984d4a84a4aa94ca3adc31c004aa6a388
python -m reprsi prepare-diagnostics --domain manufactoria --output data/factory_diagnostics
```

之后使用第 4 节同样的三步流程，将配置换成 `configs/manufactoria.json`，路径换为 `factory_diagnostics`，并给 `calibration-trajectory`、`calibrate`、`train` 加 `--delta-root third_party/DELTA`。

适配器直接调用官方 START、APPEND、EXACT、REGEX、COMPR、HAS 构造器及 DSL 执行器，奖励为全部测试通过。评估入口支持官方原始 JSONL 和 `messages/ground_truth/id` 格式 JSONL；必须自行提供官方 train/test 文件，包内不重新划分或伪造 HAS 的 103 个测试实例。

HAS/REGEX 诊断编译为确定性有限自动机并最小化，以 residual-language state 定义等价关系。每类奖励池 128 个 specification，每轮选 8 个，每个 specification 4 状态 × 4 上下文，合计 256。生成后的校准、奖励和监控池互不重叠。

DELTA 的上述 commit 是本适配器核验过的公开版本；论文只提供了仓库地址，没有锁定原实验 commit。二者不能自动视为完全相同的原始实验环境。

## 7. 成本、实现选择和复现边界

`compute.jsonl` 分阶段记录生成 token 数、同步后的 wall time、累计 GPU·s。分支按顺序执行；默认策略和参考模型在同一 GPU。可在配置中增加 `"reference_device": "cuda:1"`，把冻结参考模型放到第二张 GPU；计时按实际占用的不同 GPU 数量累加。CPU smoke 的 GPU 成本始终为 0。该计时是预留设备的阶段时长，不是 FLOPs，也不是 GPU 利用率积分。`calibrate` 将独立校准训练和层选择成本写入报告，`train` 自动计入它们。

训练/开发集筛选的摘要可通过 `--preparation-costs FILE.summary.json ...` 等额计入各对照；测试筛选和正式评估不计入训练。比较不同方法时应在同硬件、精度、提示/验证接口下使用总预算；本最小包不会把相同 round 数自动宣称为相同总计算量。

以下是明确的重建选择：

1. KL 采用采样前缀上的逐 token **完整词表 forward KL**，累加后乘 0.01，与任务 RLOO 项分开。论文没有指定具体 KL 估计器；这里未采用额外 advantage 归一化或 PPO clipping。
2. 诊断渲染、构造器 JSON 参数编码、校准训练课程、校准采样量、DFA 实例由本实现固定生成。它们遵循给出的语义接口，但不是作者原始清单。
3. 同类配对使用全配对精确平均；固定 batch 内没有额外 Monte Carlo 配对误差。
4. MATH/HARP 的完整实验结果、论文第 2 节的二跳任务/干预实验、全部 baseline/消融、曲线与置信区间不在这个核心代码包中。
5. 大模型训练需要自行取得模型权重和原版 benchmark。全参数模型、参考模型、梯度和 FP32 AdamW 状态需要较大显存及主机内存；磁盘还需容纳当前教师/学生与临时分支，完整权重的串行快照可能需要数百 GB。代码未将全参数训练暗改成 LoRA。未实测 Gemma 的峰值内存，不能保证单卡容纳该配置。

## 8. 匿名性与验证

代码不含作者、单位、邮箱、个人仓库地址、私人绝对路径或访问令牌。没有随包附带 Git 历史、原论文、缓存、模型权重、运行日志或用户文件。官方模型/数据集地址保留以明确依赖，不能删除后假装这些组件由本包提供。

实际检查结果见 `VALIDATION.md`。核心算法与接口经过小规模执行；论文规模训练和最终分数没有验证。

公开依赖参考：

- [Gemma 4 Transformers 文档](https://huggingface.co/docs/transformers/model_doc/gemma4)
- [LM Format Enforcer](https://github.com/noamgat/lm-format-enforcer)
- [MATH 官方仓库](https://github.com/hendrycks/math)
- [HARP 官方仓库](https://github.com/aadityasingh/HARP)
- [DELTA 官方仓库](https://github.com/sunblaze-ucb/rl-grok-recipe)

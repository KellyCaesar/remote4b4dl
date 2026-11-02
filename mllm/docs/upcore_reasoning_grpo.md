# B4DL 结构化几何推理与 GRPO

本实验在已有 B3 和 SST 特征上增加独立版本的结构化推理、规则奖励及 GRPO 训练入口。它是尚待完整 GPU 与任务效果验证的候选方案；不修改旧 `geometry_json_v1`、冻结实验配置或历史检查点。

2026-10-31云端跟进：8dc2c18版本的数据、环境和114项回归通过，50步真实7B SFT短跑及完整13,851步正式SFT终点核验通过。其后的GRPO短跑在完成18步后尝试第19步时因采样/重算概率超出裁剪保护退出，正式GRPO未启动。SFT终点完整几何与六任务效果评测已启动，尚无全量结果。实际配置和回执见[云端训练记录](../../docs/research/upcore云端训练_20261030.md)及[终点评测](../../docs/research/upcore_SFT终点评测_20261031.md)。

2026-11-02 本地修复：旧实现采样使用 KV 缓存逐 token 前向，GRPO 评分和反向则使用整段前向；BF16 下两者的运算形状不同，已有 FP32 输出头和严格累加仍不足以保证两条路径的概率落入裁剪界限。第 19 步记录的最大绝对 log-prob 差为 0.2754228，超过 `clip=0.2` 两侧界限。现有本地记录没有该步的逐 token logits 或完整回放现场，无法确认具体 CUDA 算子或断言本次修复已消除该步的实际偏差。

本次只改 GRPO 采样及评分的序列形状，不改普通评测生成。GRPO 对每题按 `max_new_tokens` 固定补齐，逐 token 对相同完整形状做无缓存因果前向，从原始 logits 的 softmax 采样并保存被抽中 token 的实际 log-prob；遇 EOS 或预算结束即停止。训练和参考评分使用相同形状，EOS 占位在因果遮罩下不会进入前面 token 的上下文；实际生成的 EOS 参与评分，后续占位不参与损失。该 GRPO 采样明确使用无温度变化、无 top-k/top-p 截断的完整 softmax，不依赖生成配置中的其他 logits 处理器。

本地 Python 3.10 / PyTorch 2.8 CPU 选择性回归 33 项通过，包含微型模型 FP32、CPU BF16、采样概率、EOS/截断/非法 token、因果遮罩、梯度和保护阻止更新；其中 31 项检查期间禁止真实 `AdamW.step`，另外 2 项只检查六任务协议，没有启动训练。旧 GRPO 检查点、SFT 终点和远端现场未回放；新实现的 CUDA/真实 7B 概率、显存和耗时，特别是固定形状逐 token 全序列前向可能显著增加耗时，均未验证。正式 GRPO 和效果评测未运行。

注意源码身份边界：本次修改改变 `implementation_identity`，现有 8dc2c18 的 SFT/GRPO 检查点与新源码不再满足严格身份匹配，不能直接用新源码加载旧 SFT 终点做推理或作为 GRPO 起点；旧续跑也不能混用。对本地 `code-8dc2c18.tar` 按实现身份算法重算，得到已记录的 SFT 源码身份 `d0e9c996aa9edb788fc01b3c12ddfc62e55a2f191276db1f887a114bd9e37420`；用本次修改的两个 reasoning 文件替换对应哈希后，身份发生变化。应保留原源码读取旧产物，并另行设计和验证可追溯的身份兼容或迁移方案；本次没有放宽校验或执行迁移。

## 数据与输出约定

`b4dl_core.reasoning build-data` 以既有几何数据集为父集，重新从 nuScenes 原始表、ego 位姿和输入裁剪支持记录核验每条监督；源表和支持文件允许换路径，但内容哈希必须与父数据清单一致。输出独立的 `requests.jsonl`、`supervision.jsonl`、`exclusions.jsonl` 和 `manifest.json`，模型输入不包含对象框、标注实例、几何真值及答案。场景划分沿用父集。对已知无法确认不存在的对象记录排除原因；来源、身份或标注真值不一致时停止构建。

当前独立数据协议为 `b4dl-reasoning-v2`：公开提示明确规定观测键的坐标、距离、运动差值含义，以及二分类答案必须使用 `"Yes"` / `"No"`。此前的 `b4dl-reasoning-v1` 数据和检查点保留原状；由于提示、过程标准化和源码身份已经改变，不能用新代码直接混用或续训旧产物。使用原始几何父集与标注在新目录重新构建和审计 v2 数据，再从 B3 重新启动相应实验。

每个回答严格以 `steps` 开头、`answer` 结尾。步骤分别使用 `observe`、`exists`、`sector`、`argmin`、`delta`、`predicate` 中该任务规定的操作，逐步引用前一项。存在性、方位、距离和相对运动分别使用固定模板；`motion` 估计两端距离和横向坐标，再计算差值。与原几何数据一样，运动量是两个时刻各自 ego 坐标中的相对变化，不是世界坐标速度。对象支持仅指历史 `CAM_BACK_RIGHT` 裁剪输入内的标注框点支持，不代表点级真值或完整全景。

观测数值按 1 m 容差连续计分，数量精确匹配；推导步骤还需与前序生成数值一致，得分不超过其依赖步骤。答案正确为 1、错误为 0；非法结构、禁止的特殊 token、未产生 EOS 的截断为 −1，步骤项不计分。缺失或矛盾的私有真值是错误，不能转成模型的失败奖励。评测的步骤通过率定义为对应分数大于 0，即米制误差严格小于 1 m 且依赖链通过；另报告连续步骤分数和有效数值的绝对误差。

## 准备与运行

使用 `mllm/docs/upcore_runtime.md` 的 Python 3.10 依赖栈。真实实验需要独立 CUDA 环境、支持 BF16 且空闲的单卡，以及已锁定的 B3 权重、SST 特征、nuScenes 表、裁剪支持记录和几何父集。启动时核验基底及 SST 绑定、训练数据身份、生成上下文长度；默认 GPU 锁沿用旧本体作业的锁。24 GB 是否足够必须由短跑峰值显存确认。先用独立输出目录完成工程短跑，不接管任何历史任务。

以下示例在仓库根目录运行，路径请按已核验资产替换；`BASE`、`PREP` 是已有本体清单，`DATA` 是旧几何父集。每个 `--output` 都必须是新路径。

```bash
export PYTHONPATH="$PWD/mllm"
PY=/path/to/wqlc/bin/python
DATA=/path/to/trainval-v1
TABLES=/path/to/nuScenes/v1.0-trainval
SUPPORT=/path/to/input-support.jsonl
BASE=mllm/configs/core_geometry_baseline_original_b3_trainval.json
PREP=mllm/configs/core_geometry_preparation.json
RECIPE=mllm/configs/reasoning_grpo_v1.json
RUN=/path/to/new-reasoning-run

"$PY" -m b4dl_core.runtime --require-cuda
"$PY" -m b4dl_core.reasoning build-data --geometry-dataset "$DATA" --tables "$TABLES" --support "$SUPPORT" --output "$RUN/data"
"$PY" -m b4dl_core.reasoning audit --dataset "$RUN/data" --geometry-dataset "$DATA" --tables "$TABLES" --support "$SUPPORT"

# 先进行有界工程检查；工程产物不视为完整训练或模型效果。
"$PY" -m b4dl_core.reasoning sft --dataset "$RUN/data" --geometry-dataset "$DATA" --baseline "$BASE" --preparation "$PREP" --recipe "$RECIPE" --engineering --max-updates 50 --output "$RUN/sft-smoke"
"$PY" -m b4dl_core.reasoning grpo --dataset "$RUN/data" --geometry-dataset "$DATA" --baseline "$BASE" --preparation "$PREP" --recipe "$RECIPE" --engineering --max-updates 50 --sft-checkpoint "$RUN/sft-smoke/step-50.pt" --output "$RUN/grpo-smoke"
```

工程 SFT 和 GRPO 的 `latest.json` 记录检查点及哈希，`training.jsonl` 记录损失、梯度、奖励、合法率、生成长度、旧策略概率误差、组内奖励零方差比例及 CUDA 峰值显存；`completion.json` 标注工程预算及累计峰值，不代表有效的全量模型。异常会写入 `failure.json`。记录身份不符、非有限数值、采样与重算概率偏差超出下述保护范围或显存不足时停止，不自动更改批量或配方。

短跑确认后，正式 SFT 使用同一条命令去掉 `--engineering --max-updates 50`，以训练集单轮、固定末步为终点。正式 GRPO 必须加载有完整预算完成回执的 SFT 最终检查点；选择 `--mode outcome`、`--mode scalar` 或 `--mode process`，各组使用相同数据、配置、种子及 SFT 终点。工程 GRPO 只接受有工程完成回执的工程 SFT 终点或已完成的正式 SFT。训练续跑使用相同数据、源码、配方和原总预算，指定 `--resume 旧检查点 --output 新目录`；只能从已保存更新恢复，不能把诊断或不同身份的检查点当作续跑输入。

默认配方见 `mllm/configs/reasoning_grpo_v1.json`：SFT 更新投影器、A3 时序适配器和 rank 16 的语言 LoRA；GRPO 冻结前两者，只更新策略 LoRA。参考 LoRA 是 SFT 终点的副本，与策略共享冻结基座。每题 4 条候选，逐条生成并反传，每 4 个问题更新一次。old-policy 概率取自采样时的策略，reference 概率来自固定的 SFT 参考策略。有效 token 的优势由同题答案标准化奖励及对应步骤的后续标准化奖励构成；`scalar` 对照把相同步骤项广播到整个回答，`outcome` 不使用步骤项。

old-policy 保存实际采样分布的 token log-prob，不用重算值替代。采样和重算已统一为固定形状的无缓存因果前向，但 BF16 不保证数值完全相同；在策略尚未更新时，逐 token 的重算概率除以采样概率必须严格落在配置的 `[1−clip, 1+clip]` 内，否则停止。CPU 对齐检查仍使用 `1e-4` 的更严容差；采样后和反向前均保留检查，日志保留最大差异。

为降低真实 7B 的 BF16 数值偏差，CUDA 路径关闭 BF16 降精度累加，并以冻结的 FP32 输出投影计算 logits；Transformer 主体仍为 BF16、可训练参数仍为 FP32。原 B3 六任务基线使用相同的数值设置；训练恢复及模型比较记录并核验输出头精度和累加设置。该设置不会替代完整的采样概率、显存及效果验证。

答案项仍在整个候选组内标准化；步骤项只在同组中通过结构解析、具备同模板步骤的候选之间标准化。解析失败的候选仅使用答案项的 −1 格式惩罚，不把虚拟零步骤放入其他候选的过程基线；有效候选少于两条或步骤方差小于阈值时，对应过程优势为零。此修订的奖励版本为 `geometry-process-valid-group-v2`。续训还要求运行时版本、设备、精度和 GPU 型号与保存检查点时一致。

## 几何评测与六任务

完成训练后，对每个检查点分别运行 `infer`、`evaluate`；`--condition` 支持 `identity`、`zero`、`repeat_first`、`shuffle_features` 和 `cross_scene`，缺少合适供体的项目保留在分母。`infer --resume` 只恢复相同身份的同一输出目录，预测尾行先校验再修复。默认评测需要完整预测回执和哈希；`--allow-incomplete` 只用于明确标记的中间诊断。`compare` 对两份完整评测结果按同一数据、输入干预和生成设置作配对比较。

模型对比还要求相同 B3、数据和 CoT SFT 终点；两个 GRPO 对照的训练配方、实际更新数和训练环境也必须匹配，允许改变优势分配模式。报告保留双方检查点及训练身份；配方中的完整预算标记只表示计划，不单独证明检查点已完成训练。使用 `compare --intervention` 时，基准必须是 `identity`，候选是其他干预且必须来自同一检查点；报告同时给出保留所有失败项的完整分母和排除无可用供体或缺失预测后的配对分母，以及各任务的对应结果。截断和生成错误仍留在可用配对中作为失败，不能把不可用干预造成的分数下降直接解释为模型依赖输入。

```bash
CKPT=/path/to/completed/final-checkpoint.pt
"$PY" -m b4dl_core.reasoning infer --dataset "$RUN/data" --geometry-dataset "$DATA" --baseline "$BASE" --preparation "$PREP" --checkpoint "$CKPT" --split test --condition identity --output "$RUN/test-identity"
"$PY" -m b4dl_core.reasoning evaluate --dataset "$RUN/data" --geometry-dataset "$DATA" --split test --predictions "$RUN/test-identity/predictions.jsonl" --output "$RUN/test-identity-report.json"
"$PY" -m b4dl_core.reasoning compare --base /path/to/sft-report.json --candidate /path/to/grpo-report.json --output "$RUN/comparison.json"
# 对同一检查点先分别完成 identity 和 zero 的 infer/evaluate：
"$PY" -m b4dl_core.reasoning compare --base "$RUN/test-identity-report.json" --candidate "$RUN/test-zero-report.json" --intervention --output "$RUN/zero-comparison.json"
```

报告包含每类任务的准确率、步骤得分/通过率、整条链通过率、格式失败、有效数值误差、生成 token 和耗时、场景级 bootstrap，以及相反标签配对和选项轮换。缺失、截断与不可用预测不从分母中删除。几何任务改善只有结合完整对照和输入干预结果，才能判断是否使用了 LiDAR 证据。

自由六任务协议使用历史协议所绑定的同一测试集和指标资源，从问题文本及 ego 元数据重新构建输入，绝不把答案传入输入构建。它与历史答案辅助协议分开保存和报告：

```bash
HISTORICAL=/path/to/six-task-manifest-final.json
"$PY" -m b4dl_core.reasoning prepare-six-task --project-root "$PWD" --geometry-dataset "$DATA" --historical-protocol "$HISTORICAL" --output "$RUN/six-free-protocol"
"$PY" -m b4dl_core.reasoning six-task-infer --dataset "$RUN/data" --geometry-dataset "$DATA" --baseline "$BASE" --preparation "$PREP" --original-b3 --protocol "$RUN/six-free-protocol/manifest.json" --output "$RUN/six-b3"
"$PY" -m b4dl_core.reasoning six-task-infer --dataset "$RUN/data" --geometry-dataset "$DATA" --baseline "$BASE" --preparation "$PREP" --checkpoint "$CKPT" --protocol "$RUN/six-free-protocol/manifest.json" --output "$RUN/six-candidate"
"$PY" -m b4dl_core.reasoning six-task-metrics --dataset "$RUN/data" --geometry-dataset "$DATA" --protocol "$RUN/six-free-protocol/manifest.json" --predictions "$RUN/six-b3/predictions.jsonl" --output "$RUN/six-b3-metrics"
"$PY" -m b4dl_core.reasoning six-task-metrics --dataset "$RUN/data" --geometry-dataset "$DATA" --protocol "$RUN/six-free-protocol/manifest.json" --predictions "$RUN/six-candidate/predictions.jsonl" --output "$RUN/six-candidate-metrics"
"$PY" -m b4dl_core.reasoning six-task-compare --base "$RUN/six-b3-metrics" --candidate "$RUN/six-candidate-metrics" --output "$RUN/six-comparison.json"
```

指标命令需要 `requirements-eval.txt` 和历史协议锁定的 NLTK、Java/METEOR、RoBERTa 等资产；缺少后端或某项指标失败时拒绝生成完成回执，不以降级回退数字冒充完整结果。内部 dev 可能被历史基底见过，应在实验报告中披露。几何 SFT、规则奖励和 CPU 测试均不能单独证明六任务整体提升。

六任务回执记录基底、准备配置、源码、生成设置、运行环境、精度和设备；对比只接受这些条件匹配的结果。两个推理模型之间还必须共享同一 SFT 起点；原始 B3 可作为独立基线。旧版缺少这些身份字段的六任务回执需要使用当前代码重新生成，不能视为已核验的配对结果。

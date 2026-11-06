# B4DL 训练与评测

当前 `upcore` 分支围绕 B4DL 主模型更新开展。代码审查、执行顺序及验收边界见 [upcore 开发方向](../docs/upcore/README.md)。

- 本体接入与检查：`python -m b4dl_core.b3`，说明见 [接入文档](docs/core_b3_integration.md)。
- 原六任务回归：`python -m b4dl_core.six_task`；几何诊断入口：`python -m b4dl_core`。
- 历史训练配置与已知中断：见 [训练记录](docs/core_geometry_training.md)。新实现不能直接绕过身份校验恢复旧实验。
- 分层依赖、版本/API 预检、几何推理 `--resume-inference` 与六任务尾行恢复：见 [环境与恢复说明](docs/upcore_runtime.md)。

运行以上命令时使当前 `mllm/` 位于 `PYTHONPATH`，并使用已验证的 `wqlc` 环境。入口存在不代表当前冻结配置、真实数据和权重已满足运行条件。

## 原项目基础训练入口

以下命令从 `mllm/` 目录执行。原论文是静态 3D 对齐、4D 时空理解两个训练阶段；数据文件名 `stage2` / `stage3` 的简单与复杂任务划分不等于额外的训练阶段。原项目入口与 upcore 从已核验 B3 继续的实验入口互相独立。

Before running, please download [this file](https://huggingface.co/lmsys/vicuna-7b-v1.5/tree/main) and place it under ./base_model/

```shell
bash run_stages.sh \
     --s1_data ./b4dl_dataset/stage1_lidarllm_mm.json \
     --s1_feat ./b4dl/stage1_features \
     --s2_data ./b4dl_dataset/stage2.json \
     --s2_feat ./b4dl/stage2_features \
     --model_name_or_path ./base_model/vicuna-v1-5-7b
```

`run_stages.sh` 将未识别的参数原样传给两个阶段；`stage2.sh` 默认读取固定的 `./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin`。若通过通用 `--output_dir` 更改第一阶段输出，第二阶段不会自动跟随，而且两个阶段会收到同一个输出目录。自定义输出应分别运行阶段脚本并显式指定第二阶段的 `--pretrain_mm_mlp_adapter`，核验实际参数和资产后再启动。脚本找不到 `nvcc` 时还会尝试执行 `conda install`。运行前应核对环境、数据和路径；这里的脚本默认值不是已完成的论文复现凭证。

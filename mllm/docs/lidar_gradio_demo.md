# B4DL 4D LiDAR 可视化 Demo

论文案例页优先载入相机图齐全的已关联时序样本。相机文件只覆盖部分场景时，
整行缺失会自动省略；前后相机均缺失则导出明确标注 `LiDAR-only evidence · RGB unavailable`
的点云案例。部分帧缺图时保留大字占位提示，并且不在占位图上投影真值框。
JSON 会记录每帧的相机可用状态、缺失面板和实际显示行。

该 Demo 直接读取本地 nuScenes 场景，提供当前帧点云的 3D/BEV 视图、六路相机切换、真值框和截至当前帧的历史轨迹。它不读取或修改训练数据。

## 安装

在项目的推理环境中安装独立 Demo 依赖：

```bash
cd mllm
pip install -r requirements-demo.txt
```

该文件在原有推理依赖之上追加 Plotly 和 nuScenes SDK，但不会修改原始依赖清单或训练代码。纯查看模式不加载模型权重，也不要求 CUDA。

## 纯查看模式

```bash
cd mllm
python -m vtimellm.demo_gradio \
  --nuscenes_root /path/to/nuScenes \
  --nuscenes_version v1.0-trainval \
  --scene_metadata /path/to/scene_metadata.json
```

`--scene_metadata` 可省略。省略后仍可浏览 nuScenes 原生场景，但由于无法解析 B4DL 的随机 `scene_id`，模型问答不会启用。数据根目录也可以通过环境变量 `B4DL_NUSCENES_ROOT` 设置。

## 查看并启用模型问答

以下四项必须成套提供，`--stage3` 可选：

```bash
python -m vtimellm.demo_gradio \
  --nuscenes_root /path/to/nuScenes \
  --scene_metadata /path/to/scene_metadata.json \
  --model_base /path/to/vicuna-7b-v1.5 \
  --pretrain_mm_mlp_adapter /path/to/mm_projector.bin \
  --stage2 /path/to/stage2-adapter \
  --feat_folder /path/to/stage2_features
```

模型始终读取完整场景的 `[N, 768]` 特征，并检查 `N` 是否与 nuScenes 场景关键帧数一致。任何缺失或错位都会只禁用当前场景的问答，不会静默使用错误特征。

常用参数：

- `--max_points 30000`：每帧发送给浏览器的最大点数。
- `--server_name 0.0.0.0`：允许局域网访问。
- `--server_port 7860`：修改监听端口。
- `--share`：启用 Gradio 分享链接。

## 测试

核心测试不需要 Gradio、nuScenes 数据集或 GPU：

```bash
cd mllm
python -m unittest tests.test_lidar_visualizer -v
```

完整验收建议先使用 `v1.0-mini`，确认点云、相机、框和轨迹在同一帧对齐，再切换到 `v1.0-trainval` 和真实 B4DL 权重。

## 模型效果看板

效果看板读取已经完成的评测结果，不重新运行模型，也不要求 CUDA。新版
`evaluation/test_b4dl.py` 会在 `predictions.json` 中保存每条预测对应的
`scene_id`；旧版预测可通过原始测试集安全补全场景关联。

```bash
cd mllm
python -m vtimellm.demo_gradio \
  --nuscenes_root /path/to/nuScenes \
  --scene_metadata ../encoders/lidarclip/annotations/scene_metadata.json \
  --predictions ./eval_results/stage2_full_seqv3_mixed_b3/predictions.json \
  --metrics ./eval_results/stage2_full_seqv3_mixed_b3/metrics.json \
  --test_data ./b4dl_dataset/test_qa.json
```

页面增加三个区域：

- **效果总览**：最终指标与论文 Table 3 对比、per-task 指标、分类混淆矩阵、
  TG IoU 分布和开始帧散点图。
- **样本诊断**：按任务、状态、关键词和得分分页筛选；选择样本后显示点云、
  BEV、相机、问题、Ground Truth、预测和任务对应的单样本得分。
- **论文案例图**：从当前评测页载入一个样本，选择 2–8 个场景帧（默认
  5 帧），生成同步的前视、后视和 LiDAR BEV 三行视图。默认比较
  `Ground Truth` 与实际评测中的 `B4DL prediction`，支持前视黄色、后视
  绿色的短语高亮，并导出 300 DPI PNG、PDF 和记录来源的 JSON 文件。

案例图以 [B4DL 论文 Figure 5](https://arxiv.org/html/2508.05269v1#S5.F5)
为版式参考：白底、从左到右的时间顺序、前后相机与 LiDAR 同列、下方问题与
答案卡片。相机画面保持完整比例，BEV 使用当前帧车辆坐标并以前方朝上显示。
图中的 RGB 相机只提供理解场景的辅助证据；B4DL 的模型输入仍为 LiDAR 特征。

黄色和绿色表示前视、后视语义，颜色不代表答案正确与否。原论文比较的是
B4DL 与其他模型；本 Demo 的默认左侧为数据集 GT，不能将其标成 VTimeLLM
或 B4DL-LiDARLLM 的输出。只有提供对应模型的真实预测时，才可使用其名称。

载入评测样本时，导出器按以下顺序选择证据帧：

- 保存了 `feat_indices`：先校验全部帧号，再从实际保存的帧中选择最多
  5 个代表帧；不会从两个有效帧之间补出未保存的模型输入帧。
- 只保存了 `feat_range`：在包含两端的区间内选帧，并标注
  `Recorded range; exact input frames not saved`。
- 两者均未保存：均匀选择全场景辅助证据，标注
  `Scene evidence; input frames not saved`；这不代表确认模型读过完整场景。

无法确认场景关联、场景标识冲突、任一输入帧号无效，或有效帧少于两个时，
会明确拒绝自动配图。导出器不根据 GT 答案中的时间段猜测模型输入。图中
显示数据集 0 基帧号和根据 nuScenes 时间戳计算的相对时间；这些帧号不是
抽帧后从零重新编号的位置。

页面中的 nuScenes 真值框和历史轨迹默认关闭。启用后它们仅为人工标注辅助层，不是
B4DL 预测的检测框。导出的来源记录与编辑信息用于区分原始预测和手动修改
文案，展示真实模型案例时应保留这份记录。

时间定位时间轴使用数据集原始 0 基帧号。界面的 `DATASET FRAME 006 ·
POSITION 7/40` 表示数据集帧号为 6，同时它是场景中的第 7 帧。

### 输入组合

| 参数 | 效果 |
|---|---|
| 仅 `--metrics` | 只显示汇总指标；没有样本列表 |
| `--predictions` | 显示文本和单样本诊断；新版结果可直接关联场景 |
| 旧 `--predictions` + `--test_data` | 校验问题和 GT 后补全 `scene_id` |
| 再提供完整模型参数 | 在效果看板之外同时启用自由问答 |

若旧结果无法与 `test_qa.json` 唯一匹配，页面保留文本诊断并明确显示未关联，
不会猜测或跳转到错误场景。

## 单独导出论文案例图

不启动 Gradio 时，可直接调用独立渲染器。`--frames` 留空会在整段场景中
均匀选择 5 帧；高亮短语支持用逗号、分号或换行分隔。下例的答案仅为命令
格式示例，导出真实结果时应替换为对应评测样本的 GT 和预测。

```bash
cd mllm
python vtimellm/paper_case_visualizer.py \
  --nuscenes-root /path/to/nuScenes \
  --scene-metadata ../encoders/lidarclip/annotations/scene_metadata.json \
  --scene-token <nuscenes-scene-token> \
  --frames "0, 10, 20, 30, 39" \
  --question "What dynamic movement is observed throughout the frames?" \
  --baseline-label "Ground Truth" \
  --baseline-answer "Front vehicles move forward while rear vehicles move away." \
  --b4dl-answer "Front vehicles move forward while rear vehicles move away." \
  --front-highlights "Front vehicles" \
  --back-highlights "rear vehicles" \
  --source-note "Manual example; replace text with a saved evaluation sample." \
  --answer-source "user supplied" \
  --no-boxes --no-tracks \
  --output-dir ./paper_cases
```

常用导出选项：

| 参数 | 含义 |
|---|---|
| `--front-highlights` / `--back-highlights` | 两张答案卡共同使用的前方黄色、后方绿色短语 |
| `--front-instances` / `--back-instances` | 以逗号分隔的 nuScenes `instance_token`；同一实例在相机和 BEV 中保持对应颜色 |
| `--no-boxes` / `--no-tracks` | 关闭全部 GT 框与历史轨迹；命令行保留旧版默认，建议显式加上这两项 |
| `--bev-range 52` | BEV 的前后、左右范围各为 ±52 米，所有所选帧使用同一量程 |
| `--source-note` / `--answer-source` | 写入图注和 JSON 的场景证据、文本来源说明 |
| `--provenance-file` | 附加来源 JSON 文件，例如评测路径、样本 ID、模型输入帧记录；内容原样写入输出 JSON |
| `--frame-labels` / `--frame-timestamps` | 可选的已排序选帧标签及绝对微秒时间戳，顺序与最终排序后的帧一致；不提供时读取 nuScenes 时间戳 |

`--baseline-answer` 和 `--b4dl-answer` 保留已有调用接口；标签默认为
`Ground Truth` 和 `B4DL prediction`。旧的 `--baseline-highlights` 和
`--b4dl-highlights` 仍可兼容，但新案例应优先使用按前后视角命名的高亮选项。
选择的目标实例始终是 nuScenes GT，关闭全部框后仍可单独显示所选目标。

相机行保留原图完整画面，可使用 nuScenes 标定叠加真值 3D 框；LiDAR 行为
适合打印的静态 BEV，可叠加真值框和截至当前帧的历史轨迹。该流程只读数据
与评测输出，不加载或修改训练代码。

输出文件应成套保留：PNG 供预览与插图，PDF 供排版，JSON 保存场景、帧号、
时间及来源。相对时间以第一个有时间戳的所选帧为零点；标定缺失时图注明确
提示使用 LiDAR 传感器坐标，不能将其当作已对齐的车辆坐标。论文参考图用于
决定布局和颜色；它不证明当前模型具有原论文中
展示的性能，案例图也不能替代完整测试集评测。

论文案例图的 CPU 单元测试：

```bash
python -m unittest tests.test_paper_case_visualizer -v
```

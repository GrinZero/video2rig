# Colab 核心训练

训练入口仍是 `python -m video2rig.reader.train`。Blender、模型导出和图片渲染留在本地，Colab 只负责每个 creature 的 joint reader 训练；不需要视频、追踪器、UniRig 或 Colab 中的 Blender。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GrinZero/video2rig/blob/colab-training/colab/train_reader.ipynb)

该链接指向 `GrinZero/video2rig` 的 `colab-training` 分支。如果使用 fork 或其他分支，请同步修改两个徽章的 owner/repo/ref；notebook 内的 repo_url/repo_ref 也须对应。

## 本地准备一次

先完成原有的 rig 导出与 start 图片/相机导出（README 步骤 1、2），再运行：

```bash
python -m video2rig.reader.prepare my_troll --blend model.blend --skip-train
python train_colab.py pack my_troll --output work/training.zip
python train_colab.py source --output work/video2rig-colab.zip
```

第一条命令只生成姿态、渲染图和可见性标签，不开始训练。`prepare --skip-train` 仍需要原项目的 Python 依赖、PyTorch 和 Blender；`pack` 本身仅使用 Python 标准库。已有完整训练数据时直接打包即可。`pack` 包含 creature.json、配置指定的 rig/start 目录和 reader 数据，排除权重、日志与 ZIP；不包含 clips/work/out。为保证换机器后路径有效，rig/start 路径必须指向项目内部。

## 在 Colab 中运行

尚未发布 GitHub 时：在 Colab 的“上传笔记本”中打开 `colab/train_reader.ipynb`。运行时选择 GPU，保留默认 `source_mode="upload"`，运行全部单元格，依次上传上一步生成的 `video2rig-colab.zip` 源码包和上一步的 `training.zip`。

点击 Open in Colab，保留默认 `source_mode="github"` 和 `repo_ref="colab-training"`，运行全部。使用 Google Drive 时打开 `use_drive`，可把 training_zip 设置为 Drive 中的数据路径，最终结果 ZIP 也会保存到 Drive。默认 steps=9000、batch=4；steps=0 时按 minutes 训练。首次需要联网下载 `facebook/dinov2-small`。

Colab 自带的 CUDA PyTorch 保持不变，仅安装 requirements-colab.txt 中的训练依赖。A100 等支持原生 BF16 的 GPU 使用 BF16；T4 自动使用 FP32，速度较慢。显存不足可把 batch 降到 2 或 1，评估也会使用同样大小的批量。训练图片会加载到主机内存；数据太大时应在本地减少 --poses 后重新准备。

## Python 一键训练与回填

同一入口也支持本地或其他 CUDA 机器，无需 notebook：

```bash
python train_colab.py train my_troll --install --steps 9000 --batch 4 --result work/reader-result.zip
```

`--install` 可省略；它安装训练依赖，但不安装 PyTorch，本地需自行先安装 CUDA PyTorch。可选 `--init /path/reader.pt` 用作权重热启动，不恢复优化器/调度器。

训练成功后 ZIP 包含 `reader/reader.pt` 和 `reader/reader-train.json`。将其解压到原项目根目录（已有权重时先自行备份），继续：

```bash
python -m video2rig run my_troll stagger
```

训练与渲染必须来自同一模型、骨架和 start 相机。Colab 运行时是临时的；当前训练器在训练结束时写权重，没有定期中间 checkpoint。Drive 保存最终产物，不提供断线续训。这里只支持标准 joint reader 流程；实验性的 depth/skin/context/crops 模式可继续使用原 train 模块。

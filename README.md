# Slam3DR

为了满足赛题对于“输入单目RGB视频，输出3D重建可视化结果”以及“输入单目RGBD视频，输出3D可视化结果”的需求，我们基于 [Slam3R](https://github.com/PKU-VCL-3DV/SLAM3R) 框架，结合深度特征可选融合机制设计实现了 Slam3DR。

---

## 1. 技术方案概述（Slam3R + Depth）

Slam3DR 的整体思路是将“多视图稠密 3D 点图预测”与“跨帧对齐/融合”拆成两个可组合的模块：

### 1.1 I2P：Image-to-Points（局部几何恢复）

- **入口类**：`slam3dr.models.Image2PointsModel`
- **输入**：一个局部窗口内的多视图（默认 224×224），指定其中一个视图为参考帧（`ref_id`）。
- **输出**：
	- 参考帧：`pts3d`（在参考帧坐标系下的点图）+ `conf`（置信度图）
	- 其他帧：`pts3d_in_other_view`（已对齐到参考帧坐标系下）+ `conf`
- **额外能力**：内置相关性打分（`get_corr_score`），用于从候选参考帧池中检索最适合的“场景帧/参考帧”（重建 pipeline 会用到）。

### 1.2 L2W：Local-to-World（全局对齐与融合）

- **入口类**：`slam3dr.models.Local2WorldModel`
- **输入**：
	- 多个参考帧（scene frames）：提供 `pts3d_world`
	- 若干待注册帧（keyframes）：提供 `pts3d_cam`
- **输出**：将 keyframe 点图变换到参考坐标系（世界/场景坐标）并输出置信度；同时可对参考帧点图做一定的联合建模/“refine”。

### 1.3 离线/在线重建 Pipeline

- **离线**：`slam3dr.pipeline.recon_offline_pipeline.scene_recon_pipeline_offline`
	- 预提取 encoder token（`get_img_tokens`）以提速。
	- 自动/固定关键帧间隔（`keyframe_stride`，支持 `-1` 自适应）。
	- 初始窗口初始化（`initialize_scene`）选取最佳参考帧。
	- 后续帧的 I2P 局部恢复 + L2W 全局注册；维护 buffering set（`reservoir` / `fifo`）作为参考帧池。
- **在线**：`slam3dr.pipeline.recon_online_pipeline.scene_recon_pipeline_online`
	- 支持从图片目录、视频文件、在线视频 URL 逐帧读取（`FrameReader`）。

### 1.4 Depth 的使用方式

仓库包含两种与深度相关的机制：

1) **RGB+Depth 作为输入拼接（可选，训练用）**：`train.py` 提供 `--depth_fuse`、`--depth_fuse_channels`、`--depth_fuse_key`，会把模型输入通道从 3 改为 `3 + depth_fuse_channels`。

2) **Depth 校正（推理/重建用，可选）**：`recon.py` 提供 `--depth_correct` 相关参数；当数据 view 里包含 `depthmap` 时，会对点图做 scale/shift 校正后再用于初始化。

---

## 2. 代码结构速览

- `train.py`：训练入口（支持分布式），模型/损失/数据集均通过字符串 `eval()` 构建。
- `recon.py`：推理入口（离线/在线重建）。
- `visualize.py`：可视化重建过程与中间预测（依赖 `--save_preds` 导出的 `preds/` 目录）。
- `app.py`：Gradio demo 入口（离线/在线）。

核心包：

- `slam3dr/models.py`：`Image2PointsModel`、`Local2WorldModel`。
- `slam3dr/inference.py`：训练时每个 batch 的前向/损失封装（`loss_of_one_batch`）。
- `slam3dr/losses.py`：3D 回归与置信度加权等损失。
- `slam3dr/pipeline/`：场景重建离线/在线 pipeline。
- `slam3dr/datasets/`：Co3D、ScanNet、Replica 等序列数据集与采样器。

---
## 3. 环境搭建

```bash
conda create -n slam3dr python=3.12 cmake=3.14.0
conda activate slam3dr 
# install torch according to your cuda version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
# optional: install additional packages to support visualization and data preprocessing
pip install -r requirements_optional.txt
```

3. 可选：使用 XFormers 和为 RoPE 编写的自定义 CUDA 内核来加速 SLAM3DR
```bash
# install XFormers according to your pytorch version, see https://github.com/facebookresearch/xformers
pip install xformers
# compile cuda kernels for RoPE
# if the compilation fails, try the propoesd solution: https://github.com/CUT3R/CUT3R/issues/7.
cd slam3dr/pos_embed/curope/
python setup.py build_ext --inplace
cd ../../../
```
要在 Replica 数据集上运行我们的演示，请下载示例场景 [here](https://drive.google.com/file/d/1NmBtJ2A30qEzdwM0kluXJOp2d1Y4cRcO/view?usp=drive_link) 并解压到 `./data/Replica_demo/`。然后运行下面的命令，从视频图像重建该场景： 

```bash
python visualize.py \
--vis_dir results/wild_demo \
--save_stride 1 \
--enhance_z \
--conf_thres_l2w 12 \
```

结果将默认保存在 ./results/ 目录下。

---
## 4. 数据准备

本次训练主要采用 Co3D 与 ScanNetV2 数据集。

### 4.1 训练数据集字符串语法

`train.py` 中的 `--train_dataset` / `--test_dataset` 是 **Python 表达式字符串**，会被 `eval()` 执行。

常见组合方式（见 `slam3dr/datasets/base/easy_dataset.py`）：

- `N @ Dataset(...)`：将数据集“重采样/重定长”为 N。
- `A + B`：拼接多个数据集。

默认配置示例（来自 `train.py`）：

- `4000 @ Co3d_Seq(...) + 1000 @ ScanNet_Seq(...)`

### 4.2 ScanNet 下载脚本

 `scannet_wrangling_scripts/` 文件夹下有处理 ScannetV2的代码，从`.sens`文件展开成对应的文件。文件放在`../scannet`目录下，训练需要配合`../scannet/data_splits`目录下`scannetv2_{split}.txt`

---

## 5. 训练（train.py）

`train.py` 支持 I2P 与 L2W 两种训练模式，主要区别在于：

- `--loss_func i2p`：训练 `Image2PointsModel`（默认）。
- `--loss_func l2w`：训练 `Local2WorldModel`（需要设置 `--ref_ids`）。

训练输出默认写入 `--output_dir`（默认 `./output/`），并会自动恢复 `checkpoint-last.pth`（如果存在）。

### 5.1 单卡训练（I2P）

```bash
python train.py \
	--output_dir ./output/i2p_run \
	--pretrained i2p \
	--loss_func i2p \
	--epochs 8 \
	--batch_size 12
```

### 5.2 单卡训练（L2W）

> L2W 的 `--model` 默认是 I2P，因此训练 L2W 时请显式传入 `Local2WorldModel(...)`。

```bash
python train.py \
	--output_dir ./output/l2w_run \
	--pretrained l2w \
	--loss_func l2w \
	--ref_ids 0 1 2 3 4 \
	--model "Local2WorldModel(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), 
    enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, \
    mv_dec1='MultiviewDecoderBlock_max',mv_dec2='MultiviewDecoderBlock_max', enc_minibatch = 12, need_encoder=True)",    required=True
    type=str, help="string containing the model to build" \
	--epochs 8 \
	--batch_size 12
```
### 5.3 分布式训练

参考 `./utils/train_i2p.sh` 以及 `./utils/train_l2w.sh` 脚本。

```bash
# train the Image-to-Points model and the retrieval module
bash ./utils/train_i2p.sh
# train the Local-to-Wrold model
bash ./utils/train_l2w.sh
```

### 5.4 可选：RGB+Depth 输入拼接

当你的 dataset view 中包含深度（例如 key 为 `img_depth`），可用：

```bash
python train.py --depth_fuse --depth_fuse_channels 1 --depth_fuse_key img_depth ...
```

---

## 6. 推理/场景重建（recon.py）

`recon.py` 是重建的主入口：

- 默认 **离线**（从图片目录读取一段序列并重建）。
- 加 `--online` 则启用在线 pipeline（可从图片目录 / mp4 / URL 逐帧读取）。

如果不指定 `--i2p_weights` / `--l2w_weights`，会自动从 Hugging Face 加载预训练。

### 6.1 离线重建（图片序列目录）

```bash
python recon.py \
	--img_dir /path/to/your/images_dir \
	--save_dir results \
	--test_name MyScene \
	--keyframe_stride -1 \
	--save_preds
```

输出：

- `results/MyScene/<scene_id>_recon.ply`：最终点云（按置信度过滤 + 重采样）。
- `results/MyScene/preds/`：如果开启 `--save_preds`，会保存 `local_pcds.npy`、`registered_pcds.npy`、`*_confs.npy`、`input_imgs.npy` 与 `metadata.json`。

### 6.2 离线重建（指定权重）

```bash
python recon.py \
	--img_dir /path/to/your/images_dir \
	--i2p_weights /path/to/checkpoint-i2p.pth \
	--l2w_weights /path/to/checkpoint-l2w.pth \
	--save_dir results \
	--test_name MyScene
```

### 6.3 在线重建

```bash
python recon.py \
	--dataset /path/to/your/images_dir_or_video.mp4 \
	--online \
	--save_dir results \
	--test_name OnlineRun
```

---

## 7. 可视化（visualize.py）

`visualize.py` 依赖 `recon.py --save_preds` 生成的 `preds/` 目录。

```bash
python visualize.py \
	--vis_dir results/MyScene \
	--save_stride 2
```

可选：

- `--vis_cam`：估计相机位姿并可视化（会额外做 intrinsics/pose 估计）。
- `--enhance_z`：增强 z 轴显示效果。

---

## 8. Demo App（Gradio）

启动离线 demo：

```bash
python app.py
```

启动在线 demo：

```bash
python app.py --online
```

---

## 9. 常见问题（FAQ）

### 9.1 运行很慢 / 显存吃紧

- RoPE2D 若回退到 PyTorch 版本会明显变慢（启动时会有 warning）。
- 重建 pipeline 会维护 buffering set；在线模式下长期运行建议关注 `buffer_size`、`num_scene_frame` 等参数。

### 9.2 推理输出为空或点很少

- 可能是 `conf_thres_i2p` / `conf_thres_l2w` 过高导致过滤过多点；可在 `recon.py` 中降低阈值。

---

## Acknowledgments

我们的实现基于以下几个优秀的开源仓库：

- [Croco](https://github.com/naver/croco)
- [DUSt3R](https://github.com/naver/dust3r)
- [NICER-SLAM](https://github.com/cvg/nicer-slam)
- [Spann3R](https://github.com/HengyiWang/spann3r)
- [Slam3R](https://github.com/PKU-VCL-3DV/SLAM3R)


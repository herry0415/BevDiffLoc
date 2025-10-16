# BevDiffLoc
Run the contrastive experiment for Hercules using this model

# Environment setup
参考diffloc的环境安装 https://github.com/herry0415/DiffLoc
- 按照给定的环境安装一下torch和其他的库。然后缺什么包 就更新什么包
- 安装torchsparse 的时候要去官方下载一下zip压缩包 然后放进去 运行`pip install -e .` 来安装
- 安装torch_scatter的时候
    - 需要去一个官网https://data.pyg.org/whl/下载一个包  手动安装一下
    - 找到一个对应版本包进行下载 `torch_scatter-2.0.9-cp38-linux_x86`
- pytorch3d 也是去官网下载git 文件。然后pip install -e .
- spvnas要放在diffloc主文件夹下
- PYTHONPATH  `export PYTHONPATH=/home/ldq/code/DiffLoc-main/spvnas:/home/ldq/code/DiffLoc-main`
- MinkowskiEngine
```
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
pip install -e . 
```

# Data prepare
We use merge_nclt.py and merge_oxford.py to generate local scenes for data augmentation.

# config 文件
在`train/test` 之前要修改`hercules_bev.yaml/hercules_radar_bev.yaml`的参数
- hercules_bev.yaml   lidar 的train/test 文件
- hercules_radar_bev.yaml   radar 的train/test 文件

# Train

## 1. 更改配置文件 
选择`lidar/radar` 对应  `hercules_bev.yaml/hercules_radar_bev.yaml`
  - 数据集路径 `train.dataroot`
  - 序列名 `train.sequence`
  - 权重路径 `exp_name 和 ckpt `
  - 数据集类别 `train.dataset`  `'Hercules_radar'`加载radar数据集类  `'Hercules'` 加载lidar数据集类
  - radar  `use_merge: False`  lidar `use_merge: True`

## 2. 修改train_bev.py加载配置文件部分
`conf = OmegaConf.load('cfgs/hercules_bev.yaml')`

## 3. 终端进行分布式训练

```python
accelerate launch --num_processes 3 --mixed_precision fp16 train_bev.py
```
- num_processes 可以指定为多个 具体使用哪个GPU 在代码里面用`os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"` 去进行调整
- 如果出现多进程训练端口被占用的情况 可以用 `--main_process_port 29501` 或任何一个大于 1024 且小于 65535 的端口

## Test

1. 更改配置文件`权重epoch部分` 选择`lidar/radar` 对应  `hercules_bev.yaml/hercules_radar_bev.yaml`
2. 修改test_bev.py加载配置文件部分`conf = OmegaConf.load('cfgs/hercules_bev.yaml')`
3. 终端进行分布式训练

```python
python test_bev.py
```







# BevDiffLoc
Run the contrastive experiment for Hercules using this model

# Environment setup
参考diffloc的环境安装即可

# Data prepare

# config 文件
在`train/test` 之前要修改`hercules_bev.yaml/hercules_radar_bev.yaml`的参数
- hercules_bev.yaml   lidar 的train/test 文件
- hercules_radar_bev.yaml   radar 的train/test 文件

## train
```python
accelerate launch --num_processes 3 --mixed_precision fp16 train_bev.py
```
- num_processes 可以指定为多个 具体使用哪个GPU 在代码里面用`os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"` 去进行调整
- 如果出现多进程训练端口被占用的情况 可以用 `--main_process_port 29501` 或任何一个大于 1024 且小于 65535 的端口

## test
```python
python test_bev.py
```







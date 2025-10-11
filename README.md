# BevDiffLoc
Run the contrastive experiment for Hercules using this model

# Environment setup
参考diffloc的环境安装即可

# config 文件
用于修改train or test的参数
- hercules_bev.yaml
- hercules_radar_bev.yaml
## train

## test
python test_bev.py





bevdiffloc部分   在训练或者测试的时候需要改加载的yaml文件
然后具体参数或者数据集路径。在yaml文件里面改
radar就用hercules_radar_bev.yaml 
train和test是一致的

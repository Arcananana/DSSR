# DSSR
This is a pytorch implementation of [Learning Detail-Structure Alternative Optimization for Blind Super-Resolution](https://ieeexplore.ieee.org/abstract/document/9721549).
## Requirement
+ python3
+ NVIDIA GPU + CUDA
+ pytorch >= 1.7.1
+ python packages: pip3 install numpy opencv-python lmdb pyyaml
## Quick Start
Download the pretrained models at [百度网盘](https://pan.baidu.com/s/1J11LyvdSWsiYZfia1a6YVw?pwd=dssr) and put them into `checkpoints` folder.
```
python inference.py -input_dir=<input_dir> -output_dir=<output_dir>
```
## Test
## Train

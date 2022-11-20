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
python inference.py -input_dir=<your_input_dir> -output_dir=<your_output_dir>
```
## Test
There are two blind settings mentioned in our paper. For setting1, we synthesize the *Gaussian8* datasets using `scripts/generate_mod_blur_LR_bic.py` with five datasets: Set5, Set14, BSD100, Urban100, Manga109. Download the [benchmark datasets](https://github.com/XPixelGroup/BasicSR/blob/a19aac61b277f64be050cef7fe578a121d944a0e/docs/Datasets.md) and put them into `datasets` folder like this (only put the ground-truth images) 
```
datasets
|── Set5
|── Set14
|── ...
```
Then run the script `generate_mod_blur_LR_bic.py`, and you will get folder like this
```
datasets
|── Set5
|── Set5G8
|   |── LRblur
|   |── HR
|── Set14
|── ...
```
For setting2, we using the benchmark DIV2KRK from KernelGAN.
```
python test.py -opt=options/test_setting.yml
```
## Train
```
python train.py -opt=options/train_setting.yml
```

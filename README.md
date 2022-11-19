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
There are two blind settings mentioned in our paper. For setting1, we synthesize the *Gaussian8* datasets using `scripts/generate_mod_blur_LR_bic.py` with five datasets: Set5, Set14, BSD100, Urban100, Manga109. You should first put them into `datasets` folder like this 
```

```
For setting2, we using the benchmark DIV2KRK from KernelGAN.
## Train

# DSSR
This is a pytorch implementation of [Learning Detail-Structure Alternative Optimization for Blind Super-Resolution](https://ieeexplore.ieee.org/abstract/document/9721549).
This repo is built on the basis of [DAN](https://github.com/greatlog/DAN), thanks for their open-sourcing!
## Requirement
+ python3
+ NVIDIA GPU + CUDA
+ pytorch >= 1.7.1
+ python packages: pip3 install numpy opencv-python lmdb pyyaml
## Quick Start
Download the pretrained models at [百度网盘](https://pan.baidu.com/s/1J11LyvdSWsiYZfia1a6YVw?pwd=dssr) and put them into `checkpoints` folder. For different settings, you may still have to modify the `options/test_setting.yml`.
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
For setting2, we using the benchmark dataset [DIV2KRK]((http://www.wisdom.weizmann.ac.il/~vision/kernelgan/DIV2KRK_public.zip)) from [KernelGAN](https://github.com/sefibk/KernelGAN).

Modify the dataset path and test settings in `options/test_setting.yml` and run the following command
```
python test.py -opt=options/test_setting.yml
```
## Train
Download the [DIV2K] (https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) and merge it into one folder. Modify `options/train_setting.yml` and run the following command
```
python train.py -opt=options/train_setting.yml
```
## Citation
If you find this repo useful, please consider citing our work:
```
@ARTICLE{9721549,
  author={Li, Feng and Wu, Yixuan and Bai, Huihui and Lin, Weisi and Cong, Runmin and Zhang, Chunjie and Zhao, Yao},
  journal={IEEE Transactions on Multimedia}, 
  title={Learning Detail-Structure Alternative Optimization for Blind Super-Resolution}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMM.2022.3152090}}
```

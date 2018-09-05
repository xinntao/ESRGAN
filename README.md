# ESRGAN (Enhanced SRGAN) [[Paper]](https://arxiv.org/abs/1809.00219) [[BasicSR]](https://github.com/xinntao/BasicSR)
## Enhanced Super-Resolution Generative Adversarial Networks
By Xintao Wang, [Ke Yu](https://yuke93.github.io/), Shixiang Wu, [Jinjin Gu](http://www.jasongt.com/), Yihao Liu, [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=en), [Chen Change Loy](http://personal.ie.cuhk.edu.hk/~ccloy/), [Yu Qiao](http://mmlab.siat.ac.cn/yuqiao/), [Xiaoou Tang](https://scholar.google.com/citations?user=qpBtpGsAAAAJ&hl=en)

This repo only provides simple testing codes, pretrained models and the netwrok strategy demo. 

### :smiley: **For full training and testing codes, please refer to  [BasicSR](https://github.com/xinntao/BasicSR).**

We won the first place in [PIRM2018-SR competition](https://www.pirm2018.org/PIRM-SR.html) (region 3) and got the best perceptual index.
The paper is accepted to [ECCV2018 PIRM Workshop](https://pirm2018.org/).
### BibTeX

    @article{wang2018esrgan,
        author={Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Loy, Chen Change and Qiao, Yu and Tang, Xiaoou},
        title={ESRGAN: Enhanced super-resolution generative adversarial networks},
        journal={arXiv preprint arXiv:1809.00219},
        year={2018}
    }

<p align="center">
  <img height="400" src="figures/baboon.png">
</p>

The **RRDB_PSNR** PSNR_oriented model trained with DF2K dataset (a merged dataset with [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (proposed in [EDSR](https://github.com/LimBee/NTIRE2017))) is also able to achive high PSNR performance.

| Method | Training dataset | Set5 | Set14 | BSD100 | Urban100 | Manga109 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [SRCNN](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)| 291| 30.48/0.8628 |27.50/0.7513|26.90/0.7101|24.52/0.7221|27.58/0.8555|
| [EDSR](https://github.com/thstkdgus35/EDSR-PyTorch) | DIV2K | 32.46/0.8968 | 28.80/0.7876 | 27.71/0.7420 | 26.64/0.8033 | 31.02/0.9148 |
| [RCAN](https://github.com/yulunzhang/RCAN) |  DIV2K | 32.63/0.9002 | 28.87/0.7889 | 27.77/0.7436 | 26.82/ 0.8087| 31.22/ 0.9173|
|RRDB(ours)| DF2K| **32.73/0.9011** |**28.99/0.7917** |**27.85/0.7455** |**27.03/0.8153** |**31.66/0.9196**|


## Quick Test
#### Dependencies
- Python 3
- PyTorch >= 0.4.0
- Python package `cv2`, `numpy` 
#### Test
1. Clone this github repo. 
```
git clone https://github.com/xinntao/ESRGAN
cd ESRGAN
```
2. Place your own **low-resolution images** in `./LR` folder. (There are two sample images - baboon and comic). 
3. Download pretrained models from [Google Drive](https://drive.google.com/drive/u/0/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY) or [Baidu Drive](https://pan.baidu.com/s/1-Lh6ma-wXzfH8NqeBtPaFQ). Place the models in `./models`. We provide two models with high perceptual quality and high PSNR performance (see [model list](https://github.com/xinntao/ESRGAN/tree/master/models)).
4. Run test. We provide ESRGAN model and RRDB_PSNR model.
```
python test.py models/RRDB_ESRGAN_x4.pth
python test.py models/RRDB_PSNR_x4.pth
```
5. The results are in `./results` folder.


## Introduction 
We improve the [SRGAN](https://arxiv.org/abs/1609.04802) from three aspects:
1. adopt a deeper model using Residual-in-Residual Dense Block (RRDB) without batch normalization layers.
2. employ [Relativistic average GAN](https://ajolicoeur.wordpress.com/relativisticgan/) instead of vanilla GAN.
3. improve the perceptual loss by using the features before activation.

In contrast to SRGAN, which claimed that **deeper models are increasingly difficult to train**, our deeper ESRGAN model shows its superior performance with easy training.

<p align="center">
  <img height="100" src="figures/architecture.png">
</p>
<p align="center">
  <img height="130" src="figures/RRDB.png">
</p>

## Network Interpolation
We propose the **network interpolation strategy** to balance the visual quality and PSNR.

<p align="center">
  <img height="500" src="figures/net_interp.png">
</p>

We show the smooth animation with the interpolation parameters changing from 0 to 1. 
<p align="center">
  <img height="400" src="figures/43074.gif">
</p>
<p align="center">
  <img height="480" src="figures/81.gif">
  &nbsp &nbsp
  <img height="480" src="figures/102061.gif">
</p>
  
## Qualitative Results
PSNR (evaluated on the luminance channel in YCbCr color space) and the perceptual index used in the PIRM-SR challenge are also provided for reference.

<p align="center">
  <img src="figures/rlt_01.png">
</p>
<p align="center">
  <img src="figures/rlt_02.png">
</p>
<p align="center">
  <img src="figures/rlt_03.png">
</p>
<p align="center">
  <img src="figures/rlt_04.png">
</p>
<p align="center">
  <img src="figures/rlt_05.png">
</p>
<p align="center">
  <img src="figures/rlt_06.png">
</p>

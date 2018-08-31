# ESRGAN (Enhanced SRGAN) [[Paper]](https://github.com/xinntao/ESRGAN) [[BasicSR]](https://github.com/xinntao/BasicSR)
## Enhanced Super-Resolution Generative Adversarial Networks
By Xintao Wang, [Ke Yu](https://yuke93.github.io/), Shixiang Wu, [Jinjin Gu](http://www.jasongt.com/), Yihao Liu, [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=en), [Chen Change Loy](http://personal.ie.cuhk.edu.hk/~ccloy/), [Yu Qiao](http://mmlab.siat.ac.cn/yuqiao/), [Xiaoou Tang](https://scholar.google.com/citations?user=qpBtpGsAAAAJ&hl=en)

This repo only provides simple testing codes and pretrained models. 

### :smiley: **For full training and testing codes, please refer to  [BasicSR](https://github.com/xinntao/BasicSR).**

We won the first place in [PIRM2018-SR competition](https://www.pirm2018.org/PIRM-SR.html) (region 3) and got the best perceptual index.
<!--pirm ECCV'2018 Workshop-->
### BibTeX
<!--
    @article{wang2018esrgan,
        author={Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Loy, Chen Change and Qiao, Yu and Tang, Xiaoou},
        title={ESRGAN: Enhanced super-resolution generative adversarial networks},
        journal={arXiv preprint arXiv:},
        year={2018}
    }
-->

<p align="center">
  <img height="400" src="figures/baboon.png">
</p>
                                             
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

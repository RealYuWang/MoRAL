# MoRAL: 4D Radar and LiDAR Fusion for Robust 3D Object Detection

![motivation](https://s2.loli.net/2025/05/14/oS8UYNJdnyxDKWu.png)

## Abstract

In this paper, we propose **MoRAL**, a motion-aware multi-frame 4D radar and LiDAR fusion framework for robust 3D object detection. First, a **Motion-aware Radar Encoder (MRE)** is designed to compensate for inter-frame radar misalignment from moving objects. Later, a **Motion Attention Gated Fusion (MAGF)** module integrate radar motion features to guide LiDAR features to focus on dynamic foreground objects. Extensive evaluations on the [View-of-Delft (VoD)](https://github.com/tudelft-iv/view-of-delft-dataset) dataset demonstrate that MoRAL outperforms existing methods, achieving the highest mAP of **73.30%**  in the entire area and 88.68%  in the driving corridor. Notably, our method also achieves the best AP of 69.67% for pedestrians in the entire area and **96.25%** for cyclists in the driving corridor. 

## Architecture

![arch-1](https://s2.loli.net/2025/05/14/CNzcgHtKI6hD7wl.png)

![mre-1](https://s2.loli.net/2025/05/14/yahgYjicSIkmn5E.png)

## Getting Started

The implementation is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

### Installation

Please refer to this file ([Installation](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md)) to install the latest version of OpenPCDet.

### Dataset Preparation

Please refer to this file ([Dataset Preparation](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md)) to prepare VoD dataset. Our customized dual-modal dataset configuration file is provided [here](https://github.com/RealYuWang/MoRAL/blob/main/tools/cfgs/dataset_configs/vod_bimodal.yaml).

### Train and Eval

- Train with:

```
python train.py --cfg_file tools/cfgs/kitti_models/MoRAL.yaml
```

- Evaluate and visualize with(frame id can be any frame in test set):

```
python predict.py --cfg_file tools/cfgs/kitti_models/MoRAL.yaml --ckpt your_trained_model.pth --frame_id 00000
```

## BibTex

If you find this work helpful for your research, please consider citing the following entry:

```
@misc{peng2025moralmotionawaremultiframe4d,
      title={MoRAL: Motion-aware Multi-Frame 4D Radar and LiDAR Fusion for Robust 3D Object Detection}, 
      author={Xiangyuan Peng and Yu Wang and Miao Tang and Bierzynski Kay and Lorenzo Servadei and Robert Wille},
      year={2025},
      eprint={2505.09422},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.09422}, 
}
```

## Acknowledgement

Many thanks to these excellent works and repos:

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/tree/master)
- [View-of-Delft](https://github.com/tudelft-iv/view-of-delft-dataset)
- [RLNet](https://openreview.net/pdf?id=I5IIhtSbMe)

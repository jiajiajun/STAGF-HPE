### Hierarchical Spatial-temporal Adaptive Graph Fusion for Monocular 3D Human Pose Estimation

###News
* [2023/11/23] The codes of our method are released!

### TODO
*Provide pretrained models
*Provide training and testing code

### Dependencies
Make sure you have the following dependencies installed before proceeding:
- Python >=3.6
- PyTorch >= 1.0.1
- matplotlib
- numpy

### Data preparation
- Download the raw data from [Human3.6M](http://vision.imar.ro/human3.6m) and [HumanEva-I](http://humaneva.is.tue.mpg.de/)
- Preprocess the dadaset in the same way as like [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md)
- Then put the preprocessed dataset under the data directory

       -data\
            data_2d_h36m_gt.npz
            data_3d_36m.npz
            data_2d_h36m_cpn_ft_h36m_dbb.npz
            data_2d_h36m_sh_ft_h36m.npz
        
            data_2d_humaneva15_gt.npz
            data_3d_humaneva15.npz
            data_2d_humaneva15_detectron_pt_coco.npz

### Training & Testing
Code will be released soon.

### Download our pretrained models
Models will be released soon.

### Acknowledgements
Our code is extended from the following works:
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
- [Gast-Net](http://www.juanrojas.net/gast/)

Thanks to the original authors for their work!

### Reference
If you find our paper and repo useful, please cite our paper. Thanks!

```
@ARTICLE{Zhang2024,
  author={Zhang, Lijun and Lu, Feng and Zhou, Kangkang and Zhou, Xiang-Dong and Shi, Yu},
  journal={IEEE Signal Processing Letters}, 
  title={Hierarchical Spatial-Temporal Adaptive Graph Fusion for Monocular 3D Human Pose Estimation}, 
  year={2024},
  volume={31},
  number={},
  pages={61-65},
  doi={10.1109/LSP.2023.3339060}}
}
```

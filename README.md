# SGCN4HOI
SGCN4HOI: A Skeleton-aware Graph Convolutional Network for Human-Object Interaction Detection
[Paper](https://arxiv.org/pdf/2207.05733.pdf)
![Image text](https://github.com/zhumanli/SGCN4HOI/blob/main/framework.png)

# Dataset & Requirements
Please follow the Installation in [VSGNet](https://github.com/ASMIftekhar/VSGNet) to prepare the V-COCO dataset and install requirements.txt. Note that this is python2 virtual environment. After preparing the dataset by following VSGNet, replace Object_Detections_vcoco folder with [our processed data](https://drive.google.com/drive/folders/1HU4x470_VZRl2NSJsl2yBnUiwWSxsw9l?usp=sharing) which includes keypoints of humans and objects. Our pretrained model can be downloaded [here](https://drive.google.com/file/d/1qmCrDzw7C32TJQ5U47YBrXUuPXd5DPGg/view?usp=sharing).

# Training 
To train the model from scratch (inside "scripts/"):
```
CUDA_VISIBLE_DEVICES=0, 1 python2 main.py -fw new_test -ba 16 -l 0.005 -e 60 -sa 20 
```

# Test
```
CUDA_VISIBLE_DEVICES=0, 1 python2 main.py -fw soa_paper -ba 16 -r t -i t
```

# Citing
If you find this work useful, please consider our paper to cite:
```
@inproceedings{zhu22skeleton,
 author={Zhu, Manli and Ho, Edmond S. L. and Shum, Hubert P. H.},
 booktitle={Proceedings of the 2022 IEEE International Conference on Systems, Man, and Cybernetics},
 series={SMC '22},
 title={A Skeleton-aware Graph Convolutional Network for Human-Object Interaction Detection},
 year={2022},
}
```

# Other
Our SGCN4HOI is based on [VSGNet](https://github.com/ASMIftekhar/VSGNet).

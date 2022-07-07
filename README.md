# SGCN4HOI
SGCN4HOI: A Skeleton-aware Graph Convolutional Network for Human-Object Interaction Detection
![Image text](https://github.com/zhumanli/SGCN4HOI/blob/main/framework.png)

# Dataset & Requirements
Please follow the Installation in [VSGNet] (https://github.com/ASMIftekhar/VSGNet) to prepare the V-COCO dataset and install requirements.txt. Note that this is python2 virtual environment.

# Training 
To train the model from scratch (inside "scripts/"):
```
CUDA_VISIBLE_DEVICES=0, 1 python2 main.py -fw new_test -ba 16 -l 0.005 -e 60 -sa 20 
```

# Test
```
CUDA_VISIBLE_DEVICES=0, 1 python2 main.py -fw soa_paper -ba 16 -r t -i t
```

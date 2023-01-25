# State Abstraction in DeepSeer

## Preliminaries

The state abstraction technique is developed from DeepStellar. However, this version of DeepStellar is slightly different with the implementation in the original paper, 
which uses grid splitting for state abstraction while this repository uses unsupervised clustering.


## Get Started

### Environments

Basic environments: `Python >= 3.6`.

*We suggest use virtual environment to avoid messing up your own environments.*

~~~sh
$ python -m venv ./venv
$ source ./venv/bin/activate
~~~

Required packages:
```bash
$ pip install -r requirements.txt
```

Organize your dataset as follows (put in `file/`):
```
├── abstraction
├── data
├── file
│   ├── checkpoints
│   │   │── toxic_ckpt_best.pth
│   ├── data
│   │   │── train.csv
│   ├── profile
│   ├── java
│   │   │── spmf.jar
├── graphics
├── model
├── utils
```

### Datasets

We provide supports with three different datasets:

- Toxic (download dataset from [Kaggle](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data))
- Quora (download dataset from [Kaggle](https://www.kaggle.com/c/quora-insincere-questions-classification))
- AGNews (dataset will be automatically downloaded from torchtext)

### (optional) Step 1 Train an RNN model

If you want to skip this step, please checkout `../server/deepstellar_backend/File/checkpoint/` for the pre-trained models.

```bash
$ python train_rnn_toxic.py # Toxic dataset
# $ python train_rnn_quora.py # Quora dataset
# $ python train_rnn_agnews.py # AGNews dataset
```

### Step 2 Build state abstraction model on training set and extract traces from test set

```bash
$ python state_abstraction_toxic.py # Toxic dataset
$ python mining_state_words.py # Toxic dataset

# $ python state_abstraction_quora.py # Quora dataset
# $ python mining_state_words.py --export_path=./file/export/quora/ # Quora dataset

# $ python state_abstraction_agnews.py # AGNews dataset
# $ python mining_state_words.py --export_path=./file/export/agnews/ # AGNews dataset
```

### Step 3 Generate Pattern Summary

#### Influential Pattern Summary

```bash
$ python influential_pattern_mining.py # Toxic dataset
# $ python influential_pattern_mining.py --export_path=./file/export/quora/ # Quora dataset
# $ python influential_pattern_mining.py --export_path=./file/export/agnews/ # AGNews dataset
```

#### Buggy Pattern Summary

Please put `spmf.jar` in `file/java/` (download from [here](http://www.philippe-fournier-viger.com/spmf/index.php?link=download.php))

```bash
$ python buggy_pattern_mining.py # Toxic dataset
# $ python buggy_pattern_mining.py --export_path=./file/export/quora/ # Quora dataset
# $ python buggy_pattern_mining.py --export_path=./file/export/agnews/ # AGNews dataset
```

### Step 4 Run with web application

Now you can move your generated files into `../web/dashboard/src/data/`.

```shell
$ mv ./file/export/[toxic/quora/agnews]/*.json ../web/dashboard/src/data/
$ mv ./file/profile/[toxic/quora/agnews]/*.profile ../server/deepstellar_backend/File/profile
```

## Citation

---

If you found our paper/code useful in your research, please consider citing:

```
@inproceedings{wang2023deepseer,
 author = {Wang, Zhijie and Huang, Yuheng and Song, Da and Ma, Lei and Zhang, Tianyi},
 title = {DeepSeer: Interactive RNN Explanation and Debugging via State Abstraction},
 booktitle = {Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems},
 year = {2023},
} 

@inproceedings{du2019deepstellar,
author = {Du, Xiaoning and Xie, Xiaofei and Li, Yi and Ma, Lei and Liu, Yang and Zhao, Jianjun},
title = {DeepStellar: Model-Based Quantitative Analysis of Stateful Deep Learning Systems},
year = {2019},
booktitle = {Proceedings of the 2019 27th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
pages = {477–487},
numpages = {11}
}
```
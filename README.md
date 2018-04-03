## MemoryGAN

Code for our paper [Memorization Precedes Generation: Learning 	Unsupervised GANs with Memory Networks](https://openreview.net/pdf?id=rkO3uTkAZ)
 by Youngjin Kim, Minjung Kim, Gunhee Kim.
This repository includes codes for training and testing MemoryGAN with Fashion-MNIST, affine-transformed MNIST and CIFAR10 datasets. It also include model parameters of MemoryGAN that we trained with CIFAR10 dataset. If you use this in your research, we kindly ask that you cite the below ICLR 2018 paper.

```
@inproceedings{
	kim2018memorization,
	title={Memorization Precedes Generation: Learning 	Unsupervised {GAN}s with Memory Networks},
	author={Youngjin Kim and Minjung Kim and Gunhee Kim},
	booktitle={International Conference on Learning Representations},
	year={2018},
	url={https://openreview.net/forum?id=rkO3uTkAZ},
}
```

## Dependencies

* python 2
* tensorflow 1.4

Install python packages

```
pip install -r requirements.txt

```

## How to use

### Downloading datasets
* Once you run the trainig script, it will automatically download and places datasets into `dataset` folder.
* See `model/train.py`, `affmnist.py`, `fashion.py` and `cifar10.py` for more details. 

### Training 

* Run `run.py` with arguments. For examples, run one of following commands.

```
python run.py --dataset=fashion --lr_decay=False --use_augmentation=False
python run.py --dataset=affmnist --lr_decay=False --use_augmentation=False
python run.py --dataset=cifar10 --lr_decay=True --use_augmentation=True
```

* See more arguments and hyperparameter settings in `run.py` and `models/config.py`

### Running pretrained model (CIFAR10)

* Run `run.py` with `--load_cp_dir` and `--is_train` arguments

```
python run.py --dataset=cifar10 --is_train=False --load_cp_dir=checkpoint/pretrained_model
```

### License
MIT License. Please see the LICENSE file for details.

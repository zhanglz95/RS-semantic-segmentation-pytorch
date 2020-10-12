# RS-semantic-segmentation-pytorch

[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]

## Introduction
This repo aims at implementing multiple semantic segmentation models on Pytorch(***1.x***) for RS(***RemoteSensing***) image datasets.

## Preparation
The code was tested with Anaconda, python3.7. You will do the follow to be ready for running the code:
```
# RS-semantic-segmentation-pytorch dependencies, torch and torchvision are installed by pip.
pip install -r requirements.txt

# Then clone this Repo using SSH:
git clone git@github.com:zhanglz95/RS-semantic-segmentation-pytorch.git
# or using HTTPS:
git clone https://github.com/zhanglz95/RS-semantic-segmentation-pytorch.git
```
## Todo
- [X] U-Net - Convolutional Networks for Biomedical Image Segmentation (2015). [[Paper]](https://arxiv.org/abs/1505.04597)

## Usage
### Train
First, you should put your datasets into the "data" folder.Then you can set up your training parameters by create a json file according to the json file format in "./config" folder. If you want to add a model for training, you can simply add the model.py to "./models" folder and modify the correspond configs.json. Everything being ready, then you can simply run:
```
python train.py -c "./configs/xxx.json"
```
Or if you want to run with multiple configs for experiments, you can put multiple json files in the sub-folder of "./configs" and then run:
```
python train.py -c_dir "./configs/configs_folder/"
```

### Inference
***Have not implement.***

## Acknowledgement
[awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)

[Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)

[python-image]: https://img.shields.io/badge/Python-3.x-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.x-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
[lic-image]: https://img.shields.io/badge/Apache-2.0-blue.svg
[lic-url]: #



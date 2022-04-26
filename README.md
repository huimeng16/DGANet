# DGANet
## Introduction
We propose a novel dual global attention neural network (DGANet) to improve the accuracy of breast lesion detection in ultrasound images. Specifically, we design a bilateral spatial attention module and a global channel attention module to enhance features in spatial and channel dimensions respectively. The bilateral spatial attention module enhances features by capturing supporting information in neighboring region of breast lesions and reducing integration of noise signal. The global channel attention module enhances features of important channels by weighted calculation, where the weights are decided by the learned interdependencies among all channels. We achieve accurate breast lesion detection on our collected dataset and a public dataset (BUSI).
## Updates
2022/4: DGANet released. The trained model with ResNet101 achieves mAP of 84.0% on our collected dataset.
## Usage
1. Install pytorch
   The code is tested on python 3.6 and torch 1.4.0.
   The code is modified from [yolov3](https://github.com/ultralytics/yolov3).


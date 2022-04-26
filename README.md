# DGANet
## Introduction
We propose a novel dual global attention neural network (DGANet) to improve the accuracy of breast lesion detection in ultrasound images. Specifically, we design a bilateral spatial attention module and a global channel attention module to enhance features in spatial and channel dimensions respectively. The bilateral spatial attention module enhances features by capturing supporting information in neighboring region of breast lesions and reducing integration of noise signal. The global channel attention module enhances features of important channels by weighted calculation, where the weights are decided by the learned interdependencies among all channels. We achieve accurate breast lesion detection on our collected dataset and a public dataset (BUSI).
## Updates
2022/4: DGANet released. The trained model with ResNet101 achieves mAP of 84.0% on our collected dataset.
## Usage
1. Install pytorch
   - The code is tested on python 3.6 and torch 1.4.0.
   - The code is modified from [ultralytics/yolov3](https://github.com/ultralytics/yolov3).
2. Clone the reposity
   ```
   git clone https://github.com/huimeng16/DGANet.git
   ```
3. Dataset
   - Download the BUSI dataset and convert the dataset to VOC style.
   - Please put dataset in folder ./data
4. Train on BUSI data
   ```
   python train.py --epochs 500 --batch-size 16 --cfg cfg/DGANet-resnet101-conv345-ratio1.cfg --img-size 640 --data data/BUSI_fold1.data
   ```
5. Test using trained model
   ```
   python test.py --cfg cfg/DGANet-resnet101-conv345-ratio1.cfg --data data/BUSI_fold1.data --weights weights/best.pt --batch-size 16 --img-size 416
   ```
6. Visualization of detected samples
   ```
   python detect.py --cfg cfg/DGANet-resnet101-conv345-ratio1.cfg --names data/cancer.names --weights weights/best.pt --source data/samples --output output
   ```
## Citation



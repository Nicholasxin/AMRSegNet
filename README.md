# AMRSegNet:Adaptive Modality Recalibration Network for Lung Tumor Segmentation on Multi-modal MR Images
This is the Pytorch implementation of AMRSegNet for paper《Adaptive Modality Recalibration Network for Lung Tumor Segmentation on Multi-modal MR Images》.
  
## Installation
- Install [pytorch](https://pytorch.org/get-started/previous-versions/) with python 3.7, pytorch==1.4.0, torchvision==0.5.0, CUDA==10.1.
- Python package requirement: SimpleITk, pydicom, tensorboardX
- Clone this repository: 
'''
<git clone https://github.com/Nicholasxin/AMRSegNet>
'''

## Dataset
- Our T2W-DWI MR dataset is private. For code implementation, the dataset for training and testing consist of T2W slices, DWI slices, label slices, which are all paired. 
  
## Training
- In the folder of repository, open terminal and run '<python train.py>'. If 


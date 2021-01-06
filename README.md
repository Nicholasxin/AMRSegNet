# AMRSegNet:Adaptive Modality Recalibration Network for Lung Tumor Segmentation on Multi-modal MR Images
This is the Pytorch implementation of AMRSegNet for paper《Adaptive Modality Recalibration Network for Lung Tumor Segmentation on Multi-modal MR Images》.
  
## Installation
- Install [pytorch](https://pytorch.org/get-started/previous-versions/) with python 3.7, pytorch==1.4.0, torchvision==0.5.0, CUDA==10.1.
- Python package requirement: SimpleITk, pydicom, tensorboardX
- Clone this repository:  
```
git clone https://github.com/Nicholasxin/AMRSegNet  
cd AMRSegNet  
```

## Dataset
- Our T2W-DWI MR dataset is private. For code implementation, the dataset for training and testing consist of T2W slices, DWI slices, label slices, which are all paired. 
  
## Training
- In the folder of repository `AMRSegNet`, open terminal and run `python train.py`. 
- Note: adding `--ngpu` to alter to the number of GPUs, adding `--batchSz` to change the batch size, adding `--nEpochs` to set the number of training epochs.  
- For showing the training process on tensorboard, the folder `runs` will be created. The trained model will be saved in auto-created folder `work`. 
- To open the tensorboard, open terminal and run `tensorboard --logdir runs`.

## Testing  
- run `python train.py` with `--inference` following the path of inference T2W data, `--dwiinference` following the path of inference DWI data, `--target` following the path of label of T2W data, `--resume` following the path of the best saved training model. All the added commands are requisite. 


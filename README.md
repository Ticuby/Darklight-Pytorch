# Darklight CNN

DarkLight Networks，a high-efficiency 3D Convolution network for low light illumination action recognition. 

 Paper to be presented at CVPR2021

This repository is modeling on the foundation of LateTemporalModeling3DCNN architectures and taking advantage of self-attention to build the classification network. 

## Dependencies

The code runs on Python 3.8 but in fact it is not a big deal to try to construct a low version Python. You can create a conda environment with all the dependecies by running 

```python
conda env create -f requirements.yml -n <env_name>
```

Note: this project needs the CUDA 11.1

## Dataset Preparation

In order to improve the dataset's load speed an facilitate pretreatment. The video should be pre-cut into frames and save in \datasets folder and the list of training and validation samples should be created as txt files in \datasets\settings folder.

For example, using ARID as the data. the frames is saved in \datasets\ARID_frames. And then using the different name of classifications to name the folder. 

The format of the frames like as "img_%05d.jpg" 

## Training

### Training with tow view

training with attention

```
python darklight.py --split=1 --batch-size=8 --gamma=1.8 --both-flow = True
```

training without attention

```
python darklight.py --split=1 --batch-size=8 --gamma=1.8 --both-flow = True --no_attention
```

For multi-gpu training, comment the two lines below os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" os.environ["CUDA_VISIBLE_DEVICES"]="0"

To continue the training from the best model, add -c. To evaluate the single clip single crop performance of best model, add -e

## Testing

```
python spatial_demo_bert.py  --split=1
```

## Related Projects

[R2+1D-IG65](https://github.com/moabitcoin/ig65m-pytorch): IG-65M Pytorch

[self-attention-cv](https://github.com/The-AI-Summer/self-attention-cv):self-attention-cv Pytorch

[LateTemporalModeling3DCNN](https://github.com/artest08/LateTemporalModeling3DCNN):2_Plus_1D_BERT






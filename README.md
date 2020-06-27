# SFSSD
Thanks to https://github.com/amdegroot/ssd.pytorch code framework, we got the job done on his basis

## Datasets
To make things easy, we provide bash scripts to handle the dataset downloads and setup for you.  We also provide simple dataset loaders that inherit `torch.utils.data.Dataset`, making them fully compatible with the `torchvision.datasets` [API](http://pytorch.org/docs/torchvision/datasets.html).

### VOC Dataset
PASCAL VOC: Visual Object Classes

##### Download VOC2007 trainval & test
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```
## train
run train.py
The proposed SFSSD network was trained for 450,000 iterations in the Pytorch framework with training batch size of 16. 
We update the network gradient every 4 iterations to bring the training batch size to 64. 
In the middle, use the Adam optimizer to train 150,000 iterations at a learning rate of 0.001, train 200,000 iterations at 0.0001, and train 100,000 iterations at 0.00001. \
Our experimental environment is RTX 2700S GPU and AMD Ryzen R5-3600 CPU


##show
Show our model


##weights
SFSSD512 can reach mAP 68.0 on PASCAL VOC2007 testval

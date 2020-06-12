# Image Classifier for Tiny ImageNet

## Project structure


## Data
The dataset resides in `/project/data` by default. If you do not have it downloaded, run the following commands.
Downloading and unpacking our dataset will take a while, but it will only have to be done once.
```
export DATA_PATH="/project/data"

wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -O $DATA_PATH/tiny-imagenet-200.zip
[ ! -d $DATA_PATH/tiny-imagenet-200 ] && unzip $DATA_PATH/tiny-imagenet-200.zip -d $DATA_PATH
rm $DATA_PATH/tiny-imagenet-200.zip
```

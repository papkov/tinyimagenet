# Image Classifier for Tiny ImageNet

## Project structure
| Local directory | Description | 
|:--------------- |:----------- |
| `data/` | Data | 
| `modules/` | Python modules | 
| `config/` | Configuration files | 
| `notebooks/` | Jupyter notebooks |
| `results/` | Logs and results | 
| `tests/` | Minimalistic tests |  
| `train.py` | Training script |
| `evaluate.py` | Evaluation script |

## Data
The dataset resides in `/project/data` by default. If you do not have it downloaded, run the following commands.
Downloading and unpacking our dataset will take a while, but it will only have to be done once.
```
export DATA_PATH="/project/data"

wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -O $DATA_PATH/tiny-imagenet-200.zip
[ ! -d $DATA_PATH/tiny-imagenet-200 ] && unzip $DATA_PATH/tiny-imagenet-200.zip -d $DATA_PATH
rm $DATA_PATH/tiny-imagenet-200.zip
```

## Config
Training is orchestrated by `hydra` config. It saves each run with the respective config
in a separate folder in `results/model/datetime`. You can manipulate configs either from file before running `make train`
or by substituting parameters. See [hydra docs](hydra.cc) and the following section for details. 

## Training
Training parameters are specified in `config/config.yaml` or by substitution. For example, we can train 
`torchvision` version of ResNet18 and the custom one using the following two commands respectively:
```
python train.py model.module=torchvision model.arch=resnet18
python train.py model.module=models model.arch=resnet18
``` 

## Evaluation
Provide the path to results folder which you want to evaluate. 
`evaluate.py` will build the model by hydra config in the folder and write logs. Call `-h` to see other parameters.
```
python evaluate.py -r results/models.resnet18/2020-06-17_10-30-50 -p val 
```

## Augmentation
We use `albumentations` serialization to store augmentation parameters. See details [here](https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/serialization.ipynb).
Default set of augmentations is in `config/augmentation/default.yaml` include random crop, rotation, scale, contrast, gamma. 

| File | Description | 
|:--------------- |:----------- |
| `dataset.py` | Data handling | 
| `evaluate.py` | Evaluation script | 
| `models.py` | Custom models | 
| `pytorch_typing.py` | Static types |
| `runner.py` | Training loop logic | 
| `train.py` | Training script | 
| `transform.py` | Albumentations transforms | 

## Train
`python modules\train.py --help`
```
train is powered by Hydra.

== Configuration groups ==
Compose your configuration from those groups (group=option)

augmentation: default, minimum
data: tinyimagenet


== Config ==
Override anything in the config (foo.bar=value)

augmentation:
  root: config/augmentation/default.yaml
data:
  classes: 200
  name: tinyimagenet
  root: data/tiny-imagenet-200
  test: test
  train: train
  train_labels: words.txt
  val: val
  val_labels: val_annotations.txt
model:
  arch: resnet18
  module: models
  pretrained: false
optimizer:
  name: SGD
  parameters:
    lr: 0.1
    momentum: 0.9
    weight_decay: 1.0e-05
results:
  checkpoints:
    name: model
    root: checkpoints
    tag: tensorboard
scheduler:
  name: CosineAnnealingLR
  parameters:
    T_max: 23460
train:
  batch_size: 128
  epochs: 30
  monitor: val_acc
  num_workers: 4
```

For example:
```
python modules/train.py model.module=torchvision model.arch=resnet18
```

## Evaluation

`python modules\evaluate.py --help`
```
usage: evaluate.py [-h] [-r RESULTS] [-p {train,val,test}] [-d {cuda,cpu}]

optional arguments:
  -h, --help            show this help message and exit
  -r RESULTS, --results RESULTS
                        results root
  -p {train,val,test}, --data_part {train,val,test}
                        data partition to evaluate on
  -d {cuda,cpu}, --device {cuda,cpu}
```

For example:
```
python modules/evaluate.py -r results/models.resnet18/2020-06-17_10-30-50 -p val 
```
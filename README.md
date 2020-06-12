# Research Engineer test task: Image Classifier for Tiny ImageNet
## Introduction
Dear applicant! We are asking you to implement a thoughtful pipeline for a rather simple toy task.

You are given the dataset from the Tiny ImageNet Challenge which is the default final project for Stanford CS231N course. It runs similar to the ImageNet challenge (ILSVRC). The goal of the original challenge is for you to do as well as possible on the Image Classification problem.

Although the goal of this task is not just to do as good as possible on the Image Classification problem. We would rather encourage you to demonstrate your best practices and skills of rapid prototyping reliable and maintainable pipelines.

We provide you with [a simple pytorch baseline](https://github.com/neuromation/test-task/blob/master/notebooks/baseline.ipynb). You are free to use it or to design the entire solution from scratch. We are not restricting you with the frameworks, you can install any package you need and organise your files however you want. Just please make sure to provide all the sources, checkpoints, logs, visualisations etc. It is desirable, but not demanded, that you use our platform for the development as it would ease the process of your further on-boarding. If you decide to use our platform setup, it is all already taken care of. We will just review the artifacts of your work on our storage. Otherwise, it is your responsibility.

To add some measurable results to the task, your final goal will be to achieve the best accuracy on the provided test split of the Tiny ImageNet dataset. Also, you are expected to show all the visualizations you find necessary alongside the final evaluation metrics. We already took care of a tensorboard setup for you, so you can track some of your plots there.

We would emphasize once again that your goal is to show your best practices as a researcher and developer in approaching deep learning tasks. Good luck!

# Development
For your convenience, the team has created the environment using [Neuro Platform](https://neu.ro), so you can jump into problem-solving right away.

# References

* [Recipe for Training Neural Networks, A. Karpathy](https://karpathy.github.io/2019/04/25/recipe/)

## Quick Start

Sign up at [neu.ro](https://neu.ro) and setup your local machine according to [instructions](https://neu.ro/docs).
 
Then run:

```shell
neuro login
make setup
make jupyter
```

See [Help.md](HELP.md) for the detailed Neuro Project Template Reference.

## Developing Locally

### With `cpu` support
To run `jupyter-notebook` locally in `Docker` container use the following command 
in the root dir of the project:
```shell
docker run -it --rm \
           -p 8889:8889 \
           -v $(pwd)/data:/project/data \
           -v $(pwd)/code:/project/code \
           -v $(pwd)/notebooks:/project/notebooks \
           -v $(pwd)/results:/project/results \
           neuromation/base:v1.4 jupyter-notebook --no-browser --port 8889 --ip=0.0.0.0 --allow-root --notebook-dir=/project/notebooks
```

To start a container with `bash` as `entrypoint` use:
 
```shell
docker run -it --rm \
           -p 8889:8889 \
           -v $(pwd)/data:/project/data \
           -v $(pwd)/code:/project/code \
           -v $(pwd)/notebooks:/project/notebooks \
           -v $(pwd)/results:/project/results \
           --entrypoint /bin/bash \
           neuromation/base:v1.4
```

### With `gpu` support

If you have installed `nvidia-docker v2` you can use the previous commands with 
`gpu` support. You just need to add `--gpus all` parameter in `docker run` command.

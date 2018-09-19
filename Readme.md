# Requirements
- docker
- docker-compose
- nvidia docker 2 with nvidia-runtime setup by defaults
- [kaggle-api](https://github.com/Kaggle/kaggle-api])
- Optional
    - virtualenv
    - yapf

# Setup
Copy paste the `env.dist` file to `.env` and edit the specified paths:

    cp env.dist .env

Build the docker images:

    make build

Download the competion data:

    make download-data

# Usage
The Makefile contains the following commands:

    make docker-bash # instantiate a bash session from the learning-tool image
    make format # format all python files using yapf
    make submit # submit the predictions to kaggle
    make tensorbard # instantiate a docker container running tensorboard accessible on localhost:8888

# Algorithm
- The current pipeline is composed of:
    - an ensemble of resnet18 that predicts if the image as salt in it (ModelsEnsemble(NullMaskClassifier))
    - a Unet model, trained from scratch on the images not rejected by the ModelsEnsemble
    - a post processing step using crf to smooth predictions

# TODO
- Implement training checkpoints during training: https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/2
- Write a small notebook to explain first dataset transform
    - check if assumption np.sum(np.std(image, axis=2) == 0) is always True
- Add saving predictions of the null_mask_classifier model on the train set
- Train using https://www.kdnuggets.com/2017/08/train-deep-learning-faster-snapshot-ensembling.html
- Add ArgumentParsing (and json loader)
- Create module and model factories

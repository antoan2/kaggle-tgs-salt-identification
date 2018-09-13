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
    make tensorbard # instantiate a docker container running tensorboard

# TODO
- implement training checkpoints: https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/2
- write a small notebook to explain first dataset transform
    - check if assumption np.sum(np.std(image, axis=2) == 0) is always True
- Add saving predictions of the null_mask_classifier model on the train set

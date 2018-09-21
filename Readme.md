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
## Makefile commands
The Makefile contains the following commands:

    make docker-bash # instantiate a bash session from the learning-tool image
    make format # format all python files using yapf
    make submit # submit the predictions to kaggle
    make tensorbard # instantiate a docker container running tensorboard accessible on localhost:8888
    make notebook # instantiate a docker container running jupyter notebook accessible on localhost:8889 (password="tgschallenge")

## Typical workflow
A typical workflow will look like the following

    make tensorbard # to get logs
    docker-compose run --rm learning-tool python3 null_mask_classifier_trainer.py \
        --model Resnet18 \
        --n_folds 4 \
        --n_epoches 40
    docker-compose run --rm learning-tool python3 segmentaion_trainer.py \
        --model UNet \
        --n_epoches 60 \
        --null_mask_classifier null_mask_classifier-model-Resnet18-n_folds-1-epoches-2-lr-0.01-batch_size-16-timestamp-1537346155.162557
    docker-compose run --rm learning-tool python3 submission.py \
        -nmc null_mask_classifier-model-Resnet18-n_folds-1-epoches-2-lr-0.01-batch_size-16-timestamp-1537346155.162557 \
        -seg segmentation-model-UNet-epoches-1-lr-0.01-batch_size-16-timestamp-1537347431.518587.csv \
        -p
    cp ./learning-tool/src/submission_script.sh .
    bash ./submission_script.sh # It will submit up to 5 files (this is the competition limit, so you can first edit this file)

You can also run the `docker-compose run --rm learning-tool` commands directly in a container bash / ipython by using the `make docker-bash` command.
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
[] Train using https://www.kdnuggets.com/2017/08/train-deep-learning-faster-snapshot-ensembling.html
[x] Add ArgumentParsing (and json loader)
[x] Create module and model factories
[] Complete kaggle submit instruction

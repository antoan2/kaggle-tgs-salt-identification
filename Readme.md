# Requirements
- docker
- nvidia docker 2 with nvidia-runtime setup by defaults

#TODO
- implement training checkpoints: https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/2
- write a small notebook to explain first dataset transform
    - check if assumption np.sum(np.std(image, axis=2) == 0) is always True
- Add saving predictions of the null_mask_classifier model on the train set

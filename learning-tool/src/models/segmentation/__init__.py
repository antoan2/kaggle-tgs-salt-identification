from .unet import UNet
from .unet_new import UNet11, UNet16, AlbuNet

models = {}
models['UNet'] = UNet
models['UNet11'] = UNet11
models['UNet16'] = UNet16
models['AlbuNet'] = AlbuNet

import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.v2 import GaussianNoise

# sensor noise
def sensor_noise(img):
    h, w = img.shape

    # poisson
    poisson = torch.poisson(img)

    # gaussian
    gaussian = torch.randn(h, w)

    return poisson + gaussian

def make_psf(h, w, points=5, seed=999):
    torch.manual_seed(seed)

    psf = torch.zeros((1, 1, h, w))

    # generate random points
    x = torch.randint(2, h - 2, (points,))
    y = torch.randint(2, w - 2, (points,))
    psf[0, 0, x, y] = 1.0

    # blur (9x9)
    kernel = torch.ones((1, 1, 9, 9)) / 81.0
    psf = F.conv2d(psf, kernel, padding = 9 // 2)

    # normalize
    psf = psf / torch.sum(psf)

    return psf


# normal lens forward model
def normal_lens_fwd(img):
    
    # psf does nothing so we can skip

    # add sensor noise
    noisy = sensor_noise(img)

    # clipping
    min_lum = 0
    max_lum = 255
    clipped = torch.clip(noisy, min_lum, max_lum)

    return clipped

# lensless system forward model
def lensless_fwd(img):
    h, w = img.shape

    # psf
    psf = make_psf(h, w)

    # apply psf
    img_4d = img.unsqueeze(0).unsqueeze(0)
    img_conv = F.conv2d(img_4d, psf)
    img_conv = img_conv.squeeze() # to get rid of batch + channel dimensions
    # to extend to color, just apply psf to all color channels

    # add sensor noise
    noisy = sensor_noise(img_conv)

    # clipping
    min_lum = 0
    max_lum = 255
    clipped = torch.clip(noisy, min_lum, max_lum)

    return clipped
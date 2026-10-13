# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "torch.nn.functional",
#   "torchvision.transforms.v2"
# ]
# ///

import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.v2 import GaussianNoise

# sensor noise
def sensor_noise(img):

    # poisson
    poisson = torch.poisson(img)

    # gaussian
    gaussian = torch.randn_like(img)

    return poisson + gaussian

def make_psf(ksize=9, points=5, seed=999):
    torch.manual_seed(seed)

    psf = torch.zeros((1, 1, ksize, ksize))

    # generate random points
    x = torch.randint(2, ksize, (points,))
    y = torch.randint(2, ksize, (points,))
    psf[0, 0, x, y] = 1.0

    # blur (3x3)
    kernel = torch.ones((1, 1, 3, 3)) / 9.0
    psf = F.conv2d(psf, kernel, padding = 1)

    # normalize
    psf = psf / torch.sum(psf)

    return psf


# normal lens forward model
def normal_fwd(img):
    
    # psf does nothing so we can skip

    # add sensor noise
    noisy = sensor_noise(img)

    # clipping
    min_lum = 0
    max_lum = 255
    clipped = torch.clip(noisy, min_lum, max_lum)

    return clipped

# lensless system forward model
def lensless_fwd(img, psf):

    # apply psf
    img_4d = img.unsqueeze(0)
    img_conv = F.conv2d(img_4d, psf, padding="same")
    img_conv = img_conv.squeeze(0) # to get rid of batch + channel dimensions
    # to extend to color, just apply psf to all color channels

    # add sensor noise
    noisy = sensor_noise(img_conv)

    # clipping
    min_lum = 0
    max_lum = 255
    clipped = torch.clip(noisy, min_lum, max_lum)

    return clipped

# burst method (baseline)
def burst_m(img, num_frames=8, exposure=0.1):

    burst_frames = []
    underexposed = img * exposure

    for _ in range(num_frames):
        frame = normal_fwd(underexposed)
        burst_frames.append(frame)

    return burst_frames
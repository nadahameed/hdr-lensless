# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "torch.nn.functional"
# ]
# ///

import torch
import torch.nn.functional as F
import numpy as np

def reconstruct_burst(burst_frames, exposure=0.1):

    # stack frames
    stacked = torch.stack(burst_frames, dim=0)

    # temporal denoising
    # this is incomplete/basic
    # original paper combined a patch from each with the corresponding part of 
    # the reference frame
    # there is currently no reference frame used
    denoised = torch.mean(stacked, dim=0)

    reconstructed = denoised / exposure

    return reconstructed

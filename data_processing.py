import torch
import torchvision
import torchvision.transforms.v2 as T

def cifar_to_hdr(img, max_lum = 1000.0):
    # cifar rgb tensor --> 1 channel hdr scene

    # grayscale
    gray_T = T.Grayscale(num_output_channels=1)
    gray = gray_T(img)

    # scale intensity
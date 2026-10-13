# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "torchvision"
# ]
# ///

import torch
import torchvision
import torchvision.transforms.v2 as T
import os

def cifar_to_hdr(img, max_lum = 1000.0, gamma = 3.0):
    # cifar rgb tensor --> 1 channel hdr scene

    # grayscale
    gray = T.functional.rgb_to_grayscale(img, num_output_channels=1)

    # scale intensity --> exponential curve to push brightest pixels higher
    hdr = (gray ** gamma) * max_lum

    return hdr

def make_cifar_dataset(num_samples=10):

    # create directories
    raw = "data/raw"
    hdr = "data/hdr"

    os.makedirs(raw, exist_ok=True)
    os.makedirs(hdr, exist_ok=True)

    print("loading cifar...")

    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True)
    ])

    cifar = torchvision.datasets.CIFAR10(
        root=raw,
        train=True,
        download=True,
        transform=transform
    )

    # process to hdr and save
    print("processing to hdr.....")
    hdr_imgs = []
    for i in range(num_samples):
        img, _ = cifar[i]
        hdr_img = cifar_to_hdr(img)
        hdr_imgs.append(hdr_img)

    # save as a pytorch file
    hdr_dataset = torch.stack(hdr_imgs)
    save_path = os.path.join(hdr, "hdr_cifar10.pt")
    torch.save(hdr_dataset, save_path)
    print(f"saved processed hdr dataset")

def make_synthetic_dataset(num_samples=10, size=32):
    hdr = "data/hdr/synthetic"
    os.makedirs(hdr, exist_ok=True)

    hdr_imgs = []
    for _ in range(num_samples):
        img = torch.zeros((1, size, size), dtype=torch.float32)

        # random bright
        rx, ry = torch.randint(4, size - 4, (2,))
        img[0, rx:rx+3, ry:ry+3] = 1000.0

        # random dim
        rx, ry = torch.randint(4, size - 12, (2,))
        img[0, rx:rx+8, ry:ry+8] = torch.rand(1) * 40.0 + 10.0

        hdr_imgs.append(img)

    hdr_dataset = torch.stack(hdr_imgs)
    save_path = os.path.join(hdr, "synthetic.pt")
    torch.save(hdr_dataset, save_path)


if __name__ == "__main__":
    make_cifar_dataset(num_samples=10)
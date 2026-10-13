# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "torchvision"
# ]
# ///

import torch
import os
from forward import normal_fwd, lensless_fwd, burst_m, make_psf
from reconstruct import wiener_deconv, fista_recon, admm_recon, burst_recon

def simulate(datapath="data/hdr/hdr_cifar10.pt", outputpath="data/simulated"):
    os.makedirs(outputpath, exist_ok=True)

    hdr_scenes = torch.load(datapath)
    n, _, h, w = hdr_scenes.shape # num images, color channels, h, w

    psf = make_psf()

    normal_imgs = []
    lensless_imgs = []
    burst_imgs = []

    exposure = 0.1

    print("running fwd models now")
    for i in range(n):
        img = hdr_scenes[i]

        normal = normal_fwd(img)
        lensless = lensless_fwd(img, psf)
        burst = burst_m(img, exposure=exposure)
        burst = torch.stack(burst)

        normal_imgs.append(normal)
        lensless_imgs.append(lensless)
        burst_imgs.append(burst)

    print("finished running fwd models")

    # saving stuff idk
    normal_tensor = torch.stack(normal_imgs)
    lensless_tensor = torch.stack(lensless_imgs)
    burst_tensor = torch.stack(burst_imgs)

    data = {
        "ground_truth": hdr_scenes,
        "psf": psf,
        "normal": normal_tensor,
        "lensless": lensless_tensor,
        "burst": burst_tensor
    }

    # print("Ground truth:", hdr_scenes.shape)
    # print("PSF:", psf.shape)
    # print("Normal:", normal_tensor.shape)
    # print("Lensless:", lensless_tensor.shape)
    # print("Burst:", burst_tensor.shape)

    # print("PSF sum:", psf.sum().item())
    # print("Normal range:", normal_tensor.min().item(), normal_tensor.max().item())
    # print("Lensless range:", lensless_tensor.min().item(), lensless_tensor.max().item())

    save_path = os.path.join(outputpath, "simulated_data.pt")
    torch.save(data, save_path)

# reconstruction
def reconstruct(datapath="data/simulated/simulated_data.pt", outputpath="data/simulated"):
    data = torch.load(datapath, map_location="cpu", weights_only=True)

    lensless_tensor = data["lensless"]
    psf = data["psf"]
    n = lensless_tensor.shape[0]

    wiener, fista, admm = [], [], []

    print("running recon on lensless...")
    for i in range(n):
        y = lensless_tensor[i]

        wiener.append(wiener_deconv(y, psf))
        fista.append(fista_recon(y, psf))
        admm.append(admm_recon(y, psf, n=30))

    results = {
        "wiener": torch.stack(wiener),
        "fista": torch.stack(fista),
        "admm": torch.stack(admm)
    }

    savepath = os.path.join(outputpath, "recon_results.pt")
    torch.save(results, savepath)
    print(f"recon results saved to {savepath}")


if __name__ == "__main__":
    simulate()
    reconstruct()
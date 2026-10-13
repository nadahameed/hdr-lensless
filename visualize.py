# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "torchvision",
#   "matplotlib"
# ]
# ///

import torch
import os
import matplotlib.pyplot as plt

def visualize_sim(datapath="data/simulated/simulated_data.pt", index=0):

    # load previously simulated data
    data = torch.load(datapath, map_location="cpu", weights_only=True)

    # get images for one scene
    ground_truth = data["ground_truth"][index, 0]
    normal = data["normal"][index, 0]
    lensless = data["lensless"][index, 0]
    burst = data["burst"][index, 0]

    # average of burst frames
    burst_avg = torch.mean(burst, dim=0).squeeze(0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(ground_truth.numpy(), cmap="gray", vmin=0, vmax=100)
    axes[0].set_title("ground truth: 0 to 1000")

    axes[1].imshow(normal.numpy(), cmap="gray", vmin=0, vmax=100)
    axes[1].set_title("normal lens: 0 to 255")

    axes[2].imshow(lensless.numpy(), cmap="gray", vmin=0, vmax=100)
    axes[2].set_title("lensless: 0 to 255")

    axes[3].imshow(burst_avg.numpy(), cmap="gray", vmin=0, vmax=100)
    axes[3].set_title("burst avg: 0 to 255")

    for a in axes:
        a.axis("off")

    plt.tight_layout()
    plt.show()

def visualize_recon(simpath="data/simulated/simulated_data.pt", recpath="data/simulated/recon_results.pt", index=0):

    # load sim
    simdata = torch.load(simpath, map_location="cpu", weights_only=True)

    ground_truth = simdata["ground_truth"][index].squeeze()
    normal = simdata["normal"][index].squeeze()
    lensless = simdata["lensless"][index].squeeze()
    burst = simdata["burst"][index]
    psf = simdata["psf"].squeeze()

    burst_avg = torch.mean(burst, dim=0).squeeze(0)

    # load recon if they exist
    recdata = None
    if os.path.exists(recpath):
        recdata = torch.load(recpath, map_location="cpu", weights_only=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 4))

    axes[0, 0].imshow(ground_truth.numpy(), cmap="gray")
    axes[0, 0].set_title("ground truth hdr")

    axes[0, 1].imshow(normal.numpy(), cmap="gray")
    axes[0, 1].set_title("normal lens clipped")

    axes[0, 2].imshow(lensless.numpy(), cmap="gray")
    axes[0, 2].set_title("lensless capture (blurred)")

    axes[0, 3].imshow(burst_avg.numpy(), cmap="gray")
    axes[0, 3].set_title("burst avg")

    if recdata is not None:
        rec_wiener = recdata["wiener"][index].squeeze()
        rec_fista = recdata["fista"][index].squeeze()
        rec_admm = recdata["admm"][index].squeeze()

        axes[1, 0].imshow(rec_wiener.numpy(), cmap="gray")
        axes[1, 0].set_title("wiener deconv")

        axes[1, 1].imshow(rec_fista.numpy(), cmap="gray")
        axes[1, 1].set_title("fista")

        axes[1, 2].imshow(rec_admm.numpy(), cmap="gray")
        axes[1, 2].set_title("ADMM")
        
    else:
        axes[1, 0].set_title("Wiener (Not run)")
        axes[1, 1].set_title("FISTA (Not run)")
        axes[1, 2].set_title("ADMM (Not run)")

    axes[1, 3].imshow(psf.numpy(), cmap="gray")
    axes[1, 3].set_title("psf")

    for a in axes.ravel():
        a.axis("off")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/comparison_sample_{index}.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    visualize_recon()
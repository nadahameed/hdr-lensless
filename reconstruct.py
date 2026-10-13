# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "torch.fft",
# ]
# ///

import torch
import torch.fft
import numpy as np

def burst_recon(burst_frames, exposure=0.1):

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

# psf helper
def psf_shift(psf, h, w):

    # dimensions
    psf_2d = psf.squeeze()

    kh, kw = psf_2d.shape

    # pad psf
    psf_upd = torch.zeros((h, w))
    i = (h - kh) // 2
    j = (w - kw) // 2
    psf_upd[i:i+kh, j:j+kw] = psf_2d

    # center to top left
    psf_upd = torch.fft.ifftshift(psf_upd)

    # fourier representation
    H = torch.fft.fft2(psf_upd)

    return H

def wiener_deconv(y, psf, K=1e-2):

    # format tensors
    if y.dim() > 2:
        y_2d = y.squeeze()
    else:
        y_2d = y

    if psf.dim() > 2:
        psf_2d = psf.squeeze()
    else:
        psf_2d = psf

    h, w = y_2d.shape

    # computed 2d ffts
    Y = torch.fft.fft2(y_2d)
    H = psf_shift(psf, h, w)

    # apply wiener filter eq
    Hc = torch.conj(H) # H*
    Hm = torch.abs(H) ** 2 # H^2

    W = Hc / (Hm + K) # wiener filter
    X = W * Y

    # convert back to spatial domain
    X = torch.fft.ifft2(X).real

    # non neg
    X = torch.clamp(X, min=0.0)

    return X.unsqueeze(0)

def fft_helper(psf, h, w):
    psf_2d = psf.squeeze()
    kh, kw = psf_2d.shape

    ph = h + kh - 1
    pw = w + kw - 1

    # flip psf
    kernel = torch.flip(psf_2d, dims=(0,1))

    # fourier representations
    H = torch.fft.fft2(kernel, s=(ph, pw))

    # output crop
    top = kh // 2
    left = kw // 2

    def forward(x):
        X = torch.fft.fft2(x, s=(ph, pw))
        full = torch.fft.ifft2(X * H).real
        return full[top:top + h, left:left + w]

    def adjoint(r):
        padded = torch.zeros((ph, pw))
        padded[top:top + h, left:left + w] = r

        R = torch.fft.fft2(padded)
        full = torch.fft.ifft2(R * torch.conj(H)).real

        return full[:h, :w]

    return forward, adjoint, H

def fista_recon(y, psf, n=75, a=1e-1):

    if y.dim() > 2:
        y_2d = y.squeeze()
    else:
        y_2d = y

    if psf.dim() > 2:
        psf_2d = psf.squeeze()
    else:
        psf_2d = psf

    h, w = y_2d.shape

    # otfs + helpers
    forward, adjoint, H = fft_helper(psf, h, w)

    # FISTA initializations
    x = torch.zeros_like(y_2d) # recon estimate
    z = x.clone() # momentum-accelerated search point
    t = 1.0 # nesterov momentum factor

    # reconstruction loop
    for k in range(n):
        x_old = x.clone()

        # predict sensor measurement from current estimate
        y_est = forward(z)

        # error
        residual = y_est - y_2d

        # backproject res to calculate gradient
        grad = adjoint(residual)

        # gradient descent update + non neg
        x = torch.clamp(z - a * grad, min=0.0)

        # Nesterov acceleration
        t_1 = (1.0 + (1.0 + 4.0 * (t**2))**0.5) / 2.0
        beta = (t - 1.0) / t_1
        z = x + beta * (x - x_old)
        t = t_1

    return x.unsqueeze(0)

def admm_recon(y, psf, n, mu1=1e-3, mu2=1e-3, mu3=1e-3, tau=1e-4):

    if y.dim() > 2:
        y_2d = y.squeeze()
    else:
        y_2d = y

    if psf.dim() > 2:
        psf_2d = psf.squeeze()
    else:
        psf_2d = psf

    h, w = y_2d.shape
    kh, kw = psf_2d.shape

    ph = h + kh - 1
    pw = w + kw - 1

    H = psf_shift(psf, ph, pw)
    Hc = torch.conj(H)
    Hm = torch.abs(H) ** 2

    # TV (finite difference) operators in fourier space
    Dx = torch.tensor([[0, -1, 1]])
    Dy = torch.tensor([[0], [-1], [1]])

    # in fourier spaec
    Dx = torch.fft.fft2(Dx, s=(ph, pw))
    Dy = torch.fft.fft2(Dy, s=(ph, pw))

    # conjugates
    Dxc = torch.conj(Dx)
    Dyc = torch.conj(Dy)

    # Psi ?
    P = torch.abs(Dx) ** 2 + torch.abs(Dy) ** 2

    # crop mask C
    cropy = kh // 2
    cropx = kw // 2

    C = torch.zeros((ph, pw))
    C[cropy:cropy+h, cropx:cropx+w] = 1.0

    # padding
    y_pad = torch.zeros((ph, pw))
    y_pad[cropy:cropy+h, cropx:cropx+w] = y_2d

    def soft_thresh(v, t):
        return torch.sgn(v) * torch.clamp(torch.abs(v) - t, min=0.0)

    # state vars and dual multipliers
    v = torch.zeros((ph, pw))
    x = torch.zeros((ph, pw))
    wv = torch.zeros((ph, pw))
    ux = torch.zeros((ph, pw))
    uy = torch.zeros((ph, pw))

    xi = torch.zeros((ph, pw))
    eta = torch.zeros((ph, pw))
    rx = torch.zeros((ph, pw))
    ry = torch.zeros((ph, pw))

    denominator = mu1 * Hm + mu2 + mu3 * P

    # admm iterations
    for _ in range(n):
        # update v
        M = torch.fft.ifft2(torch.fft.fft2(x - xi) * Hc).real

        ux_d = ux - rx
        uy_d = uy - ry
        Pa = torch.fft.ifft2(torch.fft.fft2(ux_d) * Dxc + torch.fft.fft2(uy_d) * Dyc).real

        b = mu1 * M + mu2 * (wv - eta) + mu3 * Pa

        Vh = torch.fft.fft2(b) / denominator
        v = torch.fft.ifft2(Vh).real

        # forward step
        v_conv = torch.fft.ifft2(torch.fft.fft2(v) * H).real
        grad_x = torch.fft.ifft2(torch.fft.fft2(v) * Dx).real
        grad_y = torch.fft.ifft2(torch.fft.fft2(v) * Dy).real

        # update x (sensor crop?)
        x = (C * (y_pad + mu1 * (v_conv + xi)) / (1.0 + mu1)) + ((1.0 - C) * (v_conv + xi))

        # update w (non neg)
        wv = torch.clamp(v + eta, min=0.0)

        # update u (total variation soft thresh)
        ux = soft_thresh(grad_x + rx, tau / mu3)
        uy = soft_thresh(grad_y + ry, tau / mu3)

        # update multipliers
        xi = xi + (v_conv - x)
        eta = eta + (v - wv)
        rx = rx + (grad_x - ux)
        ry = ry + (grad_y - uy)

    x_rv = v[cropy:cropy+h, cropx:cropx+w]
    x_rv = torch.clamp(x_rv, min=0.0)

    return x_rv.unsqueeze(0)
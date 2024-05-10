import numpy as np
import torch

__all__ = ["dct1", "idct1", "dct", "idct"]


def dct1(x: torch.Tensor) -> torch.Tensor:
    """
    Discrete Cosine Transform, Type I (DCT-I).

    Reference:
    * J. Makhoul, "A fast cosine transform in one and two dimensions," in IEEE Transactions on Acoustics, Speech, and Signal Processing,
      vol. 28, no. 1, pp. 27-34, February 1980, doi: 10.1109/TASSP.1980.1163351.
    * https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py

    :param x: the input signal in shape of (..., L) with L is number of samples.
    :return: the DCT-I of the signal over the last dimension.
    """
    # shape: (..., L)
    shape = x.size()
    # shape: (..., L) -> (*, L)
    z = x.view(-1, shape[-1])
    z = torch.cat([z, z.flip([1])[:, 1: -1]], dim=-1)
    z = torch.view_as_real(torch.fft.rfft(z, dim=-1))
    # get the real part only, then turn back the original shape
    # shape: (*, L) -> (..., L)
    z = z[..., 0].view(*shape)
    return z


def idct1(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse Discrete Cosine Transform, Type I (iDCT-I).

    Reference:
    * J. Makhoul, "A fast cosine transform in one and two dimensions," in IEEE Transactions on Acoustics, Speech, and Signal Processing,
      vol. 28, no. 1, pp. 27-34, February 1980, doi: 10.1109/TASSP.1980.1163351.
    * https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py

    :param x: the DCT-I of signal in shape (..., L) with L is number of samples.
    :return: the inverse DCT-I of the signal over the last dimension.
    """
    # shape: (..., L)
    shape = x.size()
    z = dct1(x) / (2 * (shape[-1] - 1))
    return z


def dct(x: torch.Tensor, norm: str = None) -> torch.Tensor:
    """
    Discrete Cosine Transform, Type II (a.k.a. DCT)

    Reference:
    * J. Makhoul, "A fast cosine transform in one and two dimensions," in IEEE Transactions on Acoustics, Speech, and Signal Processing,
      vol. 28, no. 1, pp. 27-34, February 1980, doi: 10.1109/TASSP.1980.1163351.
    * https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py

    :param x: the input signal in shape of (..., L) with L is number of samples.
    :param norm: normalization mode, None (no normalization) or "ortho" (normalize by 1/sqrt(L)).
    :return: the DCT of the signal over the last dimension.
    """
    # shape: (..., L)
    shape = x.size()
    # shape: (..., L) -> (*, L)
    z_v = x.contiguous().view(-1, shape[-1])

    z_v = torch.cat([z_v[:, ::2], z_v[:, 1::2].flip([1])], dim=-1)
    z_v = torch.view_as_real(torch.fft.fft(z_v, dim=-1))

    k = -torch.arange(shape[-1], dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * shape[-1])
    w_r = torch.cos(k)
    w_i = torch.sin(k)

    z = z_v[:, :, 0] * w_r - z_v[:, :, 1] * w_i

    if norm == "ortho":
        z[:, 0] /= np.sqrt(shape[-1]) * 2
        z[:, 1:] /= np.sqrt(shape[-1] / 2) * 2

    # shape: (*, L) -> (..., L)
    z = 2 * z.view(*shape)
    return z


def idct(x: torch.Tensor, norm: str = None) -> torch.Tensor:
    """
    Inverse Discrete Cosine Transform, Type II (a.k.a. DCT-III)

    Reference:
    * J. Makhoul, "A fast cosine transform in one and two dimensions," in IEEE Transactions on Acoustics, Speech, and Signal Processing,
      vol. 28, no. 1, pp. 27-34, February 1980, doi: 10.1109/TASSP.1980.1163351.
    * https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py

    :param x: the DCT of signal in shape of (..., L) with L is number of samples.
    :param norm: normalization mode, None (no normalization) or "ortho" (normalize by 1/sqrt(L)).
    :return: the inverse DCT of the signal over the last dimension.
    """
    # shape: (..., L)
    shape = x.size()
    # shape: (..., L) -> (*, L)
    z_v = x.contiguous().view(-1, shape[-1]) / 2

    if norm == "ortho":
        z_v[:, 0] *= np.sqrt(shape[-1]) * 2
        z_v[:, 1:] *= np.sqrt(shape[-1] / 2) * 2

    k = torch.arange(shape[-1], dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * shape[-1])
    w_r = torch.cos(k)
    w_i = torch.sin(k)

    zt_r = z_v
    zt_i = torch.cat([z_v[:, :1] * 0, -z_v.flip([1])[:, :-1]], dim=-1)

    z_r = zt_r * w_r - zt_i * w_i
    z_i = zt_r * w_i + zt_i * w_r

    z_t = torch.cat([z_r.unsqueeze(2), z_i.unsqueeze(2)], dim=2)
    z_t = torch.fft.irfft(torch.view_as_complex(z_t), n=shape[-1], dim=1)
    z = z_t.new_zeros(z_t.shape)
    z[:, ::2] += z_t[:, :shape[-1] - (shape[-1] // 2)]
    z[:, 1::2] += z_t.flip([1])[:, : (shape[-1] // 2)]

    z = z.view(*shape)
    return z

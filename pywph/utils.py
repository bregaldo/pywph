# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.fft import fftn, ifftn
from torch.autograd import Function
import psutil
import subprocess


def get_precision_type(precision, module="numpy", cplx=False):
    """
    Returns the right dtype object, depending on the module (numpy or torch),
    and if we are dealing with complex numbers or not.

    Parameters
    ----------
    precision : str
        Either 'single' or 'double' precision.
    module : str, optional
        Is this for 'numpy' or 'torch' objects? The default is "numpy".
    cplx : bool, optional
        Are we dealing with complex number or not? The default is False.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if precision == "single":
        if module == "numpy":
            if cplx:
                return np.complex64
            else:
                return np.float32
        elif module == "torch":
            if cplx:
                return torch.complex64
            else:
                return torch.float32
    elif precision == "double":
        if module == "numpy":
            if cplx:
                return np.complex128
            else:
                return np.float64
        elif module == "torch":
            if cplx:
                return torch.complex128
            else:
                return torch.float64
    else:
        raise Exception("precision parameter must be equal to either 'single' or 'double'!")
    raise Exception("Unknown module!")


def to_torch(data, device=None, precision="single"):
    """
    Converts input data to pytorch tensor. The tensor is automatically sent to the specified device,
    and using the specified precision.
    
    Takes as an input a numpy array or a pytorch tensor.

    Parameters
    ----------
    data : array or tensor or list
        Input data.
    device : str, optional
        Target device. The default is None.
    precision : str, optional
        Either 'single' or 'double' precision. The default is np.float32.

    Returns
    -------
    tensor
        Tensor with the same shape than the input data.

    """
    ret = None
    
    if isinstance(data, list): # List
        return to_torch(np.array(data), device=device, precision=precision)
    elif isinstance(data, np.ndarray): # Numpy array
        ret = torch.from_numpy(data.astype(get_precision_type(precision, module="numpy", cplx=np.iscomplexobj(data))))
    elif isinstance(data, torch.Tensor):
        ret = data
    else:
        raise Exception("Unknown data type!")
    if device is not None:
        ret = ret.to(device)
    return ret.contiguous()


def get_gpu_memory_map():
    """
    Get the current gpu usage. (from mjstevens777)

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MiB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def get_memory_available(device):
    """
    Returns available memory in bytes.

    Parameters
    ----------
    device : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if device == "cpu":
        return psutil.virtual_memory().available
    else:
        t = torch.cuda.get_device_properties(device).total_memory
        a = get_gpu_memory_map()[device]
        return t - (a - (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))  // (1024 ** 2)) * 1024 ** 2


def fft(z):
    return fftn(z, s=(-1, -1))


def ifft(z):
    return ifftn(z, s=(-1, -1))


def real(z):
    return z.real


def imag(z):
    return z.imag


def mean(z, axes=(-1, -2), keepdim=False):
    if torch.is_complex(z):
        return torch.mean(z.real, axes, keepdim=keepdim) + 1j*torch.mean(z.imag, axes, keepdim=keepdim)
    else:
        return torch.mean(z, axes, keepdim=keepdim)


def mul(z1, z2):
    zr = real(z1) * real(z2) - imag(z1) * imag(z2)
    zi = real(z1) * imag(z2) + imag(z1) * real(z2)
    z = torch.stack((zr, zi), dim=-1)
    return z


def modulus(z):
    norm = z.norm(p=2, dim=-1, keepdim=True)
    return torch.cat([norm, torch.zeros_like(norm)], -1)


class PhaseHarmonics(Function):
    @staticmethod
    def forward(ctx, z, k, indices_k_0=None, indices_k_1=None, indices_other_k=None):
        """
            z: (..., N_cov, M, N, 2)
            k: (N_cov)

        Returns:
            (..., N_cov, M, N, 2)

        """
        z =  z.detach()
        
        if indices_k_0 is None:
            indices_k_0 = torch.where(k == 0)[0]
        if indices_k_1 is None:
            indices_k_1 = torch.where(k == 1)[0]
        if indices_other_k is None:
            indices_other_k = torch.where(k >= 2)[0]
        other_k = k[indices_other_k]

        z_0 = z[..., indices_k_0, :, :]
        z_other_k = z[..., indices_other_k, :, :]

        x_0, y_0 = real(z_0), imag(z_0)  # (..., ??, M, N)  (..., ??, M, N)
        r_0 = torch.abs(z_0)
        del z_0
        
        x, y = real(z_other_k), imag(z_other_k)  # (..., ??, M, N)  (..., ??, M, N)
        r = torch.abs(z_other_k)  # (..., ??, M, N)
        theta = torch.angle(z_other_k)  # (..., ??, M, N)
        del z_other_k
        
        ctx.save_for_backward(x_0, y_0, x, y, k, indices_k_0, indices_k_1, indices_other_k)
        del x_0, y_0, x, y
        
        ktheta = other_k.unsqueeze(-1).unsqueeze(-1) * theta  # (??, M, N)
        eiktheta = torch.cos(ktheta) + 1j*torch.sin(ktheta)  # (??, M, N)
        result_other_k = r * eiktheta  # (..., ??, M, N)
        del r, theta, ktheta, eiktheta

        result = z.clone()
        result[..., indices_k_0, :, :] = r_0 + 1j*0 # 1j*0 for dtype consistency
        result[..., indices_other_k, :, :] = result_other_k

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
            grad_output: (..., N_cov, M, N, 2)

        Returns: (..., N_cov, M, N, 2), (N_cov) (k gradient not used)

        """
        x_0, y_0, x, y, k, indices_k_0, indices_k_1, indices_other_k = ctx.saved_tensors  # (..., ??, M, N)  (..., ??, M, N) (..., ??, M, N)  (..., ??, M, N) (N_cov) (??) (??) (??)

        other_k = k[indices_other_k].unsqueeze(-1).unsqueeze(-1) # (??, 1, 1)

        theta = torch.atan2(y, x)   # (..., ??, M, N)
        ktheta = other_k * theta  # (..., ??, M, N)
        cosktheta = torch.cos(ktheta)  # (..., ??, M, N)
        sinktheta = torch.sin(ktheta)  # (..., ??, M, N)
        costheta = torch.cos(theta)  # (..., ??, M, N)
        sintheta = torch.sin(theta)  # (..., ??, M, N)

        costheta_cosktheta = costheta * cosktheta  # (..., ??, M, N)
        sintheta_sinktheta = sintheta * sinktheta  # (..., ??, M, N)
        costheta_sinktheta = costheta * sinktheta  # (..., ??, M, N)
        sintheta_cosktheta = sintheta * cosktheta  # (..., ??, M, N)

        df1dx = costheta_cosktheta + other_k*sintheta_sinktheta  # (..., ??, M, N)
        df2dx = costheta_sinktheta - other_k*sintheta_cosktheta  # (..., ??, M, N)
        df1dy = sintheta_cosktheta - other_k*costheta_sinktheta  # (..., ??, M, N)
        df2dy = sintheta_sinktheta + other_k*costheta_cosktheta  # (..., ??, M, N)

        dx1_other_k = df1dx*grad_output[..., indices_other_k, :, :, 0] + df2dx*grad_output[..., indices_other_k, :, :, 1]  # (..., ??, M, N)
        dx2_other_k = df1dy*grad_output[..., indices_other_k, :, :, 0] + df2dy*grad_output[..., indices_other_k, :, :, 1]  # (..., ??, M, N)

        dx1_k_0 = torch.cos(torch.atan2(y_0, x_0)) * grad_output[..., indices_k_0, :, :, 0]  # (..., ??, M, N)
        dx2_k_0 = torch.sin(torch.atan2(y_0, x_0)) * grad_output[..., indices_k_0, :, :, 0]  # (..., ??, M, N)

        dx1_k_1 = grad_output[..., indices_k_1, :, :, 0]  # (..., ??, M, N)
        dx2_k_1 = grad_output[..., indices_k_1, :, :, 1]  # (..., ??, M, N)

        dx1 = grad_output.new_zeros((grad_output.size()[:5]))
        dx2 = grad_output.new_zeros((grad_output.size()[:5]))

        dx1[..., indices_k_0, :, :] = dx1_k_0
        dx1[..., indices_k_1, :, :] = dx1_k_1
        dx1[..., indices_other_k, :, :] = dx1_other_k

        dx2[..., indices_k_0, :, :] = dx2_k_0
        dx2[..., indices_k_1, :, :] = dx2_k_1
        dx2[..., indices_other_k, :, :] = dx2_other_k

        return torch.stack((dx1, dx2), -1), k, indices_k_0, indices_k_1, indices_other_k  # (..., N_cov, M, N, 2), (N_cov,) (??) (??) (??) (only first gradient is used)


class PowerHarmonics(Function):
    @staticmethod
    def forward(ctx, z, k):
        """
            z: (..., N_cov, M, N, 2)
            k: (N_cov)

        Returns:
            (..., N_cov, M, N, 2)

        """
        z =  z.detach()
        k_unsqueeze = k.unsqueeze(-1).unsqueeze(-1) # (..., N_cov, 1, 1)
        x, y = real(z), imag(z)  # (..., N_cov, M, N) (..., N_cov, M, N)
        r = torch.abs(z)  # (..., N_cov, M, N)
        theta = torch.angle(z)  # (..., N_cov, M, N)
        ktheta = k_unsqueeze * theta  # (..., N_cov, M, N)
        eiktheta = torch.cos(ktheta) + 1j*torch.sin(ktheta)  # (..., N_cov, M, N)
        k_tilde = k_unsqueeze.clone()   # (..., N_cov, 1, 1)
        k_tilde[torch.where(k_unsqueeze == 0)] = 1  # (..., N_cov, 1, 1)
        ctx.save_for_backward(x, y, r, k_unsqueeze, k_tilde)
        return torch.pow(r, k_tilde) * eiktheta  # (..., N_cov, M, N)

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: (..., N_cov, M, N, 2)

        Returns: (..., N_cov, M, N, 2), (N_cov) (k gradient not used)

        """
        x, y, r, k_unsqueeze, k_tilde = ctx.saved_tensors  # (..., N_cov, M, N) (..., N_cov, M, N) (..., N_cov, M, N) (..., N_cov, 1, 1) (..., N_cov, 1, 1)
        theta = torch.atan2(y, x)  # (..., N_cov, M, N)
        ktheta = k_unsqueeze * theta  # (..., N_cov, M, N)
        cosktheta = torch.cos(ktheta)  # (..., N_cov, M, N)
        sinktheta = torch.sin(ktheta)  # (..., N_cov, M, N)
        costheta = torch.cos(theta)  # (..., N_cov, M, N)
        sintheta = torch.sin(theta)  # (..., N_cov, M, N)

        costheta_cosktheta = costheta * cosktheta  # (..., N_cov, M, N)
        sintheta_sinktheta = sintheta * sinktheta  # (..., N_cov, M, N)
        costheta_sinktheta = costheta * sinktheta  # (..., N_cov, M, N)
        sintheta_cosktheta = sintheta * cosktheta  # (..., N_cov, M, N)

        r_k_tilde_minus_1 = torch.pow(r, k_tilde-1)

        df1dx = r_k_tilde_minus_1*(k_tilde*costheta_cosktheta + k_unsqueeze*sintheta_sinktheta)  # (..., N_cov, M, N)
        df2dx = r_k_tilde_minus_1*(k_tilde*costheta_sinktheta - k_unsqueeze*sintheta_cosktheta)  # (..., N_cov, M, N)
        df1dy = r_k_tilde_minus_1*(k_tilde*sintheta_cosktheta - k_unsqueeze*costheta_sinktheta)  # (..., N_cov, M, N)
        df2dy = r_k_tilde_minus_1*(k_tilde*sintheta_sinktheta + k_unsqueeze*costheta_cosktheta)  # (..., N_cov, M, N)

        dx1 = df1dx*grad_output[..., 0] + df2dx*grad_output[..., 1]  # (..., N_cov, M, N)
        dx2 = df1dy*grad_output[..., 0] + df2dy*grad_output[..., 1]  # (..., N_cov, M, N)
        
        return torch.stack((dx1, dx2), -1), k_unsqueeze[:, 0, 0]  # (..., N_cov, M, N, 2) (N_cov) (k gradient not used)

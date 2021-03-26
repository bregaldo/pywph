# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.fft import fft2, ifft2
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
    if module == "numpy":
        mod = np
    elif module == "torch":
        mod = torch
    else:
        raise Exception("Unknown module!")
    
    if precision == "single":
        if cplx:
            return mod.complex64
        else:
            return mod.float32
    elif precision == "double":
        if cplx:
            return mod.complex128
        else:
            return mod.float64
    else:
        raise Exception("precision parameter must be equal to either 'single' or 'double'!")


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


def to_numpy(data):
    """
    Converts input data into numpy arrays.
    Accepted input types: list, np.ndarray, torch.tensor

    Parameters
    ----------
    data : list, np.ndarray, torch.tensor
        Input data.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    np.ndarray
        Numpy array version of input data.

    """
    if isinstance(data, list): # List
        return np.array(data)
    elif isinstance(data, np.ndarray): # Numpy array
        return data
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    else:
        raise Exception("Unknown data type!")


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
        if isinstance(device, torch.device):
            device = device.index
        t = torch.cuda.get_device_properties(device).total_memory
        a = get_gpu_memory_map()[device]
        return t - (a - (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))  // (1024 ** 2)) * 1024 ** 2


def fft(z):
    """
    Torch 2D FFT wrapper. No padding. The FFT is applied to the 2 last dimensions.

    Parameters
    ----------
    z : tensor
        Input.

    Returns
    -------
    tensor
        Output.

    """
    return fft2(z, s=(-1, -1))


def ifft(z):
    """
    Torch 2D IFFT wrapper. No padding. The IFFT is applied to the 2 last dimensions.

    Parameters
    ----------
    z : tensor
        Input.

    Returns
    -------
    tensor
        Output.

    """
    return ifft2(z, s=(-1, -1))


def real(z):
    """
    Returns the real part of pytorch tensor.

    Parameters
    ----------
    z : tensor
        Input.

    Returns
    -------
    tensor
        Output.

    """
    if torch.is_complex(z):
        return z.real
    else:
        return z


def imag(z):
    """
    Returns the imaginary part of pytorch tensor.

    Parameters
    ----------
    z : tensor
        Input.

    Returns
    -------
    tensor
        Output.

    """
    if torch.is_complex(z):
        return z.imag
    else:
        return torch.zeros_like(z)


def phase_harmonics(z, k):
    """
    Compute the phase harmonics of the input tensor.

    Parameters
    ----------
    z : tensor
        Input.
    k : tensor
        Exponents.

    Returns
    -------
    result : tensor
        Output.

    """
    indices_k_0 = torch.where(k == 0)[0]
    indices_other_k = torch.where(k >= 2)[0]
    
    result = z.clone()
    del z
    
    # k == 0
    result[..., indices_k_0, :, :] = torch.abs(torch.index_select(result, -3, indices_k_0)).to(result.dtype)
    
    # k == 1 is left unchanged
    
    # k >= 2
    other_k = k[indices_other_k].unsqueeze(-1).unsqueeze(-1)
    z_other_k = torch.index_select(result, -3, indices_other_k)
    r = torch.abs(z_other_k)
    theta = torch.angle(z_other_k)
    result[..., indices_other_k, :, :] = r * (torch.cos(other_k*theta) + 1j*torch.sin(other_k*theta))
    
    return result


def power_harmonics(z, k):
    """
    Compute the power harmonics of the input tensor.

    Parameters
    ----------
    z : tensor
        Input.
    k : tensor
        Exponents.

    Returns
    -------
    result : tensor
        Output.

    """
    indices_k_0 = torch.where(k == 0)[0]
    indices_other_k = torch.where(k >= 1)[0]
    
    result = z.clone()
    del z
    
    # k == 0
    result[..., indices_k_0, :, :] = torch.abs(torch.index_select(result, -3, indices_k_0)).to(result.dtype)

    # k >= 1
    other_k = k[indices_other_k].unsqueeze(-1).unsqueeze(-1)
    result[..., indices_other_k, :, :] = (torch.index_select(result, -3, indices_other_k)) ** other_k
    
    return result

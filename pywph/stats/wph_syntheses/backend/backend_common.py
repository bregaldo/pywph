import torch
#import torch.fft
import torch.nn as nn
from torch.nn import ReflectionPad2d
from torch.autograd import Function
import numpy as np
from torch.fft import fftn, ifftn


def iscomplex(input):
    return input.size(-1) == 2


def ones_like(z):
    re = torch.ones_like(z[..., 0])
    im = torch.zeros_like(z[..., 1])
    return torch.stack((re, im), dim=-1)


def real(z):
    return z[..., 0]


def imag(z):
    return z[..., 1]


def conjugate(z):
    z_copy = z.clone()
    z_copy[..., 1] = -z_copy[..., 1]
    return z_copy


def pows(z, max_k, dim=0):
    z_pows = [ones_like(z)]
    if max_k > 0:
        z_pows.append(z)
        z_acc = z
        for k in range(2, max_k + 1):
            z_acc = mul(z_acc, z)
            z_pows.append(z_acc)
    z_pows = torch.stack(z_pows, dim=dim)
    return z_pows


def mul(z1, z2):
    zr = real(z1) * real(z2) - imag(z1) * imag(z2)
    zi = real(z1) * imag(z2) + imag(z1) * real(z2)
    z = torch.stack((zr, zi), dim=-1)
    return z


def modulus_complex(input):
    norm = input.norm(p=2, dim=-1, keepdim=True)
    return torch.cat([norm, torch.zeros_like(norm)], -1)



def modulus(z):
    z_mod = z.norm(p=2, dim=-1)
    return z_mod


def inv(z):
    z_conj = conjugate(z)
    norm = modulus(z).unsqueeze(-1)
    return z_conj / (norm * norm)


# substract spatial mean (complex valued input)
class SubInitSpatialMeanC(object):
    def __init__(self, is_isotropic=False, scaling_version=False):
        self.minput = None
        self.is_isotropic = is_isotropic
        self.scaling_version = scaling_version

    def __call__(self, input, nbr_of_coeff=None, norm="auto"):
        """

        Args:
            input: (Nimg, L2, P_c, M, N, 2)

        Returns: (Nimg, L2, P_c, M, N, 2)

        """
        if norm == "auto":
            if self.minput is None:
                minput = input.clone().detach()
                minput = torch.mean(minput, -2, True)
                minput = torch.mean(minput, -3, True)
                #minput = torch.mean(minput, 0, True)
                if self.is_isotropic:
                    minput = torch.mean(minput, 1, True)
                if nbr_of_coeff is not None:
                    if self.scaling_version:
                        minput = minput * nbr_of_coeff.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    else:
                        minput = minput * nbr_of_coeff.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                self.minput = minput
                
            # minput = input.clone().detach()
            # minput = torch.mean(minput, -2, True)
            # minput = torch.mean(minput, -3, True)
            # minput = torch.mean(minput, 0, True)
            # if self.is_isotropic:
            #     minput = torch.mean(minput, 1, True)
            # if nbr_of_coeff is not None:
            #     if self.scaling_version:
            #         minput = minput * nbr_of_coeff.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            #     else:
            #         minput = minput * nbr_of_coeff.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            # self.minput = minput
    
            return input - self.minput
        elif norm == None: # No memory, just substract the mean of the input
            minput = input.clone().detach()
            minput = torch.mean(minput, -2, True)
            minput = torch.mean(minput, -3, True)
            #minput = torch.mean(minput, 0, True)
            if self.is_isotropic:
                minput = torch.mean(minput, 1, True)
            if nbr_of_coeff is not None:
                if self.scaling_version:
                    minput = minput * nbr_of_coeff.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                else:
                    minput = minput * nbr_of_coeff.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            return input - minput
        else:
            raise Exception(f"Unknown norm value: {norm}")


class DivInitStd(object):
    def __init__(self, is_isotropic=False, scaling_version=False):
        self.stdinput = None
        self.is_isotropic = is_isotropic
        self.scaling_version = scaling_version
        self.eps = 1e-8

    def __call__(self, input, nbr_of_coeff=None):
        """

        Args:
            input: (Nimg, L2, P_c, M, N, 2
        Returns: (Nimg, L2, P_c, M, N, 2)

        """
        if self.stdinput is None:
            stdinput = input.clone().detach()  # input size:(...,M,N,2)
            if self.is_isotropic:
                sqrt_d = np.sqrt(input.shape[-2]*input.shape[-3]*input.shape[1]*input.shape[0])
            elif nbr_of_coeff is not None:
                if self.scaling_version:
                    sqrt_d = torch.pow(nbr_of_coeff * input.shape[0], 0.5).unsqueeze(0).unsqueeze(-1).unsqueeze(
                        -1).unsqueeze(-1).unsqueeze(-1)
                else:
                    sqrt_d = torch.pow(nbr_of_coeff*input.shape[0], 0.5).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            else:
                sqrt_d = np.sqrt(input.shape[-2] * input.shape[-3])
            stdinput = torch.norm(stdinput, dim=-1, keepdim=True)
            if self.is_isotropic:
                stdinput = torch.norm(stdinput, dim=(-2, -3), keepdim=True)
                stdinput = torch.norm(stdinput, dim=1, keepdim=True)
                #stdinput = torch.norm(stdinput, dim=0, keepdim=True)
            else:
                stdinput = torch.norm(stdinput, dim=(-2, -3), keepdim=True)
                #stdinput = torch.norm(stdinput, dim=0, keepdim=True)
            self.stdinput = (stdinput + self.eps) / sqrt_d

        return input/self.stdinput


def fft(input, inverse=False):
    """
        Interface with torch FFT routines for 2D signals.

        Example
        -------
        x = torch.randn(128, 32, 32, 2)
        x_fft = fft(x, inverse=True)

        Parameters
        ----------
        input : tensor
            complex input for the FFT
        inverse : bool
            True for computing the inverse FFT.
            NB : if direction is equal to 'C2R', then the transform
            is automatically inverse.
    """

    if not iscomplex(input):
        raise(TypeError('The input should be complex (e.g. last dimension is 2)'))

    if (not input.is_contiguous()):
        raise (RuntimeError('Tensors must be contiguous!'))

    if inverse:
        output = ifftn(input[..., 0] + 1j*input[..., 1], s=(-1, -1))
        #output = torch.ifft(input, 2, normalized=False)
        #output = torch.fft.ifft(input, 2, norm= "forward")
    else:
        output = fftn(input[..., 0] + 1j*input[..., 1], s=(-1, -1))
        #output = torch.fft(input, 2, normalized=False)
        #output = torch.fft.fft(input, 2, norm= "forward")
    output = torch.stack((output.real, output.imag), dim=-1)
    return output


class PhaseHarmonics(Function):
    @staticmethod
    def forward(ctx, z, k):

        """
            z: (Nimg, L2, P_c, M, N, 2)
            k: (P_c)

        Returns:
            (Nimg, L2,P_c,M,N,2)

        """
        z = z.detach()  # (Nimg, L2, P_c, M, N, 2)

        indices_k_0 = torch.where(k == 0)[0]
        indices_other_k = torch.where(k >= 2)[0]
        other_k = k[indices_other_k]

        z_0 = z[:, :, indices_k_0, ...]
        z_other_k = z[:, :, indices_other_k, ...]

        x_0, y_0 = real(z_0), imag(z_0)  # (Nimg, L2, ??, M, N)  (Nimg, L2, ??, M, N)
        x, y = real(z_other_k), imag(z_other_k)  # (Nimg, L2, ??, M, N)  (Nimg, L2, ??, M, N)
        r = z_other_k.norm(p=2, dim=-1)  # (Nimg, L2, ??, M, N)
        theta = torch.atan2(y, x)  # (Nimg, L2, ??, M, N)
        ktheta = other_k.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * theta  # (Nimg, L2, ??, M, N)
        eiktheta = torch.stack((torch.cos(ktheta), torch.sin(ktheta)), dim=-1)  # # (Nimg, L2, ??, M, N,2)
        ctx.save_for_backward(x_0, y_0, x, y, k)

        result_other_k = r.unsqueeze(-1)*eiktheta  # (1,P_c,M,N, 2)

        result = z.clone()
        result[:, :, indices_k_0, ...] = modulus_complex(z_0)
        result[:, :, indices_other_k, ...] = result_other_k

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
            grad_output: (Nimg, L2,P_c,M,N,2)

        Returns: (Nimg, L2,P_c,M,N,2) , (P_c) (k gradient not used)

        """
        x_0, y_0, x, y, k = ctx.saved_tensors  # (1,P_c, M, N), (1,P_c, M, N), (1,P_c, 1, 1)

        indices_k_0 = torch.where(k == 0)[0]
        indices_k_1 = torch.where(k == 1)[0]
        indices_other_k = torch.where(k >= 2)[0]

        other_k = k[indices_other_k].unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        theta = torch.atan2(y, x)   # (Nimg, L2, ??, M, N)
        ktheta = other_k * theta  # (Nimg, L2, ??, M, N)
        cosktheta = torch.cos(ktheta)  # (Nimg, L2, ??, M, N)
        sinktheta = torch.sin(ktheta)  # (Nimg, L2, ??, M, N)
        costheta = torch.cos(theta)  # (Nimg, L2, ??, M, N)
        sintheta = torch.sin(theta)  # (Nimg, L2, ??, M, N)

        costheta_cosktheta = costheta * cosktheta  # (Nimg, L2, ??, M, N)
        sintheta_sinktheta = sintheta * sinktheta  # (Nimg, L2, ??, M, N)
        costheta_sinktheta = costheta * sinktheta  # (Nimg, L2, ??, M, N)
        sintheta_cosktheta = sintheta * cosktheta  # (Nimg, L2, ??, M, N)

        df1dx = costheta_cosktheta + other_k*sintheta_sinktheta  # (Nimg, L2, ??, M, N)
        df2dx = costheta_sinktheta - other_k*sintheta_cosktheta  # (Nimg, L2, ??, M, N)
        df1dy = sintheta_cosktheta - other_k*costheta_sinktheta  # (Nimg, L2, ??, M, N)
        df2dy = sintheta_sinktheta + other_k*costheta_cosktheta  # (Nimg, L2, ??, M, N)

        dx1_other_k = df1dx*grad_output[:,:, indices_other_k, :, :, 0] + df2dx*grad_output[:,:, indices_other_k, :, :, 1]  # (Nimg, L2, ??, M, N)
        dx2_other_k = df1dy*grad_output[:,:, indices_other_k, :, :, 0] + df2dy*grad_output[:,:, indices_other_k, :, :, 1]  # (Nimg, L2, ??, M, N)

        dx1_k_0 = torch.cos(torch.atan2(y_0, x_0)) * grad_output[:,:, indices_k_0, :, :, 0]  # (Nimg, L2, ??, M, N)
        dx2_k_0 = torch.sin(torch.atan2(y_0, x_0)) * grad_output[:,:, indices_k_0, :, :, 0]  # (Nimg, L2, ??, M, N)

        dx1_k_1 = grad_output[:, :, indices_k_1, :, :, 0]  # (Nimg, L2, ??, M, N)
        dx2_k_1 = grad_output[:, :, indices_k_1, :, :, 1]  # (Nimg, L2, ??, M, N)

        dx1 = grad_output.new_zeros((grad_output.size()[:5]))
        dx2 = grad_output.new_zeros((grad_output.size()[:5]))

        dx1[:, :, indices_k_0, :, :] = dx1_k_0
        dx1[:, :, indices_k_1, :, :] = dx1_k_1
        dx1[:, :, indices_other_k, :, :] = dx1_other_k

        dx2[:, :, indices_k_0, :, :] = dx2_k_0
        dx2[:, :, indices_k_1, :, :] = dx2_k_1
        dx2[:, :, indices_other_k, :, :] = dx2_other_k

        return torch.stack((dx1, dx2), -1), k  #  (Nimg, L2,P_c,M,N,2) , (P_c,) (k gradient not used)
    
class PowerHarmonics(Function):
    @staticmethod
    def forward(ctx, z, k):
        """
            z: (Nimg, L2, P_c, M, N, 2)
            k: (P_c)

        Returns:
            (Nimg, L2,P_c,M,N,2)

        """
        z = z.detach()  # (Nimg, L2, P_c, M, N, 2)
        k_unsqueeze = k.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x, y = real(z), imag(z)  # (Nimg, L2, P_c, M, N) (Nimg, L2, P_c, M, N)
        r = z.norm(p=2, dim=-1)  # (Nimg, L2, P_c, M, N)
        theta = torch.atan2(y, x)  # (Nimg, L2, P_c, M, N)
        ktheta = k_unsqueeze * theta  # (Nimg, L2, P_c, M, N)
        eiktheta = torch.stack((torch.cos(ktheta), torch.sin(ktheta)), dim=-1)  # (Nimg, L2, P_c, M, N)
        k_tilde = k_unsqueeze.clone()   # (1, 1, P_c, 1, 1)
        k_tilde[torch.where(k_unsqueeze == 0)] = 1   # (1, 1, P_c, 1, 1)
        ctx.save_for_backward(x, y, r, k_unsqueeze, k_tilde)
        return torch.pow(r, k_tilde).unsqueeze(-1)*eiktheta   #  (Nimg, L2, P_c, M, M, 2)

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: (Nimg, L2,P_c,M,N,2)

        Returns: (Nimg, L2,P_c,M,N,2) , (P_c) (k gradient not used)

        """
        x, y, r, k_unsqueeze, k_tilde = ctx.saved_tensors  # (Nimg, L2, P_c, M, M), (Nimg, L2, P_c, M, M), (1, 1, P_c, 1, 1), (1, 1, P_c, 1, 1)
        theta = torch.atan2(y, x)  # (Nimg, L2, P_c, M, N)
        ktheta = k_unsqueeze * theta  # (Nimg, L2, P_c, M, N)
        cosktheta = torch.cos(ktheta)  # (Nimg, L2, P_c, M, N)
        sinktheta = torch.sin(ktheta)  # (Nimg, L2, P_c, M, N)
        costheta = torch.cos(theta)  # (Nimg, L2, P_c, M, N)
        sintheta = torch.sin(theta)  # (Nimg, L2, P_c, M, N)

        costheta_cosktheta = costheta * cosktheta  # (Nimg, L2, P_c, M, N)
        sintheta_sinktheta = sintheta * sinktheta  # (Nimg, L2, P_c, M, N)
        costheta_sinktheta = costheta * sinktheta  # (Nimg, L2, P_c, M, N)
        sintheta_cosktheta = sintheta * cosktheta  # (Nimg, L2, P_c, M, N)

        r_k_tilde_minus_1 = torch.pow(r, k_tilde-1)

        df1dx = r_k_tilde_minus_1*(k_tilde*costheta_cosktheta + k_unsqueeze*sintheta_sinktheta)  # (Nimg, L2, P_c, M, N)
        df2dx = r_k_tilde_minus_1*(k_tilde*costheta_sinktheta - k_unsqueeze*sintheta_cosktheta)  # (Nimg, L2, P_c, M, N)
        df1dy = r_k_tilde_minus_1*(k_tilde*sintheta_cosktheta - k_unsqueeze*costheta_sinktheta)  # (Nimg, L2, P_c, M, N)
        df2dy = r_k_tilde_minus_1*(k_tilde*sintheta_sinktheta + k_unsqueeze*costheta_cosktheta)  # (Nimg, L2, P_c, M, N)

        dx1 = df1dx*grad_output[..., 0] + df2dx*grad_output[..., 1]  # (Nimg, L2, P_c, M, N)
        dx2 = df1dy*grad_output[..., 0] + df2dy*grad_output[..., 1]  # (Nimg, L2, P_c, M, N)

        return torch.stack((dx1, dx2), -1), k_unsqueeze[0,0,:,0,0]  #  (1,P_c, M, N, 2) , (P_c,) (k gradient not used)

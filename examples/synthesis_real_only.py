# -*- coding: utf-8 -*-

import numpy as np
import time
import torch
import scipy.optimize as opt
import pywph as pw

#######
# INPUT PARAMETERS
#######

M, N = 256, 256
J = 7
L = 7
dn = 2

norm = "auto"   # Normalization
pbc = True      # Periodic boundary conditions

device = 0

optim_params_joint = {"maxiter": 400, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

data = np.load('../sim_QiU_stdr_3_256.npy').real

output_filename = "synthesis_real_only_256_moments6.npy"

#######
# PREPARATION AND INITIAL GUESS
#######

# Normalize input data
data_std = data.std()
data_mean = np.mean(data)
data -= data_mean
data /= data_std

# Initial guess
x0 = np.zeros((M, N, 1), dtype=np.float64)
x0[:, :, 0] = np.random.normal(data.real.mean(), data.real.std(), data.shape)

print("Building operator...")
start_time = time.time()
wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=device)
wph_op.load_model(p_list=[0, 1, 2, 3, 4])
print(f"Done! (in {time.time() - start_time}s)")

print("Computing stats of target image...")
start_time = time.time()
coeffs = wph_op.apply(data, norm=norm, padding=not pbc)
print(f"Done! (in {time.time() - start_time}s)")


#######
# SYNTHESIS
#######

eval_cnt = 0

def objective_real(x):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    
    x_curr = x.reshape((M, N))
    
    # Compute the loss (squared 2-norm)
    loss_tot = torch.zeros(1)

    # Loss for real image alone
    x_curr, nb_chunks = wph_op.preconfigure(x_curr, requires_grad=True)
    for i in range(nb_chunks):

        coeffs_chunk, indices = wph_op.apply(x_curr, i, norm=norm,
                                                  padding=not pbc, ret_indices=True)
        loss = torch.sum(torch.abs(coeffs_chunk - coeffs[indices]) ** 2)

        loss.backward(retain_graph=True)
        loss_tot += loss.detach().cpu()
        del coeffs_chunk, indices, loss
    
    # Reshape the gradient
    x_grad = np.zeros_like(x).reshape((M, N))
    x_grad[:, :] = x_curr.grad.cpu().numpy()
    
    print(f"Loss: {loss_tot.item()} (computed in {time.time() - start_time}s)")
    eval_cnt += 1
    return loss_tot.item(), x_grad.ravel()



total_start_time = time.time()
result = opt.minimize(objective_real, x0.ravel(), method='L-BFGS-B',
                      jac=True, tol=None, options=optim_params_joint)
final_loss, x_final, niter, msg = result['fun'], result['x'], result['nit'], result['message']
print(f"Synthesis joint constraint ended in {niter} iterations with optimizer message: {msg}")
print(f"Synthesis time: {time.time() - total_start_time}s")

#######
# OUTPUT
#######

x_final = x_final.reshape((M, N, 1)).astype(np.float32)
x_final = x_final[..., 0] 

x_final = x_final  * data_std + data_mean

if output_filename is not None:
    np.save(output_filename, x_final)

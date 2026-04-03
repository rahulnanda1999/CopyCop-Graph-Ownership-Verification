import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

import nevergrad as ng

from functools import partial
import sys
import psutil, os

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Compute gradient of model embedding H w.r.t. input x at target_node, projected along coeffs_x direction
def calc_grad_x(x, coo, model, target_node, coeffs_x=None, do_grad_of_normed = False):
  out = model.get_H(Data(my_x=x, edge_index=coo))[target_node]
  out_normed = out / (1e-9 + torch.norm(out))
  grad = []
  for j in range(len(out_normed)):
    if do_grad_of_normed:
      grad_j = (torch.autograd.grad(out[j], x, create_graph=True)[0][target_node]).detach() / (torch.norm(out))
    else:
      grad_j = torch.autograd.grad(out[j], x, create_graph=True)[0][target_node].detach()
    grad.append(grad_j)
  grad = torch.stack(grad)
  if coeffs_x is not None:
    grad = torch.sum(grad * coeffs_x[None, :], dim=1)
  return out, out_normed, grad

# Same as calc_grad_x but using finite differences (for integer-valued features)
def calc_grad_x_int(x, coo, model, target_node, coeffs_x, mesh=2):
  assert coeffs_x is not None

  with torch.no_grad():
    tmp = torch.zeros_like(x)
    tmp[target_node] = coeffs_x

    Z = []
    out_orig = model.get_H(Data(my_x=x, edge_index=coo))[target_node].detach()
    out_orig_normed = (out_orig / (1e-9 + torch.norm(out_orig))).detach()
    for i, scale in enumerate(np.arange(-mesh, mesh+1)):
      out = model.get_H(Data(my_x=x+scale*tmp, edge_index=coo))[target_node].detach()
      Z.append(out)
    Z = torch.stack(Z)
    grad = ((Z[mesh+1] - Z[mesh])/torch.norm(Z[mesh])).detach()
    del Z, tmp
  return out_orig, out_orig_normed, grad


# Loss = squared gradient norm + optional regularization; minimized to find stationary points
def our_loss(x, coo, model, target_node, reg_opts={'lambda_reg':0, 'x0':None, 'reg_type':'l1'}, verbose=0, coeffs_x=None, is_int=False, **kwargs):

  calc_grad_fun = calc_grad_x_int if is_int else calc_grad_x
  _, _, grad = calc_grad_fun(x, coo, model, target_node, coeffs_x, **kwargs)

  loss_grad = (torch.sum(grad**2)).detach()
  loss_reg = 0
  if reg_opts['lambda_reg'] > 0:
    assert reg_opts['x0'] is not None
    if reg_opts['reg_type'] == 'l1':
      loss_reg = torch.abs(torch.sum(torch.abs(x)) - torch.sum(torch.abs(reg_opts['x0'])))
    elif reg_opts['reg_type'] == 'l2':
      loss_reg = torch.linalg.norm(x-reg_opts['x0']) / torch.linalg.norm(reg_opts['x0']) 
    else:
      print(f'Unknown reg_type={reg_opts["reg_type"]}')
      sys.exit(1)
    this_loss = loss_grad + reg_opts['lambda_reg'] * loss_reg
    if verbose:
      print(f'loss_grad={loss_grad.item():.4f}, loss_reg={loss_reg.item():.4f}, this_loss={this_loss:.4f}')
  else:
    this_loss = loss_grad
    if verbose:
      print(f'this_loss={this_loss:.4f}')
  return this_loss

# Wrapper for our_loss compatible with Nevergrad (numpy in, scalar out)
def our_loss_ng(x, device, overall_mask, reg_opts, **kwargs):
  if overall_mask is not None:
    this_x = reg_opts['x0'].clone().detach().cpu().numpy()
    this_x[overall_mask] = np.ravel(x)
  else:
    this_x = x
  this_x = torch.tensor(this_x, dtype=torch.float32).requires_grad_(True).to(device)
  loss = our_loss(this_x, reg_opts=reg_opts, **kwargs).detach().item()
  del this_x
  return loss

# Find input x that minimizes gradient norm using Nevergrad black-box optimizer
def optimize_x_with_ng(x0, coo, model, target_node, opt_iterations=5000, lambda_reg=0, coeffs_x=None, reg_type='l1', verbose=2, 
                       is_int=False, device='cpu', ng_sigma=None, coeffs_x_mask=None, node_mask=None, do_grad_of_normed=False): 
  
  assert coeffs_x is not None 
  assert coeffs_x_mask is None or torch.linalg.norm(coeffs_x[~coeffs_x_mask])<1e-5
  model.eval()
  model = model.to(device)
  x0 = x0.to(device); coo = coo.to(device)

  if coeffs_x_mask is None and node_mask is None:
    overall_mask = None
    x0masked = x0
  else:
    if coeffs_x_mask is None:
      coeffs_x_mask = torch.ones_like(x0[0], dtype=torch.bool)
    if node_mask is None:
      node_mask = torch.ones_like(x0[:,0], dtype=torch.bool)
    overall_mask = node_mask[:,None] & coeffs_x_mask[None,:]
    x0masked = x0[overall_mask]
    overall_mask = overall_mask.detach().numpy()

  this_loss = partial(our_loss_ng, coo=coo, model=model, target_node=target_node,
                      reg_opts={'lambda_reg':lambda_reg, 'x0':x0, 'reg_type':reg_type},
                      coeffs_x=coeffs_x,
                      overall_mask=overall_mask,
                      is_int=is_int,
                      device=device,
                      do_grad_of_normed=do_grad_of_normed
                      )

  x_arr = ng.p.Array(init=x0masked if isinstance(x0masked, np.ndarray) else x0masked.detach().clone().cpu().numpy())
  if is_int:
    lower, upper = torch.min(x0masked, dim=0).values.detach().cpu().numpy().astype(int), torch.max(x0masked, dim=0).values.detach().cpu().numpy().astype(int)
    upper = np.maximum(lower+1, upper)
    lower = np.tile(lower, x0masked.shape[0]).reshape(x0masked.shape)
    upper = np.tile(upper, x0masked.shape[0]).reshape(x0masked.shape)
    x_arr = x_arr.set_bounds(lower=lower, upper=upper).set_integer_casting()
  if ng_sigma:
    x_arr.set_mutation(sigma=ng_sigma)
  instrum = ng.p.Instrumentation(x=x_arr)  
  optimizer = ng.optimizers.TBPSA(parametrization=instrum, budget=opt_iterations)


  best_x, best_loss = None, 1e99
  for i in range(optimizer.budget):
    x = optimizer.ask()
    loss = this_loss(*x.args, **x.kwargs)
    if loss < best_loss:
      best_x, best_loss = x.value[1]['x'].copy(), loss
    optimizer.tell(x, loss)
    del x
    if (verbose>0 and i==opt_iterations-1) or (verbose>1 and i%200 == 0):
      print(f"Iteration {i+1:2d}: best_loss = {best_loss:.4f} mem={psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024}")
    if loss < 1e-5:
      break

  del optimizer, instrum

  if verbose >= 0:
    _=this_loss(best_x, device=device, verbose=True)
  if overall_mask is None:
    best_x_full = best_x
  else:
    best_x_full = x0.clone().detach().cpu().numpy()
    best_x_full[overall_mask] = np.ravel(best_x)
 
  return best_x_full

# Find largest delta along coeffs_x direction where embedding change stays below max_delta_h_zero
def linesearch(x_zerograd, coo, target_node, coeffs_x, model, init_delta_x=0.01, num_iter=10, max_delta_h_zero=0.05, verbose=False):
  if type(x_zerograd) != torch.Tensor:
    x_zerograd = torch.Tensor(x_zerograd)
  tmp = torch.zeros_like(x_zerograd)
  tmp[target_node] = torch.Tensor(coeffs_x)
  model.eval()

  lo_H = model.get_H(Data(my_x=x_zerograd, edge_index=coo))[target_node]
  lo_norm = torch.norm(lo_H)
  delta_x_range, delta_x, prev_delta_x, hi = init_delta_x, init_delta_x, 0, None
  for i in range(num_iter):
    lo_plus = model.get_H(Data(my_x=x_zerograd+delta_x*tmp, edge_index=coo))[target_node]
    lo_minus = model.get_H(Data(my_x=x_zerograd-delta_x*tmp, edge_index=coo))[target_node]
    delta_h_zero = (torch.norm(lo_plus-lo_minus) / (lo_norm + 1e-9)).detach().cpu().numpy()
    if verbose:
      print(f'delta_x={delta_x:.4f}, delta_x_range={delta_x_range:.4f}, delta_h_zero={delta_h_zero:.4f}')
    if delta_h_zero < max_delta_h_zero:
      prev_delta_x = delta_x
      delta_x += delta_x_range
      if hi is None:
        delta_x_range *= 2
      else:
        delta_x_range = (hi - delta_x) / 2

    else:
      hi = delta_x
      delta_x = (prev_delta_x + delta_x)/2
      delta_x_range = (delta_x - prev_delta_x)/4
  return delta_x

# Same as linesearch but for integer features, expanding symmetrically in discrete steps
def linesearch_int(x_zerograd, coo, target_node, coeffs_x, model, num_iter=10, max_delta_h_zero=0.05, verbose=False):
  if type(x_zerograd) != torch.Tensor:
    x_zerograd = torch.Tensor(x_zerograd)
  tmp = torch.zeros_like(x_zerograd)
  tmp[target_node] = torch.Tensor(coeffs_x)
  model.eval()
  
  mid = model.get_H(Data(my_x=x_zerograd, edge_index=coo))[target_node]
  mid_norm = torch.norm(mid)

  curr_delta_lo, curr_delta_hi, lo, hi = 0, 0, mid, mid
  for _ in range(num_iter):
    dec_lo = model.get_H(Data(my_x=x_zerograd-(curr_delta_lo+1)*tmp, edge_index=coo))[target_node]
    inc_hi = model.get_H(Data(my_x=x_zerograd+(curr_delta_hi+1)*tmp, edge_index=coo))[target_node]
    delta_h_dec_lo = (torch.norm(dec_lo-hi) / (mid_norm + 1e-9)).detach().cpu().numpy()
    delta_h_inc_hi = (torch.norm(inc_hi-lo) / (mid_norm + 1e-9)).detach().cpu().numpy()
    delta_h_both = (torch.norm(dec_lo-inc_hi) / (mid_norm + 1e-9)).detach().cpu().numpy()
    this_delta_h = min(delta_h_dec_lo, delta_h_inc_hi, delta_h_both)
    if this_delta_h == delta_h_dec_lo:
      next_delta_lo, next_delta_hi = curr_delta_lo+1, curr_delta_hi
    elif this_delta_h == delta_h_inc_hi:
      next_delta_lo, next_delta_hi = curr_delta_lo, curr_delta_hi+1
    else:
      next_delta_lo, next_delta_hi = curr_delta_lo+1, curr_delta_hi+1

    if verbose:
      print(f'curr_delta_lo={curr_delta_lo}, curr_delta_hi={curr_delta_hi} delta_h_dec_lo={delta_h_dec_lo:.4f} delta_h_inc_hi={delta_h_inc_hi:.4f} delta_h_both={delta_h_both:.4f}')
    if this_delta_h > max_delta_h_zero:
      if curr_delta_lo == 0 and curr_delta_hi == 0:
        curr_delta_lo, curr_delta_hi = next_delta_lo, next_delta_hi
      break
    else:
      curr_delta_lo, curr_delta_hi = next_delta_lo, next_delta_hi

  return curr_delta_lo, curr_delta_hi




import os
os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import torch

from torch_geometric.data import Data
from pandas import Series, DataFrame
import pandas as pd
import concurrent.futures
import multiprocessing as mp
from functools import partial
import scipy.stats
from sklearn.metrics import roc_auc_score
import pickle
import time

import models
import get_data
import min_grad

import sys
import psutil, os
import gc

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None, 'display.width', 1000, 'display.max_rows', None)
pd.set_option('future.no_silent_downcasting', True)

# Get embedding and its gradient at target_node for a given input x
def get_h_and_grad(x, model, e, target_node, is_int, coeffs_x, do_grad_of_normed,):
  this_x = x.detach().clone().requires_grad_(True) if isinstance(x, torch.Tensor) else torch.Tensor(x, dtype=torch.float32).requires_grad_(True)
  calc_grad_fun = min_grad.calc_grad_x_int if is_int else min_grad.calc_grad_x
  tmp = calc_grad_fun(x=this_x, coo=e, model=model, target_node=target_node, coeffs_x=coeffs_x, do_grad_of_normed=do_grad_of_normed)
  h, gr = tmp[0], tmp[2]
  return h, gr

# Generate a random unit-norm direction in feature space
def create_coeffs_x(x0, is_int, coeffs_x_mask=None, random_coeffs=True):
  coeffs_x_str = ""
  assert random_coeffs 
  if random_coeffs:
    if is_int:
      which_feature = np.random.choice(x0.shape[1])
      coeffs_x = np.zeros(x0.shape[1])
      coeffs_x[which_feature] = 1
      coeffs_x_str = f" (coeffs_x=feature {which_feature})"
    else:
      coeffs_x = np.random.randn(x0.shape[1])
      coeffs_x /= np.linalg.norm(coeffs_x)
  else:
    pass

  if coeffs_x_mask is not None: 
    coeffs_x[~coeffs_x_mask] = 0
    coeffs_x /= np.linalg.norm(coeffs_x)

  return torch.tensor(coeffs_x, dtype=torch.float32), coeffs_x_str

# For one target_node: generate random direction, check gradient is nontrivial, then optimize to find stationary point
def get_lo_hi_points_helper(target_node, x0, e, model_to_mimic, do_grad_of_normed, device, random_coeffs, opt_iterations, is_int, ng_sigma=None, norm_along_dir_threshold=0.05, coeffs_x_mask=None, lambda_reg=0.01, return_times=False):
  time1 = time.time()
  coeffs_x, coeffs_x_str = create_coeffs_x(x0=x0, is_int=is_int, coeffs_x_mask=coeffs_x_mask, random_coeffs=random_coeffs)
  x0 = x0.to(device); e = e.to(device); coeffs_x = coeffs_x.to(device)

  time2 = time.time()
  h, gr = get_h_and_grad(x=x0, model=model_to_mimic, e=e, target_node=target_node, is_int=is_int, coeffs_x=coeffs_x, do_grad_of_normed=do_grad_of_normed)
  norm_along_this_dir = torch.norm(gr)
  if torch.isnan(norm_along_this_dir) or norm_along_this_dir < norm_along_dir_threshold or norm_along_this_dir > 100: # drop weird points # added the isnan
    print(f'Skipping target_node={target_node} out of {len(x0)}; norm_along_this_dir={norm_along_this_dir:.4f}')
    del h, gr, coeffs_x
    torch.cuda.empty_cache()
    return None

  print(f'target_node={target_node} out of {len(x0)};{coeffs_x_str} norm_along_this_dir={norm_along_this_dir:.4f}, mem={psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024}')
  time3 = time.time()
  res = min_grad.optimize_x_with_ng(x0=x0.detach().clone(), coo=e, model=model_to_mimic, target_node=target_node, lambda_reg=lambda_reg, coeffs_x=coeffs_x, opt_iterations=opt_iterations, reg_type='l2', verbose=-1, device=device, is_int=is_int, ng_sigma=ng_sigma, coeffs_x_mask=coeffs_x_mask, do_grad_of_normed=do_grad_of_normed)
  time4 = time.time()

  res_reverse = None

  coeffs_x = coeffs_x.cpu()
  del h, gr
  torch.cuda.empty_cache()
  time5 = time.time()

  result = {'coeffs_x':coeffs_x, 'lo':res, 'hi':res_reverse}
  all_times = Series([time2-time1, time3-time2, time4-time3, time5-time4, time5-time1], index=['create coeffs', 'check norm', 'find stat pt', 'clean up', 'total'])
  if return_times:
    return result, all_times
  else:
    return result


def get_lo_hi_helper_collect_pts(results, idx, target_nodes):
  all_pts = {}
  for i, this_pts in enumerate(results):
    if this_pts:
      target_node = target_nodes[i]
      if idx not in all_pts:
        all_pts[idx] = {}
      all_pts[idx][target_node] = this_pts
    else:
      pass
  return all_pts

def get_lo_hi_points(idx, dataset, model_to_mimic, opt_iterations=1000, random_coeffs=True, seed=0, device='cuda', max_workers=0, is_int=False, ng_sigma=None, norm_along_dir_threshold=0.05,
                     coeffs_x_mask=None, lambda_reg=0.01, max_targets_from_one_idx=7, do_grad_of_normed=False):
  model_to_mimic = model_to_mimic.to(device)
  model_to_mimic.eval()

  np.random.seed(seed)
  torch.manual_seed(seed)

  x0, e = dataset[idx].my_x, dataset[idx].edge_index

  fun = partial(get_lo_hi_points_helper, x0=x0, e=e, model_to_mimic=model_to_mimic, device=device, do_grad_of_normed=do_grad_of_normed, random_coeffs=random_coeffs,
                opt_iterations=opt_iterations, is_int=is_int, ng_sigma=ng_sigma, norm_along_dir_threshold=norm_along_dir_threshold, coeffs_x_mask=coeffs_x_mask,
                lambda_reg=lambda_reg)

  target_nodes = range(len(x0)) if len(x0)<=max_targets_from_one_idx else np.random.choice(len(x0), max_targets_from_one_idx, replace=False)
  if max_workers == 0:
    results = map(fun, target_nodes)
    all_pts = get_lo_hi_helper_collect_pts(results, idx, target_nodes)
  else:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
      results = pool.map(fun, target_nodes)
    all_pts = get_lo_hi_helper_collect_pts(results, idx, target_nodes)

  return all_pts

# For each stationary point, find the perturbation range where embedding barely changes (via linesearch)
def attach_delta_x(dataset, model_to_mimic, all_pts, device, max_delta_h_zero=0.05, is_int=False):
  model_to_mimic = model_to_mimic.to(device)
  for idx in all_pts.keys():
    x0, e = dataset[idx].my_x, dataset[idx].edge_index
    x0 = x0.to(device)
    e = e.to(device)
    for target_node in all_pts[idx].keys():
      z = all_pts[idx][target_node]
      x_zerograd, x_orig = torch.tensor(z['lo'], dtype=torch.float32, device=device), x0.detach().clone()
      if not is_int:
        delta_x = min_grad.linesearch(x_zerograd=x_zerograd, coo=e, target_node=target_node, coeffs_x=z['coeffs_x'].to(device), model=model_to_mimic, max_delta_h_zero=max_delta_h_zero)
        all_pts[idx][target_node]['delta_x'] = (delta_x, delta_x)
      else:
        delta_lo, delta_hi = min_grad.linesearch_int(x_zerograd=x_zerograd, coo=e, target_node=target_node, coeffs_x=z['coeffs_x'].to(device), model=model_to_mimic, max_delta_h_zero=max_delta_h_zero)
        all_pts[idx][target_node]['delta_x'] = (delta_lo, delta_hi)
  return all_pts


# Generate stationary points across multiple graphs until num_stat_pts_needed are found
def create_all_pts(dataset, model_to_mimic, data_index_ids=np.arange(10), device='cuda', max_workers=8, is_int=False, num_stat_pts_needed=100, **kwargs):
  np.random.seed(0)
  data_index = np.random.permutation(len(dataset))
  fun = partial(get_lo_hi_points, dataset=dataset, model_to_mimic=model_to_mimic, device=device, max_workers=max_workers, is_int=is_int, **kwargs)

  start_time = time.time()
  all_pts = {}
  for i, d in enumerate(data_index[data_index_ids]):
    print(f'==> i={i}, d={d}')
    all_pts.update(fun(d))
    gc.collect()
    tot_stat_pts = sum([len(all_pts[idx]) for idx in all_pts.keys()])
    print(f'So far, tot_stat_pts={tot_stat_pts}\n')
    if tot_stat_pts >= num_stat_pts_needed:
      break
  print(f'TIME per stationary point: {(time.time() - start_time) / len(all_pts):1.1f}s') 

  all_pts = attach_delta_x(dataset, model_to_mimic, all_pts, device=device, is_int=is_int)
  return all_pts

# Evaluate one (idx, target_node) pair: compute gradient and delta metrics at both the stationary point and original input for all models
def model_perf_helper(idx_and_target_node_and_x0_and_e, all_pts, list_of_models_and_names, device, is_int, do_grad_of_normed, use_delta_x=None):
  idx, target_node, x0, e = idx_and_target_node_and_x0_and_e
  z = all_pts[idx][target_node]

  x_zerograd, x_orig = torch.tensor(z['lo'], dtype=torch.float32, device=device), x0.detach().clone()
  coeffs_x = all_pts[idx][target_node]['coeffs_x']
  if isinstance(coeffs_x, torch.Tensor):
    coeffs_x = coeffs_x.to(device)
  else:
    coeffs_x = torch.tensor(coeffs_x, dtype=torch.float32, device=device)
  if use_delta_x is None:
    delta_lo, delta_hi = all_pts[idx][target_node]['delta_x'] 
  else:
    delta_lo, delta_hi = use_delta_x, use_delta_x
  tmp = torch.zeros_like(x_zerograd)
  tmp[target_node] = torch.Tensor(coeffs_x)

  results = {}
  for model, model_name in list_of_models_and_names:
    results[model_name] = {}
    for this_point_name, this_point in [('lo', x_zerograd), ('orig', x_orig)]:
      h, gr = get_h_and_grad(this_point, model, e=e, target_node=target_node, is_int=is_int, coeffs_x=coeffs_x, do_grad_of_normed=do_grad_of_normed)
      results[model_name][f'grad_{this_point_name}'] = torch.norm(gr).item()
      results[model_name][f'h_{this_point_name}'] = torch.norm(h).item()
      
      t1 = model.get_H(Data(my_x=this_point+delta_hi*tmp, edge_index=e))[target_node]
      t2 = model.get_H(Data(my_x=this_point-delta_lo*tmp, edge_index=e))[target_node]
      results[model_name][f'delta_{this_point_name}'] = (torch.norm(t1-t2)).item() #detach().cpu().numpy()
      results[model_name][f'grad_from_delta_{this_point_name}'] = (torch.norm(t1-t2) / (delta_hi+delta_lo)).item() #detach().cpu().numpy()

      if results[model_name][f'h_{this_point_name}'] == 0:
        for blah in ['norm_delta', 'norm_grad']:
          results[model_name][f'{blah}_{this_point_name}'] = np.nan
      else:
        results[model_name][f'norm_grad_{this_point_name}'] = results[model_name][f'grad_{this_point_name}'] / (results[model_name][f'h_{this_point_name}'])
        results[model_name][f'norm_delta_{this_point_name}'] = results[model_name][f'delta_{this_point_name}'] / (results[model_name][f'h_{this_point_name}'])

  this_pretty_df = DataFrame(results)
  return this_pretty_df

# Evaluate all models on all stationary points, returning per-point metric DataFrames
def model_perf(list_of_models_and_names, dataset, all_pts, is_int, do_grad_of_normed, max_workers=8, device='cpu', use_delta_x=None, verbose=True):
  all_settings = []
  for idx in all_pts.keys():
    x0, e = dataset[idx].my_x, dataset[idx].edge_index
    x0 = x0.to(device)
    e = e.to(device)
    for target_node in all_pts[idx].keys():
      all_settings.append((idx, target_node, x0, e))

  for model, model_name in list_of_models_and_names:
    model = model.to(device); model.eval()

  fun = partial(model_perf_helper, all_pts=all_pts, list_of_models_and_names=list_of_models_and_names, device=device, use_delta_x=use_delta_x, is_int=is_int, do_grad_of_normed=do_grad_of_normed)
  if max_workers == 0:
    res = map(fun, all_settings)
  else:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
      res = pool.map(fun, all_settings)

  res = list(res)
  all_results = {}
  counter = -1
  for idx in all_pts.keys():
    all_results[idx] = {}
    for target_node in all_pts[idx].keys():
      counter += 1
      all_results[idx][target_node] = res[counter]

  if verbose:
    pretty_print_results(all_results, verbose=True)
  return all_results
  
def get_all_model_mimicking_losses(loader, base_model, list_of_models_and_names):
  losses = {}
  for m, nm in list_of_models_and_names:
    if m!=base_model:
      losses[nm] = models.val_loss(m, loader, model_to_mimic=base_model).item()
  return losses

# where does each model's stationary-point metric rank among its original-point metrics
def pscores(s, aspect='norm_grad'):
  s1 = s[f'{aspect}_orig'].replace(np.inf, np.nan).dropna()
  s2 = s[f'{aspect}_lo'].replace(np.inf, np.nan).dropna()
  z = Series(np.searchsorted(np.sort(s1), s2)/len(s1)*100)
  return z


def bootstrap_anyf(tdf, fun, bootstraps=100):
  res = []
  for i in range(bootstraps):
    this_tdf = tdf.iloc[np.random.choice(len(tdf), size=len(tdf))]
    res.append(fun(this_tdf))
  res = pd.concat(res, axis=1).T
  return res.std()


def get_full_df(all_results, base_model_name=None):
  df = []
  for idx in all_results:
    for target_node in all_results[idx]:
      thisdf = all_results[idx][target_node]
      thisdf['idx'] = idx
      thisdf['target_node'] = target_node
      thisdf = thisdf.reset_index()
      df.append(thisdf)
  df = pd.concat(df, axis=0)
  if base_model_name is not None:
    cols = [x for x in df.columns.values if '_bm_' not in x and '_steal_' not in x and '_boot' not in x] + [base_model_name] + [x for x in df.columns.values if '_steal_' in x] + [x for x in df.columns.values if '_bm_' in x and x!=base_model_name] + [x for x in df.columns.values if '_boot' in x]
    df = df.reindex(cols, axis=1)
  df = df.set_index(['idx', 'target_node', 'index']).unstack('index').astype(float).swaplevel(0,1,axis=1)#.sort_index(axis=1)
  return df


def check_if_reasonable_for_basemodel(all_results, base_model_name):
  tdf = get_full_df(all_results)
  print('Check that deltas approx 0.05: tdf["norm_delta_lo"].describe()')
  print(DataFrame(tdf['norm_delta_lo'].describe()[base_model_name]).T.round(3))
  print()

# Summarize results 
def pretty_print_results(all_results, base_model_name=None, aspect='norm_delta', verbose=True):
  if len(all_results) == 0:
    print('NO RESULTS!')
    return None

  tmp1 = list(all_results.keys())[0]
  tmp2 = list(all_results[tmp1].keys())[0]
  tdf = get_full_df(all_results, base_model_name=base_model_name)

  z = (tdf[f'{aspect}_lo'] / tdf[f'{aspect}_orig'].median())
  df_desc = z.describe()
  if verbose:
    print(f"Using measure = (tdf['{aspect}_lo'] / tdf['{aspect}_orig'].median()):")
    print(df_desc.round(2))
    print()

  fun = lambda tdf: (tdf[f'{aspect}_lo'] / tdf[f'{aspect}_orig'].median()).apply(lambda s: scipy.stats.trim_mean(s.dropna(), proportiontocut=0.1))
  df_tm = DataFrame({'Trim mean':fun(tdf), 'std(bootstrap)':bootstrap_anyf(tdf, fun)}).round(3).T
  if verbose:
    print('Trim mean of measure')
    print(df_tm)
    print()
  

  fun = lambda tdf: tdf.stack(0).droplevel(['idx', 'target_node'], axis=0).apply(lambda s: pscores(s, aspect=aspect)).apply(lambda s: scipy.stats.trim_mean(s.dropna(), proportiontocut=0.1))
  df_psc = DataFrame({'Trim mean':fun(tdf), 'std(bootstrap)':bootstrap_anyf(tdf, fun)}).round(3).T
  if verbose:
    print('Trim mean of pscores(measure)')
    print(df_psc)
    print()

  if not verbose:
    return df_desc, df_tm, df_psc

# Full pipeline: train base models, create stationary points, train stolen models, evaluate all, save results
def do_all(dataset, dataset_name, split_size, dim_h, dim_out, bm_loss_type, bm_end_target, epochs=500, num_graphs_for_stat_pts=100, standardize=False, norm_along_dir_threshold=0.05, bm_lr=1e-4, steal_lr=1e-3,
           coeffs_x_mask=None, steal_num_repeats=4, is_int=False, ng_sigma=0.1, only_basic_tests=False, lambda_reg=0.01, batch_size=64, max_workers=0, max_targets_from_one_idx=5, do_grad_of_normed = True, **kwargs):

  model_types = ['GIN',
                 'GCN',
                 'GSAGE',
                 'MIX',
                 'ARMA',
                 ]

  steal_model_types = model_types
  train_loader1, val_loader1 = get_data.setup_data(dataset, n_max=split_size, seed=0, start_idx=0, standardize=standardize, batch_size=batch_size)
  train_loader2, val_loader2 = get_data.setup_data(dataset, n_max=split_size, seed=0, start_idx=split_size, standardize=standardize, batch_size=batch_size)
  if steal_num_repeats >= 1:
    train_loader1_synth, val_loader1_synth = get_data.create_synthetic_data(dataset, start_idx=0, n_max=split_size, seed=0, num_repeats=steal_num_repeats, batch_size=batch_size)
  else:
    train_loader1_synth, val_loader1_synth = train_loader1, val_loader1
  if not only_basic_tests:
    train_loader1_boot, val_loader1_boot = get_data.setup_data(dataset, n_max=split_size, seed=0, start_idx=0, standardize=standardize, do_bootstrap=True, batch_size=batch_size)
  if steal_num_repeats >= 1:
    train_loader2_synth, val_loader2_synth = get_data.create_synthetic_data(dataset, start_idx=split_size, n_max=split_size, seed=0, num_repeats=steal_num_repeats, batch_size=batch_size)
  else:
    train_loader2_synth, val_loader2_synth = train_loader2, val_loader2

  # Train all models
  all_models = {}
  all_stat_pts = {}
  for model_type in model_types:
    list_models = [(train_loader1, val_loader1, 1, None), (train_loader2, val_loader2, 2, None) ]
    if not only_basic_tests:
      list_models += [(train_loader1_boot, val_loader1_boot, 1, 0)]
    for train_loader, val_loader, dataloader_split, bootstrap_num in list_models:
      model_name = models.get_model_name(dataset_name=dataset_name, dataloader_split=dataloader_split, model_type=model_type, dim_h=dim_h, model_to_mimic_name=None, bootstrap_num=bootstrap_num)
      model_name_base = model_name[len(dataset_name)+4:] 
      model = models.train_model_wrapper(train_loader=train_loader, val_loader=val_loader, model_name=model_name, model_type=model_type,
                                         dim_h=dim_h, dim_out=dim_out, epochs=epochs, end_target=bm_end_target, loss_type=bm_loss_type,
                                         force_train=False, model_to_mimic=None, lr=bm_lr, verbose=False, **kwargs)
      all_models[model_name] = model
      
      # Create stat pts for model before anything else
      if dataloader_split == 1 and bootstrap_num is None:
        stat_pts_file = f'StatPts_{model_name}'
        if os.path.exists(stat_pts_file):
          with open(stat_pts_file, 'rb') as file:
            this_pts = pickle.load(file)
            print(f'Loaded {stat_pts_file}')
        else:
          this_pts = create_all_pts(dataset=dataset, model_to_mimic=model, data_index_ids=np.arange(-num_graphs_for_stat_pts,0), max_workers=max_workers, is_int=is_int, do_grad_of_normed=True,
                                    device='cpu', ng_sigma=ng_sigma, norm_along_dir_threshold=norm_along_dir_threshold, coeffs_x_mask=coeffs_x_mask, lambda_reg=lambda_reg,
                                    max_targets_from_one_idx=max_targets_from_one_idx)
          if len(this_pts)==0:
            print('NO RESULTS!')
          else:
            with open(stat_pts_file, 'wb') as file:
              pickle.dump(this_pts, file)
              print(f'Saved {stat_pts_file}')
        all_stat_pts[model_name] = this_pts

  

  # Now steal the models
  for model_type in model_types:
    for dataloader_split_to_steal in [1]:
      model_name = models.get_model_name(dataset_name=dataset_name, dataloader_split=dataloader_split_to_steal, model_type=model_type, dim_h=dim_h, model_to_mimic_name=None, bootstrap_num=None)
      model_name_base = model_name[len(dataset_name)+4:] 
      model = all_models[model_name]
    
      list_steal_models = [(train_loader1_synth, val_loader1_synth, 1)]
      list_steal_models += [(train_loader2_synth, val_loader2_synth, 2)]
      for steal_train_loader, steal_val_loader, steal_dataloader_split in list_steal_models:
        if only_basic_tests:
          this_steal_model_types = [model_type]
          if model_type != 'GCN':
            this_steal_model_types += ['GCN']
          if model_type != 'MIX':
            this_steal_model_types += ['MIX']
          if model_type != 'GSAGE':
            this_steal_model_types += ['GSAGE']
        else:
          this_steal_model_types = steal_model_types
        for steal_model_type in this_steal_model_types:
          steal_model_name = models.get_model_name(dataset_name=dataset_name, dataloader_split=steal_dataloader_split, model_type=steal_model_type, dim_h=dim_h, model_to_mimic_name=model_name_base)
          steal_model = models.train_model_wrapper(train_loader=steal_train_loader, val_loader=steal_val_loader, model_name=steal_model_name, model_type=steal_model_type,
                                                   dim_h=dim_h, dim_out=dim_out, epochs=epochs, end_target='embed',
                                                   force_train=False, model_to_mimic=model, lr=steal_lr, verbose=False, **kwargs)
          all_models[steal_model_name] = steal_model

  all_Z = {}
  all_mimic_losses = {}
  for model_name, model in all_models.items():
    if f'{dataset_name}_bm_' in model_name:
      model_name_base = model_name[len(dataset_name)+4:] 
      has_mimickers = (len([x for x in all_models.keys() if f'steal_{model_name_base}_' in x])>0)
      if not has_mimickers:
        continue
      this_pts = all_stat_pts[model_name]
      if len(this_pts) > 0:
        Z_file = f'Z_{model_name}'
        if os.path.exists(Z_file):
          with open(Z_file, 'rb') as file:
            all_mimic_losses[model_name], all_Z[model_name] = pickle.load(file)
            print(f'Loaded {Z_file}')
        else:
          list_of_models_and_names = [(m, nm) for nm, m in all_models.items() if ('_bm_' in nm or f'steal_{model_name_base}' in nm or ('_boot' in nm and model_name_base in nm))]
          model_order = [x[1] for x in list_of_models_and_names if '_steal_' in x[1]] + [x[1] for x in list_of_models_and_names if '_bm_' in x[1] and x[1]!=model_name] + [x[1] for x in list_of_models_and_names if '_boot' in x[1]]
          all_mimic_losses[model_name] = DataFrame(Series(get_all_model_mimicking_losses(val_loader1, base_model=model, list_of_models_and_names=list_of_models_and_names)).round(3)).T.reindex(model_order, axis=1)
          all_Z[model_name] = model_perf(list_of_models_and_names=list_of_models_and_names, dataset=dataset, all_pts=this_pts, max_workers=max_workers, is_int=is_int, do_grad_of_normed=True)
          check_if_reasonable_for_basemodel(all_Z[model_name], base_model_name=model_name)
          print(all_mimic_losses[model_name])
          with open(Z_file, 'wb') as file:
            pickle.dump([all_mimic_losses[model_name], all_Z[model_name]], file)
            print(f'Saved {Z_file}')
          print()

  for model_name, Z in all_Z.items():
    print(f'=== {model_name} ===')
    pretty_print_results(Z, base_model_name=model_name)
    print('\tMimic losses:')
    print(all_mimic_losses[model_name])
    print()
    
  return all_Z

# Load all saved Z_ files from disk and assemble into a single multi-index DataFrame for analysis
def look_at_all_results():
  datasets=[
            ('citeseer', 8), 
            ('cifar', 8), 
            ('hiv', 32),  # int
            ('yelp', 8),
            ('mnist', 8), 
            ('bbbp', 8),  # int
            ('ogbmag', 32), 
            ('qm9', 64),
            ('amazoncomp', 8),  # is_int
            ('amazon', 8),
            ('coco', 8), 
            ('dblp', 8), 
            ('pubmed', 32), 
            ('fin', 8),
            ]

  models = ['GIN', 'GCN', 'GSAGE', 'MIX', 'ARMA']
  all_df = []
  for dataset, dim_h in datasets:
    this_dataset_df = []
    for model in models:
      base_model_name = f'{dataset}_bm_{model}{dim_h}s1'
      Z_file = f'Z_{base_model_name}'
      if os.path.exists(Z_file):
        with open(Z_file, 'rb') as file:
          res = pickle.load(file)
          model_losses = res[0]
          model_losses[base_model_name] = 0
          df_desc, df_tm, df_psc = pretty_print_results(res[1], base_model_name=base_model_name, verbose=False)
          this_df = pd.concat([model_losses.rename({0:'model_loss'}, axis=0), df_tm.rename({'Trim mean':'Score', 'std(bootstrap)':'Score std'}, axis=0), df_psc.rename({'Trim mean':'pscore', 'std(bootstrap)':'pscore std'}, axis=0)])
          this_df = this_df.rename(lambda x: x[len(dataset)+1:].replace(f'GIN{dim_h}', 'GIN').replace(f'GCN{dim_h}', 'GCN').replace(f'GSAGE{dim_h}', 'GSAGE').replace(f'MIX{dim_h}', 'MIX').replace(f'ARMA{dim_h}', 'ARMA'), axis=1)
          this_df = pd.concat([this_df], keys=[dataset], names=['dataset'])
          this_df = pd.concat([this_df], keys=[model], names=['base_model'], axis=1)
          this_dataset_df.append(this_df)
    if len(this_dataset_df) > 0:
      this_dataset_df = pd.concat(this_dataset_df, axis=1)
      all_df.append(this_dataset_df)
  all_df = pd.concat(all_df)
  return all_df

def stealmodel_with_bestloss(s, target_s):
  return s.loc[[x for x in s.index.values if x[1][:5]=='steal' and x[1][-2:]==target_s]].unstack('base_model').idxmin()

# Reshape all_df for a given measure, adding rows for the best-loss stolen model per base model
def clean_alldf_for_measure(all_df, measure, verbose=False):
  tmp = all_df.swaplevel(axis=0).loc[measure].T
  best_steals = {}
  for target_s in ['s1', 's2']:
    best_steals[target_s] = all_df.swaplevel(axis=0).loc['model_loss'].T.apply(lambda s: stealmodel_with_bestloss(s, target_s))
    if verbose:
      print(f'best_steal for target_s={target_s}')
      print(best_steals[target_s].T)
  
  out_df = []
  for bm in best_steals['s1'].index.values:
    this_df = tmp.loc[bm].copy()
    for target_s in ['s1', 's2']:
      if bm in best_steals[target_s].index.values:
        tmp2 = this_df.unstack().loc[best_steals[target_s].loc[bm].dropna().items()].droplevel(1)
        this_df.loc[f'steal_{bm}s1_using_BEST{target_s}', :] = tmp2
    out_df.append(this_df)
  out_df = pd.concat(out_df, keys=best_steals['s1'].index.values, names=['base_model'])

  return out_df


# Compute AUC and classification accuracy for distinguishing stolen vs independent models
def check_alldf(all_df, measure='pscore', which_steals='bestloss', pval_threshold=0.1):
  tmp = clean_alldf_for_measure(all_df, measure)
  dataset_order = tmp.columns.values
  methods_order = ['GCN', 'GIN', 'GSAGE', 'ARMA', 'MIX']

  bms = tmp.index.get_level_values('base_model').unique()
  all_auc = []
  for bm in bms:
    tmp2 = tmp.loc[bm].copy()

    if which_steals == 'same':
      steals = [x for x in tmp2.index.values if f'steal_{bm}s1_using_{bm}s' in x]
    elif which_steals == 'all':
      steals = [x for x in tmp2.index.values if x[:5]=='steal' and x[-6:-2]!='BEST']
    elif which_steals == 'bestloss':
      steals = [x for x in tmp2.index.values if x[:5]=='steal' and x[-6:-2]=='BEST']

    indeps = [x for x in tmp2.index.values if 'bm_' in x and x!=f'bm_{bm}s1']

   
    this_auc = tmp2.loc[steals+indeps].T.dropna(how='any').T.apply(lambda s: roc_auc_score(np.concatenate([np.zeros(len(steals)), np.ones(len(indeps))]), s.values))
    all_auc.append(this_auc)
  all_auc = pd.concat(all_auc, keys=bms, names=['base_model']).unstack('dataset')

  loss_and_score_have_sign_pos_corr = (all_df.swaplevel(axis=0).loc[['model_loss', measure]].unstack('dataset').stack(1).sort_index().apply(lambda s: scipy.stats.spearmanr(s.loc['model_loss'].dropna(), s.loc[measure].dropna())[1]).unstack('dataset') < pval_threshold).astype(bool).reindex(all_auc.columns.values, axis=1)

  print('AUC')
  print(all_auc.reindex(dataset_order, axis=1).reindex(methods_order, axis=0).T.round(2))
  print(f'\nLoss and score have significant positive correlation (pvalue={pval_threshold})?')
  print(loss_and_score_have_sign_pos_corr.reindex(dataset_order, axis=1).reindex(methods_order, axis=0).T)
  return all_auc, loss_and_score_have_sign_pos_corr


def pretty_print_auc(all_auc_or_list):
  dataset_order = [
            ('citeseer', 'Citeseer'), 
            ('ogbmag', 'OGBMag'), 
            ('hiv', 'HIV'),  # int
            ('yelp', 'Yelp'),
            ('mnist', 'MNIST'), 
            ('bbbp', 'BBBP'),  # int
            ('pubmed', 'Pubmed'), 
            ('qm9', 'QM9'),
            ('fin', 'Fin'),
            ('amazon', 'Amazon'),
            ('coco', 'Coco'), 
            ('dblp', 'DBLP'), 
            ('cifar', 'CIFAR'), 
            ('amazoncomp', 'Computers'),  # is_int
            ]
  methods_order = ['GCN', 'GIN', 'GSAGE', 'ARMA', 'MIX']
  if type(all_auc_or_list) != list:
    all_auc_or_list = [all_auc_or_list]

  res = [this_auc.reindex(list(zip(*dataset_order))[0], axis=1).reindex(methods_order, axis=0).rename(dict(dataset_order), axis=1).T.round(2) for this_auc in all_auc_or_list]

  res = pd.concat(res, keys=np.arange(len(res)), axis=1)
  res.loc['Average'] = res.mean()

  def tmp_print(s):
    out = ''
    for i in range(len(all_auc_or_list)):
      out += '\n\t & '
      out += ' & '.join([f'{x:2.2f}' if x>=0.5 else f'\\hl{{{x:2.2f}}}' for x in s.loc[i].values])
    out += '\n\\\\'
    return out

  res = res.T.apply(lambda s: tmp_print(s))
  for index, row in res.items():
    print(index, row)


def do_timing(dataset, dataset_name, dim_h, dim_out, end_target, num_pts=5, device='cpu', is_int=False):
  model_types = ['GCN', 'GIN', 'GSAGE', 'ARMA', 'MIX']
  num_features = dataset[0].my_x.shape[1]
  all_time_taken = []
  for bm in model_types:
    if bm == 'GCN':
      this_bm = models.GCN(dim_h=dim_h, dim_out=dim_out, num_features=num_features, end_target=end_target)
    elif bm == 'GSAGE':
      this_bm = models.GraphSAGE(dim_h=dim_h, dim_out=dim_out, num_features=num_features, end_target=end_target)
    elif bm == 'GIN':
      this_bm = models.GIN(dim_h=dim_h, dim_out=dim_out, num_features=num_features, end_target=end_target)
    elif bm == 'MIX':
      this_bm = models.MixHop(dim_h=dim_h, dim_out=dim_out, num_features=num_features, end_target=end_target)
    elif bm == 'ARMA':
      this_bm = models.ARMA(dim_h=dim_h, dim_out=dim_out, num_features=num_features, end_target=end_target)
    models.load_model(this_bm, f'{dataset_name}_bm_{bm}{dim_h}s1', verbose=False)
    with open(f'StatPts_{dataset_name}_bm_{bm}{dim_h}s1', 'rb') as file:
      ap = pickle.load(file)
      num_pts_seen = 0
      this_bm_time_taken = []
      for idx in ap.keys():
        x0, e = dataset[idx].my_x.to(device), dataset[idx].edge_index.to(device)
        for target_node in ap[idx].keys():
          _, time_taken = get_lo_hi_points_helper(target_node, x0, e, model_to_mimic=this_bm, device=device, random_coeffs=True, is_int=is_int, opt_iterations=1000, ng_sigma=0.1, norm_along_dir_threshold=0., do_grad_of_normed=True, coeffs_x_mask=None, lambda_reg=0.01, return_times=True)
          this_bm_time_taken.append(time_taken)
          if len(this_bm_time_taken) >= num_pts:
            break
        if len(this_bm_time_taken) >= num_pts:
          break
    this_bm_time_taken = pd.concat(this_bm_time_taken, axis=1)
    all_time_taken.append(this_bm_time_taken)
  all_time_taken = pd.concat(all_time_taken, keys=model_types, names=['model'])
  print(all_time_taken.swaplevel(axis=0).sort_index().loc['total'].T.describe())
  return all_time_taken

def best_stealmodel_for_each_setting(all_df):
  z = all_df.swaplevel(axis=0).loc['model_loss'].T.unstack(0)
  best_model_df = z.loc[[x for x in z.index.values if 'steal_' in x]].idxmin().unstack().map(lambda x: x.split('_')[-1][:-2])
  return best_model_df


def epsilon_ratio(dataset, ds_name, bm_type, steal_type, split_size, dim_h, dim_out, end_target, standardize=False, batch_size=64):
  from pregip import get_model
  num_features = dataset[0].my_x.shape[1]
  train_loader1, val_loader1 = get_data.setup_data(dataset, n_max=split_size, seed=0, start_idx=0, standardize=standardize, batch_size=batch_size)
  bm = get_model(bm_type, dim_h, dim_out, num_features, end_target)
  models.load_model(bm, f'{ds_name}_bm_{bm_type}{dim_h}s1')
  s = get_model(steal_type, dim_h, dim_out, num_features, 'embed')
  models.load_model(s, f'{ds_name}_steal_{bm_type}{dim_h}s1_using_{steal_type}{dim_h}s2')
  dum = models.Dummy(dim_h=dim_h)

  s_err = models.val_loss(s, val_loader1, model_to_mimic=bm)
  dum_err = models.val_loss(dum, val_loader1, model_to_mimic=bm)
  print(f's_err={s_err:4.4f}, dum_err={dum_err:4.4f}, ratio={s_err/dum_err:4.4f}')

import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import torch
from torch_geometric.data import Data
import scipy.stats
import pickle

import sys

import models
import our_setup

def create_M_gaussian(dim_h, dim_mult, scale_min=1, scale_max=1):
  dim_h_new = int(dim_h * dim_mult)
  return torch.randn((dim_h_new, dim_h)) * (torch.rand(dim_h_new) * (scale_max-scale_min) + scale_min)[:, None]

def create_M_permute(dim_h):
  return torch.eye(dim_h)[torch.randperm(dim_h, dtype=torch.long)]

def create_M_orth(dim_h, dim_mult, scale=1):
  dim_h_new = int(dim_h * dim_mult)
  if dim_h_new < dim_h:
    P = torch.hstack([torch.eye(dim_h_new), torch.zeros((dim_h-dim_h_new, dim_h))])
  elif dim_h_new == dim_h:
    P = torch.eye(dim_h)
  else:
    P = torch.vstack([torch.eye(dim_h), torch.zeros((dim_h_new-dim_h, dim_h))])
  return scale * (torch.linalg.svd(torch.randn((dim_h_new, dim_h_new)), full_matrices=True)[0] @ P)

# Build a list of (name, type, param) tuples for all embedding transformations to test
def create_all_M(dim_h, list_params_M=[], list_params_pow=[], list_params_exp=[], list_params_norm=[], list_params_translate=[], list_params_arctan=[], list_params_orth=[], num_permute=0, seed=0):
  # list_params is a list of (dim_mult, scale_max) tuples
  np.random.seed(seed)
  torch.manual_seed(seed)
  all_M = [ ('Identity', 'M', torch.eye(dim_h)) ] 
  all_M += [ (f'dim_mult={dim_mult}, scale_max={scale_max}', 'M', create_M_gaussian(dim_h=dim_h, dim_mult=dim_mult, scale_max=scale_max)) for dim_mult, scale_max in list_params_M ]
  all_M += [ (f'x**{m}', 'power', m) for m in list_params_pow ]
  all_M += [ (f'exp({m}*x)', 'exp', m) for m in list_params_exp ]
  all_M += [ (f'x/norm(x, {m})', 'norm', m) for m in list_params_norm ]
  all_M += [ (f'x+unif({m_lo}, {m_hi})', 'translate', torch.rand(dim_h)*(m_hi-m_lo)+m_lo) for m_lo, m_hi in list_params_translate ]
  all_M += [ (f'arctan({m}*x)', 'arctan', m) for m in list_params_arctan ]
  all_M += [ ('sinh(x)', 'sinh', None), ('tanh(x)', 'tanh', None), ('sigmoid(x)', 'sigmoid', None) ]
  all_M += [ (f'{scale} * O * P ({dim_mult}x dim chg)', 'M', create_M_orth(dim_h=dim_h, dim_mult=dim_mult, scale=scale)) for dim_mult, scale in list_params_orth ]
  all_M += [ ('Permutation', 'M', create_M_permute(dim_h)) for i in range(num_permute) ]
  return all_M

# For each transform M, compute delta(M(H)) / norm(M(H)) at stationary vs original points, return pscores
def test_transforms(model, dataset, all_pts, all_M, device='cpu'):
  model.eval()
  model = model.to(device)
  results = [{'lo':[], 'orig':[]} for i in range(len(all_M))]
  with torch.no_grad():
    for idx in all_pts.keys():
      x0, e = dataset[idx].my_x.to(device), dataset[idx].edge_index.to(device)
      for target_node in all_pts[idx].keys():
        z = all_pts[idx][target_node]
        coeffs_x = z['coeffs_x'].to(device)
        delta_lo, delta_hi = z['delta_x'] 
        x_zerograd = torch.tensor(z['lo'], dtype=torch.float32, device=device)
        x_orig = x0.detach().clone()
        tmp = torch.zeros_like(x_zerograd, device=device)
        tmp[target_node] = coeffs_x
        for this_point_name, this_point in [('lo', x_zerograd), ('orig', x_orig)]:
          h = model.get_H(Data(my_x=this_point, edge_index=e))[target_node]
          hnorm = torch.norm(h).item()
          t1 = model.get_H(Data(my_x=this_point+delta_hi*tmp, edge_index=e))[target_node]
          t2 = model.get_H(Data(my_x=this_point-delta_lo*tmp, edge_index=e))[target_node]

          for i, (Mname, Mtype, M) in enumerate(all_M):
            if hnorm==0:
              val = np.nan
            else:
              if Mtype == 'M':
                M = M.to(device)
                val = torch.norm(torch.mv(M, t1-t2)).item() / torch.norm(torch.mv(M, h)).item()
              elif Mtype == 'power':
                val = torch.norm(torch.pow(t1, M) - torch.pow(t2, M)).item() / torch.norm(torch.pow(h, M)).item()
              elif Mtype == 'exp':
                val = torch.norm(torch.exp(M * t1) - torch.exp(M * t2)).item() / torch.norm(torch.exp(M * h)).item()
              elif Mtype == 'norm':
                val = torch.norm(t1/torch.linalg.vector_norm(t1, M) - t2/torch.linalg.vector_norm(t2, M)).item() / torch.norm(h/torch.linalg.vector_norm(h, M)).item()
              elif Mtype == 'translate':
                val = torch.norm(t1 - t2).item() / torch.norm(h + M).item()
              elif Mtype == 'arctan':
                val = torch.norm(torch.arctan(M * t1) - torch.arctan(M * t2)).item() / torch.norm(torch.arctan(M * h)).item()
              elif Mtype == 'sinh':
                val = torch.norm(torch.sinh(t1) - torch.sinh(t2)).item() / torch.norm(torch.sinh(h)).item()
              elif Mtype == 'tanh':
                val = torch.norm(torch.tanh(t1) - torch.tanh(t2)).item() / torch.norm(torch.tanh(h)).item()
              elif Mtype == 'sigmoid':
                val = torch.norm(torch.sigmoid(t1) - torch.sigmoid(t2)).item() / torch.norm(torch.sigmoid(h)).item()
            results[i][this_point_name].append(val)
  res_pscores = []
  for r in results:
    s1 = Series(r['orig']).replace(np.inf, np.nan).dropna()
    s2 = Series(r['lo']).replace(np.inf, np.nan).dropna()
    z = Series(np.searchsorted(np.sort(s1), s2)/len(s1)*100)
    res_pscores.append(scipy.stats.trim_mean(z, proportiontocut=0.1))
  res_pscores = Series(res_pscores, index=list(zip(*all_M))[0])
  return res_pscores


# Test all transforms against all base models for one dataset, return pass rates relative to independent models
def do_one_dataset(dataset, 
                   dataset_name='citeseer', 
                   dim_h=8, 
                   verbose=True,
                   base_models=['GIN', 'GCN', 'GSAGE', 'MIX', 'ARMA'], 
                   list_params_M=[(1,1), (5,1), (1,10), (5, 10)],
                   list_params_pow=[3, 5],
                   list_params_exp=[1, -1],
                   list_params_norm=[1, 2],
                   list_params_translate=[(1,1), (1,5), (1,10)],
                   list_params_arctan=[1, -1],
                   list_params_orth=[(1, 1), (1,5), (5,5)],
                   num_repeats=20,
                   **kwargs):

  dim_out = None
  num_features = dataset[0].my_x.shape[1]
  all_M = create_all_M(dim_h=dim_h,
                       list_params_M=list_params_M*num_repeats, 
                       list_params_pow=list_params_pow, 
                       list_params_exp=list_params_exp, 
                       list_params_norm=list_params_norm,
                       list_params_translate=list_params_translate,
                       list_params_arctan=list_params_arctan, 
                       list_params_orth=list_params_orth,
                       num_permute=num_repeats,
                       )
  df = {}
  df_pass = {}
  df_pass_allbms = {}
  for bm in base_models:
    with open(f'StatPts_{dataset_name}_bm_{bm}{dim_h}s1', 'rb') as file:
      ap = pickle.load(file)
    with open(f'Z_{dataset_name}_bm_{bm}{dim_h}s1', 'rb') as file:
      model_loss, Z = pickle.load(file)
      _, _, df_psc = our_setup.pretty_print_results(Z, base_model_name=f'{dataset_name}_bm_{bm}{dim_h}s1', verbose=False)
      cols_allbms = [x for x in df_psc.columns.values if 'bm_' in x and x!=f'{dataset_name}_bm_{bm}{dim_h}s1']
      threshold_val = df_psc.loc['Trim mean', cols_allbms].min()  # Closest independent base_model

    if bm == 'GCN':
      stolen_model = models.GCN(dim_h=dim_h, dim_out=dim_out, num_features=num_features, end_target='embed')
    elif bm == 'GSAGE':
      stolen_model = models.GraphSAGE(dim_h=dim_h, dim_out=dim_out, num_features=num_features, end_target='embed')
    elif bm == 'GIN':
      stolen_model = models.GIN(dim_h=dim_h, dim_out=dim_out, num_features=num_features, end_target='embed')
    elif bm == 'MIX':
      stolen_model = models.MixHop(dim_h=dim_h, dim_out=dim_out, num_features=num_features, end_target='embed')
    elif bm == 'ARMA':
      stolen_model = models.ARMA(dim_h=dim_h, dim_out=dim_out, num_features=num_features, end_target='embed')
    models.load_model(stolen_model, f'{dataset_name}_steal_{bm}{dim_h}s1_using_{bm}{dim_h}s2', verbose=verbose)
    this_s = test_transforms(model=stolen_model, dataset=dataset, all_pts=ap, all_M=all_M, **kwargs)
    df[bm] = this_s
    df_pass[bm] = (this_s < threshold_val)
    df_pass_allbms[bm] = this_s.map(lambda x: (df_psc.loc['Trim mean', cols_allbms]>x).mean())
  df = DataFrame(df)
  df_pass = DataFrame(df_pass)
  df_pass_allbms = DataFrame(df_pass_allbms)
  df.index.name = 'setting'
  df_pass.index.name = 'setting'
  df_pass_allbms.index.name = 'setting'
  df = pd.concat([df.reset_index().groupby('setting').mean(), df.reset_index().groupby('setting').std()], keys=['mean', 'std'], names=['stat'], axis=1).round(1)
  df_pass = df_pass.reset_index().groupby('setting').mean()
  df_pass_allbms = df_pass_allbms.reset_index().groupby('setting').mean()
  return df, df_pass, df_pass_allbms


def do_all(verbose=False, **all_datasets):

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
  
  all_df_pscores, all_df_pass, all_df_pass_allbms, all_names = [], [], [], []
  for dataset_name, dim_h in datasets:
    if dataset_name in all_datasets:
      print(dataset_name)
      this_df_pscores, this_df_pass, this_df_pass_allbms = do_one_dataset(all_datasets[dataset_name], dataset_name=dataset_name, dim_h=dim_h, verbose=verbose)
      all_df_pscores.append(this_df_pscores)
      all_df_pass.append(this_df_pass)
      all_df_pass_allbms.append(this_df_pass_allbms)
      all_names.append(dataset_name)

  all_df_pscores = pd.concat(all_df_pscores, keys=all_names, names=['dataset'], axis=1)
  all_df_pass = pd.concat(all_df_pass, keys=all_names, names=['dataset'], axis=1)
  all_df_pass_allbms = pd.concat(all_df_pass_allbms, keys=all_names, names=['dataset'], axis=1)

  pretty_print(all_df_pass_allbms)

  return all_df_pscores, all_df_pass, all_df_pass_allbms

# Format transform robustness results as relative difference from Identity
def pretty_print(all_df_pass_allbms_or_list, show_latex=True, highlight_threshold=50):
  methods_order = ['GCN', 'GIN', 'GSAGE', 'ARMA', 'MIX']
  transform_names = [
      ('Permutation', 'Permute'),
      ('1 * O * P (1x dim chg)', 'Rotate'),
      ('5 * O * P (1x dim chg)', 'Rotate, scale by 5'),
      ('5 * O * P (5x dim chg)', r'Project($\mathbb{R}^d \to \mathbb{R}^{5d}$), rotate, scale by 5'),
      ('dim_mult=1, scale_max=1', r'Multiply by $d\times d$ Gaussian mat.'),
      ('dim_mult=1, scale_max=10', r'(as above) and scale each entry by Unif$(1,10)$'),
      ('dim_mult=5, scale_max=1', r'Multiply by $5d\times d$ Gaussian mat.'),
      ('dim_mult=5, scale_max=10', r'(as above) and scale each entry by Unif$(1,10)$'),
      ('arctan(1*x)', r'$\bh_{ij}\to \tan^{-1}(\bh_{ij})$'),
      ('exp(1*x)', r'$\bh_{ij}\to \exp(\bh_{ij})$'),
      ('sigmoid(x)', r'$\bh_{ij}\to \text{sigmoid}(\bh_{ij})$'),
      ('sinh(x)', r'$\bh_{ij}\to \text{sinh}(\bh_{ij})$'),
      ('tanh(x)', r'$\bh_{ij}\to \text{tanh}(\bh_{ij})$'),
      ('x**3', r'$\bh_{ij}\to\bh_{ij}^3$'),
      ('x**5', r'$\bh_{ij}\to\bh_{ij}^5$'), 
      ('x+unif(1, 1)', r'$\bh_{ij}\to \bh_{ij}+1$'),
      ('x+unif(1, 5)', r'$\bh_{ij}\to \bh_{ij}+\mathrm{Unif}(1,5)$'), 
      ('x+unif(1, 10)', r'$\bh_{ij}\to \bh_{ij}+\mathrm{Unif}(1,10)$'), 
      ('x/norm(x, 1)', r'$\bh_{i}\to \bh_{i}/\|\bh_i\|_1$'), 
      ('x/norm(x, 2)', r'$\bh_{i}\to \bh_{i}/\|\bh_i\|_2$'),
      ]
  index, values = zip(*transform_names)
  transform_names = Series(values, index=index)

  if type(all_df_pass_allbms_or_list) != list:
    all_df_pass_allbms_or_list = [all_df_pass_allbms_or_list]

  res = []
  for all_df_pass_allbms in all_df_pass_allbms_or_list:
    all_df_pass_allbms2 = all_df_pass_allbms.T[all_df_pass_allbms.loc['Identity']>0].T
    all_df_pass_allbms3 = (all_df_pass_allbms2 / all_df_pass_allbms2.loc['Identity'] - 1).abs().drop('Identity', axis=0)

    z = all_df_pass_allbms3.T.unstack(1).mean().unstack(1).reindex(methods_order, axis=1).reindex(index, axis=0).rename(transform_names, axis=0) #.round(2)
    z = pd.concat([z, DataFrame({'Average':z.fillna(1).mean()}).T], axis=0).round(2)  # NaN is considered 100% relative difference from Identity
    res.append(z)

  res = pd.concat(res, keys=np.arange(len(all_df_pass_allbms_or_list)), axis=1)
  res.loc['Average'] = res.mean()
  
  print('Relative difference from Identity:')
  if not show_latex:
    print((res*100).round(0))
  else:
    def tmp_str(x):
      if np.isnan(x):
        return r'\hl{$\times$}'
      elif x>=50:
        return f'\\hl{{{int(x)}\\%}}'
      else:
        return f'{int(x)}\\%'
    def tmp_print(s):
      out = ''
      for i in range(len(all_df_pass_allbms_or_list)):
        out += '\n\t & '
        out += ' & '.join([tmp_str(x) for x in s.loc[i].values])
      out += '\n\\\\'
      return out
    res = (res*100).T.apply(lambda s: tmp_print(s))
    for index, row in res.items():
      print(index, row)

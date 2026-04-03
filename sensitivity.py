import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from sklearn.metrics import roc_auc_score
import torch

import min_grad, our_setup, models
import sys, os, pickle

def get_model(model_type, dim_h, dim_out, num_features, end_target):
  if model_type == 'GCN':
    model = models.GCN(dim_h=dim_h, dim_out=dim_out, num_features=num_features, end_target=end_target)
  elif model_type == 'GSAGE':
    model = models.GraphSAGE(dim_h=dim_h, dim_out=dim_out, num_features=num_features, end_target=end_target)
  elif model_type == 'GIN':
    model = models.GIN(dim_h=dim_h, dim_out=dim_out, num_features=num_features, end_target=end_target)
  elif model_type == 'MIX':
    model = models.MixHop(dim_h=dim_h, dim_out=dim_out, num_features=num_features, end_target=end_target)
  elif model_type == 'ARMA':
    model = models.ARMA(dim_h=dim_h, dim_out=dim_out, num_features=num_features, end_target=end_target)
  else:
    print(f'Unknown model_type={model_type}')
    sys.exit(1)
  return model


# how does detection AUC change when using fewer stationary points
def vary_sizeStatPts(all_df, dataset, dataset_name, dim_h, dim_out, end_target, is_int, alt_numpts=[10, 20, 40, 60, 80], chg_model=None):
  z1 = all_df.swaplevel(axis=0).loc['model_loss'].T.unstack(0)[dataset_name]
  best_steals_s1 = z1.loc[[x for x in z1.index.values if 'steal_' in x and x[-2:]=='s1']].idxmin().map(lambda x: x.split('_')[-1][:-2])

  num_features = dataset[0].my_x.shape[1]
  all_models = {}
  all_Z = {}
  aucs = []
  for model_type in ['GCN', 'GIN', 'GSAGE', 'ARMA', 'MIX']:
    for split in [1,2]:
      model_name = f'{dataset_name}_bm_{model_type}{dim_h}s{split}'
      model = get_model(model_type, dim_h, dim_out, num_features, end_target)
      models.load_model(model, model_name, verbose=False)
      all_models[model_name] = model

    if chg_model is None or model_type not in chg_model:
      steal_model_type = best_steals_s1[model_type]
    else:
      steal_model_type = best_steals_s1[chg_model[model_type]]

    steal_model_name = f'{dataset_name}_steal_{model_type}{dim_h}s1_using_{steal_model_type}{dim_h}s1'
    steal_model = get_model(steal_model_type, dim_h, dim_out, num_features, end_target='embed')
    models.load_model(steal_model, steal_model_name, verbose=False)
    all_models[steal_model_name] = steal_model

    steal_model_name = f'{dataset_name}_steal_{model_type}{dim_h}s1_using_{steal_model_type}{dim_h}s2'
    steal_model = get_model(steal_model_type, dim_h, dim_out, num_features, end_target='embed')
    models.load_model(steal_model, steal_model_name, verbose=False)
    all_models[steal_model_name] = steal_model

  for model_type in ['GCN', 'GIN', 'GSAGE', 'ARMA', 'MIX']:
    model_name_base = f'{model_type}{dim_h}s1'
    model_name = f'{dataset_name}_bm_{model_type}{dim_h}s1'
    model = all_models[model_name]
      
    stat_pts_file = f'StatPts_{model_name}'
    if os.path.exists(stat_pts_file):
      with open(stat_pts_file, 'rb') as file:
        this_pts_full = pickle.load(file)
        print(f'Loaded {stat_pts_file}')

      totpts = [len(this_pts_full[i]) for i in this_pts_full]
      cumsum = np.cumsum(totpts)
      for alt_n in alt_numpts:
        if alt_n>cumsum[-1]:
          this_pts = this_pts_full
        else:
          num_idx_to_keep = np.where(np.cumsum(totpts)>=alt_n)[0][0]+1
          print(f'num_idx_to_keep={num_idx_to_keep}')
          this_pts = dict(list(this_pts_full.items())[:num_idx_to_keep])

        Z_file = f'VaryNumpts{alt_n}_Z_{model_name}'
        if os.path.exists(Z_file):
          with open(Z_file, 'rb') as file:
            all_Z[model_name] = pickle.load(file)
            print(f'Loaded {Z_file}')
        else:
          list_of_models_and_names =[(m, nm) for nm, m in all_models.items() if ('_bm_' in nm or f'steal_{model_name_base}' in nm)]
          model_order = [x[1] for x in list_of_models_and_names if '_steal_' in x[1]] + [x[1] for x in list_of_models_and_names if '_bm_' in x[1] and x[1]!=model_name]
          all_Z[model_name] = our_setup.model_perf(list_of_models_and_names=list_of_models_and_names, dataset=dataset, all_pts=this_pts, max_workers=0, is_int=is_int, verbose=False)
          with open(Z_file, 'wb') as file:
            pickle.dump(all_Z[model_name], file)
            print(f'Saved {Z_file}')
          print()
        
        df_desc, df_tm, df_psc = our_setup.pretty_print_results(all_Z[model_name], base_model_name=model_name, verbose=False)

        scores = df_psc.loc['Trim mean'].iloc[1:]
        num_steals = len([x for x in scores.index.values if 'steal' in x])
        actuals = np.concatenate([np.zeros(num_steals), np.ones(len(scores)-num_steals)])
        aucs.append([model_type, alt_n, roc_auc_score(actuals, scores.values)])
        print(aucs[-1])

  aucs = DataFrame(aucs, columns=['model_type', 'numpts', 'AUC'])
  return aucs


# how does detection AUC change with different regularization strengths
def vary_lambdareg(all_df, dataset, dataset_name, dim_h, dim_out, end_target, is_int, ng_sigma=0.1,
                   norm_along_dir_threshold=0.05, max_targets_from_one_idx=5, alt_lambdas=[0.1, 0.5, 1], chg_model=None):


  z1 = all_df.swaplevel(axis=0).loc['model_loss'].T.unstack(0)[dataset_name]
  best_steals_s1 = z1.loc[[x for x in z1.index.values if 'steal_' in x and x[-2:]=='s1']].idxmin().map(lambda x: x.split('_')[-1][:-2])
  num_features = dataset[0].my_x.shape[1]
  all_models = {}
  all_Z = {}
  aucs = []
  for model_type in ['GCN', 'GIN', 'GSAGE', 'ARMA', 'MIX']:
    for split in [1,2]:
      model_name = f'{dataset_name}_bm_{model_type}{dim_h}s{split}'
      model = get_model(model_type, dim_h, dim_out, num_features, end_target)
      models.load_model(model, model_name, verbose=False)
      all_models[model_name] = model

    if chg_model is None or model_type not in chg_model:
      steal_model_type = best_steals_s1[model_type]
    else:
      steal_model_type = best_steals_s1[chg_model[model_type]]

    steal_model_name = f'{dataset_name}_steal_{model_type}{dim_h}s1_using_{steal_model_type}{dim_h}s1'
    steal_model = get_model(steal_model_type, dim_h, dim_out, num_features, end_target='embed')
    models.load_model(steal_model, steal_model_name, verbose=False)
    all_models[steal_model_name] = steal_model

    steal_model_name = f'{dataset_name}_steal_{model_type}{dim_h}s1_using_{steal_model_type}{dim_h}s2'
    steal_model = get_model(steal_model_type, dim_h, dim_out, num_features, end_target='embed')
    models.load_model(steal_model, steal_model_name, verbose=False)
    all_models[steal_model_name] = steal_model

  for model_type in ['GCN', 'GIN', 'GSAGE', 'ARMA', 'MIX']:
    model_name_base = f'{model_type}{dim_h}s1'
    model_name = f'{dataset_name}_bm_{model_type}{dim_h}s1'
    model = all_models[model_name]
    for lambda_reg in alt_lambdas:
      stat_pts_file = f'VaryLambda{lambda_reg}_StatPts_{model_name}'
      if os.path.exists(stat_pts_file):
        with open(stat_pts_file, 'rb') as file:
          this_pts = pickle.load(file)
          print(f'Loaded {stat_pts_file}')
      else:
        this_pts = our_setup.create_all_pts(
                                  dataset=dataset, model_to_mimic=model, data_index_ids=np.arange(-100,0), max_workers=0, is_int=is_int, 
                                  device='cpu', ng_sigma=ng_sigma, norm_along_dir_threshold=norm_along_dir_threshold, coeffs_x_mask=None, lambda_reg=lambda_reg,
                                  max_targets_from_one_idx=max_targets_from_one_idx) # Check ng_sigma
        if len(this_pts)==0:
          print('NO RESULTS!')
        else:
          with open(stat_pts_file, 'wb') as file:
            pickle.dump(this_pts, file)
            print(f'Saved {stat_pts_file}')


      if len(this_pts) > 0:
        Z_file = f'VaryLambda{lambda_reg}_Z_{model_name}'
        if os.path.exists(Z_file):
          with open(Z_file, 'rb') as file:
            all_Z[model_name] = pickle.load(file)
            print(f'Loaded {Z_file}')
        else:
          list_of_models_and_names =[(m, nm) for nm, m in all_models.items() if ('_bm_' in nm or f'steal_{model_name_base}' in nm)]
          model_order = [x[1] for x in list_of_models_and_names if '_steal_' in x[1]] + [x[1] for x in list_of_models_and_names if '_bm_' in x[1] and x[1]!=model_name]
          all_Z[model_name] = our_setup.model_perf(list_of_models_and_names=list_of_models_and_names, dataset=dataset, all_pts=this_pts, max_workers=0, is_int=is_int)
          with open(Z_file, 'wb') as file:
            pickle.dump(all_Z[model_name], file)
            print(f'Saved {Z_file}')
          print()
        
      df_desc, df_tm, df_psc = our_setup.pretty_print_results(all_Z[model_name], base_model_name=model_name, verbose=False)

      scores = df_psc.loc['Trim mean'].iloc[1:]
      num_steals = len([x for x in scores.index.values if 'steal' in x])
      actuals = np.concatenate([np.zeros(num_steals), np.ones(len(scores)-num_steals)])
      aucs.append([model_type, lambda_reg, roc_auc_score(actuals, scores.values)])
      print(aucs[-1])

  aucs = DataFrame(aucs, columns=['model_type', 'lambda', 'AUC'])
  return aucs


def do_several_vary_lambdareg(all_df, **datasets):
  dataset_configs = {
    'citeseer':   dict(dim_h=8,  dim_out=6,  end_target='node', is_int=False, chg_model={'GSAGE':'GCN'}),
    'amazoncomp': dict(dim_h=8,  dim_out=10, end_target='node', is_int=False, ng_sigma=1, norm_along_dir_threshold=0.01, max_targets_from_one_idx=10, chg_model={'GSAGE':'MIX'}),
    'amazon':     dict(dim_h=8,  dim_out=5,  end_target='node', is_int=False, max_targets_from_one_idx=3, chg_model={'ARMA':'MIX', 'GSAGE':'MIX'}),
    'dblp':       dict(dim_h=8,  dim_out=4,  end_target='node', is_int=False, chg_model={'GSAGE':'MIX'}),
    'pubmed':     dict(dim_h=32, dim_out=3,  end_target='node', is_int=False, chg_model={'ARMA':'MIX', 'GSAGE':'MIX'}),
  }

  auc_sens = []
  dataset_names = []
  for name, data in datasets.items():
    if name not in dataset_configs:
      print(f'Warning: no config for dataset "{name}", skipping')
      continue
    auc_sens.append(vary_lambdareg(all_df, data, name, **dataset_configs[name]))
    dataset_names.append(name)

  auc_sens = pd.concat(auc_sens, keys=dataset_names, names=['dataset']).reset_index().drop('level_1', axis=1)
  base_aucs = []
  for d in dataset_names:
    for m in ['GCN', 'GIN', 'GSAGE', 'ARMA', 'MIX']:
      this_auc=1.0
      if d=='amazon':
        if m=='GIN':
          this_auc=0.64
        elif m=='ARMA':
          this_auc=0.67
      elif d=='dblp':
        if m=='GSAGE':
          this_auc=0.64
      elif d=='pubmed':
        if m=='GSAGE':
          this_auc=0.79
        elif m=='ARMA':
          this_auc=0.83
        elif m=='MIX':
          this_auc=0.93
      base_aucs.append([d, m, 0.01, this_auc])
  base_aucs = DataFrame(base_aucs, columns=['dataset', 'model_type', 'lambda', 'AUC'])
  auc_sens = pd.concat([auc_sens, base_aucs])
  auc_sens = auc_sens.set_index(['dataset', 'model_type', 'lambda']).sort_index().unstack('lambda')['AUC']
  return auc_sens

def do_several_vary_sizeStatPts(all_df, **datasets):
  dataset_configs = {
    'citeseer':   dict(dim_h=8,  dim_out=6,  end_target='node', is_int=False, chg_model={'GSAGE':'GCN'}),
    'amazoncomp': dict(dim_h=8,  dim_out=10, end_target='node', is_int=False, chg_model={'GSAGE':'MIX'}),
    'amazon':     dict(dim_h=8,  dim_out=5,  end_target='node', is_int=False, chg_model={'ARMA':'MIX', 'GSAGE':'MIX'}),
    'dblp':       dict(dim_h=8,  dim_out=4,  end_target='node', is_int=False, chg_model={'GSAGE':'MIX'}),
    'pubmed':     dict(dim_h=32, dim_out=3,  end_target='node', is_int=False, chg_model={'ARMA':'MIX', 'GSAGE':'MIX'}),
  }

  auc_sens = []
  dataset_names = []
  for name, data in datasets.items():
    if name not in dataset_configs:
      print(f'Warning: no config for dataset "{name}", skipping')
      continue
    auc_sens.append(vary_sizeStatPts(all_df, data, name, **dataset_configs[name]))
    dataset_names.append(name)

  auc_sens = pd.concat(auc_sens, keys=dataset_names, names=['dataset']).reset_index().drop('level_1', axis=1)
  base_aucs = []
  for d in dataset_names:
    for m in ['GCN', 'GIN', 'GSAGE', 'ARMA', 'MIX']:
      this_auc=1.0
      if d=='amazon':
        if m=='GIN':
          this_auc=0.64
        elif m=='ARMA':
          this_auc=0.67
      elif d=='dblp':
        if m=='GSAGE':
          this_auc=0.64
      elif d=='pubmed':
        if m=='GSAGE':
          this_auc=0.79
        elif m=='ARMA':
          this_auc=0.83
        elif m=='MIX':
          this_auc=0.93
      base_aucs.append([d, m, 100, this_auc])
  base_aucs = DataFrame(base_aucs, columns=['dataset', 'model_type', 'numpts', 'AUC'])
  auc_sens = pd.concat([auc_sens, base_aucs])
  auc_sens = auc_sens.set_index(['dataset', 'model_type', 'numpts']).sort_index().unstack('numpts')['AUC']
  return auc_sens
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, MixHopConv, ARMAConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import Data

import torch
import torch.nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU

from tqdm import tqdm

import os
import gc
import sys

class MixHop(torch.nn.Module):
  """Graph Convolutional Network class with 3 convolutional layers and a linear layer"""

  def __init__(self, dim_h, num_features, dim_out=1, end_target='graph'):
    """init method for GCN
    
    Args:
        dim_h (int): the dimension of hidden layers
    """
    super().__init__()
    self.conv1 = MixHopConv(num_features, dim_h, powers=[0,1,2])
    self.conv2 = MixHopConv(dim_h*3, dim_h, powers=[0,1])
    self.lin1 = torch.nn.Linear(dim_h*2, dim_h)
    self.end_target = end_target
    if end_target=='graph' or end_target=='node':
      self.lin = torch.nn.Linear(dim_h, dim_out)
  
  def get_H(self, data):
    e = data.edge_index
    x = data.my_x
    
    x = self.conv1(x, e)
    x = x.relu()
    x = self.conv2(x, e)
    x = x.relu()
    x = self.lin1(x)
    return x

  def forward(self, data):
    x = self.get_H(data)
    if self.end_target == 'graph':
      x = global_mean_pool(x, data.batch)
      x = F.dropout(x, p=0.5, training=self.training)
    if self.end_target=='graph' or self.end_target=='node':
      x = self.lin(x)
    return x

class ARMA(torch.nn.Module):
  def __init__(self, dim_h, num_features, dim_out=1, end_target='graph'):
    super().__init__()
    self.conv1 = ARMAConv(num_features, dim_h, num_stacks=2, num_layers=2, dropout=0.25)
    self.conv2 = ARMAConv(dim_h, dim_h, num_stacks=2, num_layers=2, dropout=0.25)
    self.lin1 = torch.nn.Linear(dim_h, dim_h)
    self.end_target = end_target
    if end_target=='graph' or end_target=='node':
      self.lin = torch.nn.Linear(dim_h, dim_out)
  
  def get_H(self, data):
    e = data.edge_index
    x = data.my_x
    
    x = self.conv1(x, e)
    x = x.relu()
    x = F.dropout(x, p=0.5, training=self.training) # Added later
    x = self.conv2(x, e)
    x = x.relu()
    x = F.dropout(x, p=0.5, training=self.training) # Added later
    x = self.lin1(x)
    return x

  def forward(self, data):
    x = self.get_H(data)
    if self.end_target == 'graph':
      x = global_mean_pool(x, data.batch)
      x = F.dropout(x, p=0.5, training=self.training)
    if self.end_target=='graph' or self.end_target=='node':
      x = self.lin(x)
    return x


class GraphSAGE(torch.nn.Module):
  def __init__(self, dim_h, num_features, dim_out=1, end_target='graph'):
    super().__init__()
    self.conv1 = SAGEConv(num_features, dim_h)
    self.conv2 = SAGEConv(dim_h, dim_h)
    self.conv3 = SAGEConv(dim_h, dim_h)
    self.end_target = end_target
    if end_target=='graph' or end_target=='node':
      self.lin = torch.nn.Linear(dim_h, dim_out)
  
  def get_H(self, data):
    e = data.edge_index
    x = data.my_x
    
    x = self.conv1(x, e)
    x = x.relu()
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.conv2(x, e)
    x = x.relu()
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.conv3(x, e)
    return x

  def forward(self, data):
    x = self.get_H(data)
    if self.end_target == 'graph':
      x = global_mean_pool(x, data.batch)
      x = F.dropout(x, p=0.5, training=self.training)
    if self.end_target=='graph' or self.end_target=='node':
      x = self.lin(x)
    return x

class GCN(torch.nn.Module):
  """Graph Convolutional Network class with 3 convolutional layers and a linear layer"""

  def __init__(self, dim_h, num_features, dim_out=1, end_target='graph'):
    """init method for GCN
    
    Args:
        dim_h (int): the dimension of hidden layers
    """
    super().__init__()
    self.conv1 = GCNConv(num_features, dim_h)
    self.conv2 = GCNConv(dim_h, dim_h)
    self.conv3 = GCNConv(dim_h, dim_h)
    self.end_target = end_target
    if end_target=='graph' or end_target=='node':
      self.lin = torch.nn.Linear(dim_h, dim_out)
  
  def get_H(self, data):
    e = data.edge_index
    x = data.my_x
    
    x = self.conv1(x, e)
    x = x.relu()
    x = self.conv2(x, e)
    x = x.relu()
    x = self.conv3(x, e)
    return x

  def forward(self, data):
    x = self.get_H(data)
    if self.end_target == 'graph':
      x = global_mean_pool(x, data.batch)
      x = F.dropout(x, p=0.5, training=self.training)
    if self.end_target=='graph' or self.end_target=='node':
      x = self.lin(x)
    return x

class GIN(torch.nn.Module):
  """Graph Isomorphism Network class with 3 GINConv layers and 2 linear layers"""
  
  def __init__(self, dim_h, num_features, dim_out=1, end_target='graph'):
    """Initializing GIN class
    
    Args:
        dim_h (int): the dimension of hidden layers
    """
    super(GIN, self).__init__()
    self.conv1 = GINConv(
        Sequential(Linear(num_features, dim_h), BatchNorm1d(dim_h), ReLU(), Linear(dim_h, dim_h), ReLU())
    )
    self.conv2 = GINConv(
        Sequential(
            Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(), Linear(dim_h, dim_h), ReLU()
        )
    )
    self.conv3 = GINConv(
        Sequential(
            Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(), Linear(dim_h, dim_h), ReLU()
        )
    )
    self.end_target = end_target
    if end_target=='graph' or end_target=='node':
      self.lin1 = Linear(dim_h, dim_h)
      self.lin2 = Linear(dim_h, dim_out)
  
  def get_H(self, data):
    x = data.my_x
    edge_index = data.edge_index
    
    # Node embeddings
    h = self.conv1(x, edge_index)
    h = h.relu()
    h = self.conv2(h, edge_index)
    h = h.relu()
    h = self.conv3(h, edge_index)
    return h

  def forward(self, data):
    h = self.get_H(data)
    
    if self.end_target == 'graph':
      # Graph-level readout
      batch = data.batch
      h = global_add_pool(h, batch)
      
    if self.end_target=='graph' or self.end_target=='node':
      h = self.lin1(h)
      h = h.relu()
      h = F.dropout(h, p=0.5, training=self.training)
      h = self.lin2(h)
    
    return h
  
def val_loss(model, val_loader, model_to_mimic=None, device='cuda', loss_type='MSE'):
  if loss_type=='MSE':
    loss = torch.nn.MSELoss()
  elif loss_type=='CE':
    loss = torch.nn.CrossEntropyLoss()
  else:
    print(f'Unknown loss_type={loss_type}')
    sys.exit(1)
  
  model.eval()
  model = model.to(device)
  if model_to_mimic is not None:
    model_to_mimic.eval()
    model_to_mimic = model_to_mimic.to(device)
  val_loss = 0
  with torch.no_grad():
    for d in val_loader:
      d = d.to(device)
      if model_to_mimic is None:
        out = model(d)
        if loss_type=='MSE':
          l = loss(out.view(-1,1), d.target.view(-1,1))
        elif loss_type=='CE':
          l = loss(out, d.target)
      else:
        out = model.get_H(d)
        z = model_to_mimic.get_H(d) # if model.end_target=='embed' else model_to_mimic(d)
        l = torch.sum(torch.square(out-z))
      val_loss += l / len(val_loader)
  return val_loss


def train_gnn(epochs, model, train_loader, val_loader, device='cuda', lr=0.001, weight_decay=5e-4, model_to_mimic=None, seed=0, loss_type='MSE', early_stopping_gap=500, verbose=True):
  """Training over all epochs
  
  Args:
      epochs (int): number of epochs to train for
      model (nn.Module): the current model
      train_loader (DataLoader): training data in batches
      val_loader (DataLoader): validation data in batches
      path (string): path to save the best model
  
  Returns:
      array: returning train and validation losses over all epochs, prediction and ground truth values for training data in the last epoch
  """
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

  # Ignored if model_to_mimic is specified
  if loss_type=='MSE':
    loss = torch.nn.MSELoss()
  elif loss_type=='CE':
    loss = torch.nn.CrossEntropyLoss()
  else:
    print(f'Unknown loss_type={loss_type}')
    sys.exit(1)

  torch.manual_seed(seed)
  
  if model_to_mimic is not None:
    model_to_mimic = model_to_mimic.to(device)
    model_to_mimic.eval()
  
  best_val_loss, best_val_loss_epoch = 1e99, -1
  for epoch in (tqdm(range(epochs)) if not verbose else range(epochs)):
    model.train()
    epoch_loss = 0
    for counter, d in enumerate(tqdm(train_loader) if (epoch==0 and verbose) else train_loader):
      d = d.to(device)
      optimizer.zero_grad()
      
      if model_to_mimic is None:
        out = model(d) # n * p
        if loss_type=='MSE':
          if out.shape[0]!=d.target.shape[0]:
            print(f'out.shape[0]={out.shape[0]}, d.target.shape[0]={d.target.shape[0]}!')

          len_out = out.shape[0]
          l = loss(out.view(-1,1), d.target[:len_out].view(-1,1))
        elif loss_type=='CE':
          l = loss(out, d.target)
      else:
        out = model.get_H(d) # n * p
        z = model_to_mimic.get_H(d) # if model.end_target=='embed' else model_to_mimic(d)
        l = torch.sum(torch.square(out-z))
      
      epoch_loss += l / len(train_loader)
      l.backward()
      optimizer.step()

    if (verbose and epoch % 2 == 0) or (epoch==epochs-1):
      model.eval()
      v_loss = val_loss(model, val_loader, model_to_mimic=model_to_mimic, device=device, loss_type=loss_type)
      print(f"Epoch: {epoch}, Train loss: {epoch_loss.item():.4f}, Val loss: {v_loss.item():.4f}")
      if v_loss < best_val_loss:
        best_val_loss, best_val_loss_epoch = v_loss, epoch
      elif epoch > best_val_loss_epoch + early_stopping_gap:
        print(f'Stopping training since val loss has not improved since epoch={best_val_loss_epoch} (best_val_loss={best_val_loss:.4f})')
        break
  return model


def save_model(model, model_name):
  model_file = f'{model_name}.weights.pth'
  model.eval()
  torch.save(model.state_dict(), model_file)
  print(f'Saved {model_file}')

def load_model(model, model_name, verbose=True):
  model_file = f'{model_name}.weights.pth'
  model.eval()
  if os.path.exists(model_file):
    model.load_state_dict(torch.load(model_file, weights_only=True, map_location='cpu'))
    if verbose:
      print(f'Loaded {model_file}')
    return True
  return False


def get_model_name(dataset_name, dataloader_split, model_type, dim_h, bootstrap_num=None, model_to_mimic_name=None):
  model_name = dataset_name
  if model_to_mimic_name is None:
    model_name += '_bm_' if bootstrap_num is None else f'_boot{bootstrap_num}_'
  else:
    model_name += f'_steal_{model_to_mimic_name}_using_'
  model_name += f'{model_type}{dim_h}s{dataloader_split}'
  return model_name

def train_model_wrapper(train_loader, val_loader, model_name=None, model_type='GCN', dim_h=32, dim_out=1, epochs=10, end_target='graph', force_train=False, model_to_mimic=None, **kwargs):
  num_features = train_loader.dataset[0].my_x.shape[1]
  if model_type == 'GCN':
    model = GCN(dim_h=dim_h, dim_out=dim_out, num_features=num_features, end_target=end_target)
  elif model_type == 'GSAGE':
    model = GraphSAGE(dim_h=dim_h, dim_out=dim_out, num_features=num_features, end_target=end_target)
  elif model_type == 'GIN':
    model = GIN(dim_h=dim_h, dim_out=dim_out, num_features=num_features, end_target=end_target)
  elif model_type == 'MIX':
    model = MixHop(dim_h=dim_h, dim_out=dim_out, num_features=num_features, end_target=end_target)
  elif model_type == 'ARMA':
    model = ARMA(dim_h=dim_h, dim_out=dim_out, num_features=num_features, end_target=end_target)
  else:
    print(f'Unknown model_type={model_type}')
    sys.exit(1)

  did_load = load_model(model, model_name) if model_name is not None and not force_train else False

  if not did_load:
    print(f'Training {model_name}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = train_gnn(epochs=epochs, model=model, train_loader=train_loader, val_loader=val_loader, device=device, model_to_mimic=model_to_mimic, **kwargs)
    if model_name:
      save_model(model, model_name)
  return model




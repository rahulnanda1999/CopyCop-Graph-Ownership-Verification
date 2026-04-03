from torch_geometric.datasets import QM9, ZINC, Planetoid, CitationFull, MD17, TUDataset, Reddit2, Flickr, MoleculeNet, AttributedGraphDataset, WikiCS, LRGBDataset, EllipticBitcoinDataset, DGraphFin, HydroNet, NeuroGraphDataset, OGB_MAG, PCQM4Mv2, BrcaTcga, WebKB, HeterophilousGraphDataset, ModelNet, AmazonProducts, Coauthor, Amazon, GNNBenchmarkDataset, Yelp

from torch_geometric.transforms import RadiusGraph, Compose, FaceToEdge
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, degree, to_undirected
import torch

import numpy as np
from pandas import Series, DataFrame
from functools import partial
from tqdm import tqdm

# specify the local data path
#HERE = Path(_dh[-1])
#DATA = HERE / "data"

DATA = 'data'

def setup_data(d, start_idx=0, n_max=30000, standardize=True, batch_size=64, seed=0, do_bootstrap=False):
  n = min(len(d), n_max)
  np.random.seed(seed)
  data_index = np.random.permutation(len(d))
  train_index = data_index[start_idx:start_idx+int(0.9*n)]
  val_index = data_index[start_idx+int(0.9*n):start_idx+n]

  if standardize:
    targets = np.array([d[i].target for i in range(min(1000, len(d)))])
    data_mean, data_std = targets.mean(), targets.std()
    new_d = []
    for i in np.concatenate([train_index, val_index]):
      new_d.append(d[i])
      new_d[-1].target = float((new_d[-1].target - data_mean)/data_std)
      new_d[-1].num_nodes = new_d[-1].my_x.shape[0]
    train_data = [new_d[i] for i in range(len(train_index))]
    val_data = [new_d[len(train_index)+i] for i in range(len(val_index))]
  else:
    train_data = [d[i] for i in train_index]
    val_data = [d[i] for i in val_index]

  if do_bootstrap:
    train_index = np.random.choice(train_index, size=len(train_index), replace=True)

  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
  return train_loader, val_loader

def get_subgraph(data, around_which_node, num_hops=1):

  sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
      node_idx=around_which_node,
      num_hops=num_hops,
      edge_index=data.edge_index,
      relabel_nodes=True
  )

  x = data.x[sub_nodes]
  target = data.y[sub_nodes]
  subgraph = Data(
      my_x=x,
      edge_index=sub_edge_index,
      target=target,
      num_nodes=x.shape[0],
  )

  return subgraph

def create_synthetic_data(data, seed=0, start_idx=0, n_max=30000, batch_size=64, num_repeats=4, standardize=False):
  n = min(len(data), n_max)
  np.random.seed(seed)
  data_index = np.random.permutation(len(data))
  used_index = data_index[start_idx:start_idx+n]

  if standardize:
    num_train = int(0.9*len(used_index))
    data_mean = data.target[:num_train].mean()
    data_std = data.target[:num_train].std()
  else:
    data_mean, data_std = 0, 1

  allx = np.vstack([data[i].my_x for i in used_index])
  all_perms = [np.random.permutation(allx.shape[0]) for i in range(num_repeats)]
  new_data = []
  counter = 0
  for i in used_index:
    d = data[i]
    this_len = d.my_x.shape[0]
    new_data.append(\
        Data(
          x=torch.zeros(this_len, 1),  
          my_x=d.my_x,
          edge_index=d.edge_index,
          target=(d.target-data_mean)/data_std,
          num_nodes=this_len
        ))
    for i in range(num_repeats):
      new_data.append(\
          Data(
            x=torch.zeros(this_len, 1), 
            my_x=torch.tensor(allx[all_perms[i][counter:counter+this_len]]),
            edge_index=d.edge_index,
            target=(d.target-data_mean)/data_std,
            num_nodes=this_len
          ))
    counter += this_len
  
  train_index = np.arange(int(0.9*len(new_data)))
  val_index = np.arange(int(0.9*len(new_data)), len(new_data))

  train_loader = DataLoader([new_data[i] for i in train_index], batch_size=batch_size, shuffle=True)
  val_loader = DataLoader([new_data[i] for i in val_index], batch_size=batch_size, shuffle=True)
  return train_loader, val_loader


def get_coeffs_x_mask(data):
  coeffs_x_mask = np.zeros(data[0].my_x.shape[1], dtype=bool)
  for i in range(min(len(data), 100)):
    tmp = data[i].my_x
    coeffs_x_mask = coeffs_x_mask | ((tmp == tmp.int()).sum(dim=0) < len(tmp)).detach().cpu().numpy()
  return coeffs_x_mask

def qm9_add_target(data, which_x='pos'):
  data.target = data.y[:,0]
  data.my_x = data.pos if which_x=='pos' else data.x
  return data

def get_data_QM9(which_x='pos'):
  this_transform = partial(qm9_add_target, which_x=which_x)
  qm9 = QM9(root=DATA, transform=this_transform)
  qm9._data.target = qm9._data.y[:,0]
  qm9._data.my_x = qm9._data.pos if which_x=='pos' else qm9._data.x
  return qm9

def generic_add_target(data):
  data.target = data.y[:,0]
  data.my_x = data.x.float()
  return data

def generic_add_target2(data):
  data.target = data.y
  data.my_x = data.x.float()
  return data


def get_data_clintox():
  this_transform = bbbp_add_target
  clintox = MoleculeNet('data/clintox', 'ClinTox', transform=this_transform)
  clintox._data.target = clintox._data.y[:,0]
  clintox._data.my_x = clintox._data.x
  return clintox

def bbbp_add_target(data):
  data.target = data.y[0][0].long()
  data.my_x = data.x.float()
  return data

def get_data_bbbp():
  this_transform = bbbp_add_target
  bbbp = MoleculeNet('data/bbbp', 'BBBP', transform=this_transform)
  bbbp._data.target = bbbp._data.y[:,0]
  bbbp._data.my_x = bbbp._data.x
  return bbbp

def get_data_esol():
  this_transform = generic_add_target
  esol = MoleculeNet('data/esol', 'ESOL', transform=this_transform)
  esol._data.target = esol._data.y[:,0]
  esol._data.my_x = esol._data.x
  return esol

def get_data_hiv():
  this_transform = bbbp_add_target
  hiv = MoleculeNet('data/hiv', 'HIV', transform=this_transform)
  hiv._data.target = hiv._data.y[:,0]
  hiv._data.my_x = hiv._data.x
  return hiv

def get_data_pubmed(num_hops=1):
  data = Planetoid(root='data/pubmed', name='PubMed')[0]
  subgraphs = [get_subgraph(data, around_which_node=node, num_hops=num_hops) for node in range(data.x.shape[0])] # build subgraphs around each node
  return subgraphs

def get_data_dblp(num_hops=1):
  data = CitationFull(root='data/dblp', name='DBLP')[0]
  subgraphs = [get_subgraph(data, around_which_node=node, num_hops=num_hops) for node in range(data.x.shape[0])] # build subgraphs around each node
  return subgraphs

def get_data_citeseer(num_hops=1):
  data = CitationFull(root='data/citeseer', name='CiteSeer')[0]
  subgraphs = [get_subgraph(data, around_which_node=node, num_hops=num_hops) for node in range(data.x.shape[0])] # build subgraphs around each node
  return subgraphs

def get_data_cora(num_hops=1):
  data = CitationFull(root='data/cora', name='Cora')[0]
  subgraphs = [get_subgraph(data, around_which_node=node, num_hops=num_hops) for node in range(data.x.shape[0])] # build subgraphs around each node
  return subgraphs

def get_data_reddit2(num_hops=1):
  data = Reddit2(root='data/reddit2')[0]
  subgraphs = [get_subgraph(data, around_which_node=node, num_hops=num_hops) for node in range(data.x.shape[0])] # build subgraphs around each node
  return subgraphs

def get_data_flickr(num_hops=1):
  data = Flickr(root='data/flickr')[0]
  subgraphs = [get_subgraph(data, around_which_node=node, num_hops=num_hops) for node in tqdm(range(data.x.shape[0]))] # build subgraphs around each node
  return subgraphs

def get_data_ppi(num_hops=1):
  data = AttributedGraphDataset('data/ppi', 'PPI')[0]  
  subgraphs = [get_subgraph(data, around_which_node=node, num_hops=num_hops) for node in range(data.x.shape[0])] # build subgraphs around each node
  return subgraphs

def get_data_blogcatalog(num_hops=1):
  data = AttributedGraphDataset('data/blogcatalog', 'BlogCatalog')[0]  
  subgraphs = [get_subgraph(data, around_which_node=node, num_hops=num_hops) for node in range(data.x.shape[0])] # build subgraphs around each node
  return subgraphs

def get_data_tweibo(num_hops=2):
  data = AttributedGraphDataset('data/tweibo', 'TWeibo')[0] 
  mask = (data.edge_index[0] >= len(data.y)) | (data.edge_index[1] >= len(data.y))
  data.edge_index = data.edge_index[:, ~mask]

  np.random.seed(0)
  node_indices = np.random.choice(len(data.y), 5000, replace=False)
  subgraphs = [get_subgraph(data, around_which_node=int(node), num_hops=num_hops) for node in tqdm(node_indices)] # build subgraphs around each node
  return subgraphs

def get_data_wikics(num_hops=1):
  data = WikiCS('data/wikics')[0]
  subgraphs = [get_subgraph(data, around_which_node=node, num_hops=num_hops) for node in tqdm(range(data.x.shape[0]))] # build subgraphs around each node
  return subgraphs

def get_data_bitcoin(num_hops=1):
  data = EllipticBitcoinDataset('data/bitcoin')[0]
  data.edge_index = to_undirected(data.edge_index)

  node_indices = Series(degree(data.edge_index[0], num_nodes=data.num_nodes)).sort_values(ascending=False).index.values[:2000]
  subgraphs = [get_subgraph(data, around_which_node=int(node), num_hops=num_hops) for node in tqdm(node_indices)] # build subgraphs around each node
  return subgraphs

def get_data_fin(num_hops=1, howmany=10000):
  data = DGraphFin('data/fin')[0]
  data.edge_index = to_undirected(data.edge_index)
  s = Series(degree(data.edge_index[0], num_nodes=data.num_nodes)).sort_values(ascending=False)
  node_indices = s.index.values[:howmany]
  subgraphs = [get_subgraph(data, around_which_node=int(node), num_hops=num_hops) for node in tqdm(node_indices)] # build subgraphs around each node
  return subgraphs

def hydro_add_target(data):
  data.target = data.y
  data.my_x = data.pos.float()
  return data

def get_data_hydro():
  this_transform = hydro_add_target
  hydro = HydroNet(root='data/hydro', name='small', transform=Compose([this_transform, RadiusGraph(r=3)]))
  return hydro

def get_data_hcp():
  this_transform = generic_add_target2
  hcp = NeuroGraphDataset(root='data/hcp', name='HCPTask', transform=this_transform)
  return hcp


def peptides_add_target(data):
  data.target = torch.nonzero(data.y[0], as_tuple=False)[0][0].long()
  data.my_x = data.x.float()
  return data

def get_data_peptides():
  peptides = LRGBDataset('data/lrgb_peptides', 'Peptides-func', transform=peptides_add_target)
  return peptides

def get_data_pascal():
  pascal = LRGBDataset('data/pascal', 'PascalVOC-SP', transform=generic_add_target2)
  return pascal

def aspirin_add_target(data, which_x='pos'):
  data.target = data.energy
  data.my_x = data.pos
  return data

def get_data_aspirin():
  # Using 6 angstroms:
  #   J. Gasteiger, J. Groß, and S. Günnemann. Directional message passing for molecular graphs. In ICLR, 2020.
  #   N. Thomas, T. Smidt, S. Kearnes, L. Yang, L. Li, K. Kohlhoff, and P. Riley. Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds. Preprint arXiv:1802.08219, 2018.
  #   K. Schütt, P.-J. Kindermans, H. E. Sauceda Felix, S. Chmiela, A. Tkatchenko, and K.-R. Müller.  Schnet: A continuous-filter convolutional neural network for modeling quantum interactions.  Advances in neural information processing systems, 30, 2017. 
  # Between 6 and 10 angstrom:
  #   "This cutoff radius, which typically has a value between 6 and 10 A˚, is a convergence parameter and needs to be tested to ensure that all energetically relevant interactions are included": J6rg Behler. Constructing high-dimensional neural network potentials: A tutorial review. International Journal of Quantum Chemistry, 115(16):1032-1050, 2015.

  aspirin = MD17('data/aspirin', 'aspirin', transform=Compose([aspirin_add_target, RadiusGraph(r=6)]))  # FIXME fix r
  aspirin._data.target = aspirin._data.energy
  aspirin._data.my_x = aspirin._data.pos
  return aspirin

def get_data_benzene():
  benzene = MD17('data/benzene', 'benzene CCSD(T)', train=True, transform=Compose([aspirin_add_target, RadiusGraph(r=6)]))  # FIXME fix r
  benzene._data.target = benzene._data.energy
  benzene._data.my_x = benzene._data.pos
  return benzene


def enzymes_add_target(data):
  data.target = data.y
  data.my_x = data.x
  return data

def get_data_enzymes():
  enzymes = TUDataset('data/enzymes', 'ENZYMES', use_node_attr=True, transform=enzymes_add_target)
  enzymes._data.target = enzymes._data.y
  enzymes._data.my_x = enzymes._data.x
  return enzymes

def proteins_add_target(data):
  data.target = data.y
  data.my_x = data.x[:,:-3] # last 3 features are ints
  return data

def get_data_proteins():
  proteins = TUDataset('data/proteinsFull', 'PROTEINS_full', use_node_attr=True, transform=proteins_add_target)
  return proteins

def get_data_ogbmag(num_hops=1):
  orig_data = OGB_MAG('data/ogbmag')[0]
  data = Data(x=orig_data['paper']['x'], y=orig_data['paper']['y'], edge_index=to_undirected(orig_data['paper', 'cites', 'paper']['edge_index']))
  s = Series(degree(data.edge_index[0], num_nodes=data.num_nodes)).sort_values(ascending=False)
  mask = ((s<=100) & (s>=60))
  node_indices = s[mask].index.values
  subgraphs = [get_subgraph(data, around_which_node=int(node), num_hops=num_hops) for node in tqdm(node_indices)] # build subgraphs around each node
  return subgraphs

def get_data_pcqm2():
  pcqm2 = PCQM4Mv2('data/pcqm2', transform=generic_add_target2)
  return pcqm2

def get_data_brca():
  brca = BrcaTcga('data/brca', transform=generic_add_target2)
  return brca

def get_data_coco():
  coco = LRGBDataset('data/coco', 'COCO-SP', transform=generic_add_target2)
  return coco

def get_data_webkb(num_hops=3):
  data = WebKB('data/webkb', 'Wisconsin')[0]
  data.edge_index = to_undirected(data.edge_index)

  node_indices = np.arange(data.x.shape[0])
  subgraphs = [get_subgraph(data, around_which_node=int(node), num_hops=num_hops) for node in tqdm(node_indices)] # build subgraphs around each node
  return subgraphs

def get_data_amazon(num_hops=3):
  data = HeterophilousGraphDataset('data/amazon', 'Amazon-ratings')
  data.edge_index = to_undirected(data.edge_index)

  node_indices = np.arange(data.x.shape[0])
  subgraphs = [get_subgraph(data, around_which_node=int(node), num_hops=num_hops) for node in tqdm(node_indices)] # build subgraphs around each node
  return subgraphs

def model10_add_target(data):
  data.target = data.y
  data.my_x = data.pos
  return data

def get_data_model10():
  model10 = ModelNet('data/model10', '10', pre_transform=FaceToEdge(), transform=model10_add_target)
  return model10

def amazonprod_add_target(data):
  data.target = data.y[:,97]  # 29% rows turned on
  data.my_x = data.x
  return data

def get_data_amazonprod(num_hops=1):
  data = AmazonProducts('data/amazonprod', transform=amazonprod_add_target)[0]
  s = Series(degree(data.edge_index[0], num_nodes=data.num_nodes)).sort_values(ascending=False)
  mask = ((s<=51) & (s>=50))
  node_indices = s[mask].index.values
  subgraphs = [get_subgraph(data, around_which_node=int(node), num_hops=num_hops) for node in tqdm(node_indices)] # build subgraphs around each node
  return subgraphs

def get_data_physics(num_hops=1):
  data = Coauthor('data/physics', 'Physics')[0]
  node_indices = np.arange(data.x.shape[0])
  subgraphs = [get_subgraph(data, around_which_node=int(node), num_hops=num_hops) for node in tqdm(node_indices)] # build subgraphs around each node
  return subgraphs

def get_data_amazoncomp(num_hops=1):
  data = Amazon('data/amazoncomp', 'Computers')[0]
  node_indices = np.arange(data.x.shape[0])
  subgraphs = [get_subgraph(data, around_which_node=int(node), num_hops=num_hops) for node in tqdm(node_indices)] # build subgraphs around each node
  return subgraphs

def get_data_cifar():
  cifar = GNNBenchmarkDataset('data/cifar', 'CIFAR10', transform=model10_add_target)
  return cifar

def get_data_mnist():
  mnist = GNNBenchmarkDataset('data/mnist', 'MNIST', transform=model10_add_target)
  return mnist

def get_data_yelp(num_hops=1, howmany=2000):
  data = Yelp('data/yelp')[0]
  data.edge_index = to_undirected(data.edge_index)
  s = Series(degree(data.edge_index[0], num_nodes=data.num_nodes)).sort_values(ascending=False)
  mask = ((s<=60) & (s>=50))
  node_indices = s[mask].index.values
  subgraphs = [get_subgraph(data, around_which_node=int(node), num_hops=num_hops) for node in tqdm(node_indices)] # build subgraphs around each node
  return subgraphs

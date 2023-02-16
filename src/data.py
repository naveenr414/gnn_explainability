from dgl.data import BACommunityDataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.nn import GCNConv, Set2Set, GNNExplainer
import torch_geometric.transforms as T
import torch

import os

def get_dataset(dataset_name):
    """Return one of BACommunity or another dataset from DGL
    
    Arguments:
        dataset_name: String, which is one of 'BACommunity'
    
    Returns: dataset from dgl.data
    """
    
    name_to_class = {'BACommunity': BACommunityDataset}
    
    return name_to_class[dataset_name]()[0]

def get_cora_dataset():
    """Return the cora dataset, setting up the test and validation mask appropriately
    
    Arguments: None 
    
    Returns: The CORA dataset
    """
    
    dataset = 'cora'
    path = os.path.join(os.getcwd(), 'data', 'Planetoid')
    train_dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())

    data = train_dataset[0]

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[:data.num_nodes - 1000] = 1
    data.val_mask = None
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[data.num_nodes - 500:] = 1
    
    return train_dataset,data
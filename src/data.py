from dgl.data import BACommunityDataset, BAShapeDataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.nn import GCNConv, Set2Set, GNNExplainer
import torch_geometric.transforms as T
import torch
import numpy as np

import os

def get_dataset(dataset_name):
    """Return one of BACommunity or another dataset from DGL
    
    Arguments:
        dataset_name: String, which is one of 'BACommunity', 'BAShape'
    
    Returns: dataset from dgl.data
    """
    
    name_to_class = {'BACommunity': BACommunityDataset, 'BAShapes': BAShapeDataset}
    
    dataset = name_to_class[dataset_name]()
    data_list = []
    for i in range(len(dataset)):
        graph = dataset[i]
        label = graph.ndata['label']
        x = graph.ndata['feat']
        edge_index = torch.stack(graph.edges())
        data = Data(x=x, edge_index=edge_index.contiguous(), y=label)
        data_list.append(data)

    # Load the dataset using PyG's DataLoader
    loader = DataLoader(data_list, batch_size=32, shuffle=True)

    for i in loader:
        batch = i
    
    num_nodes = batch.x.shape[0]
    
    train_nums = int(0.8*num_nodes)
    non_train_indices = num_nodes-train_nums
    
    train_mask = [1 for i in range(train_nums)] + [0 for i in range(non_train_indices)]
    test_mask = [0 for i in range(train_nums)] + [1 for i in range(non_train_indices)]
    
    
    batch.train_mask = torch.Tensor(np.array(train_mask)).bool()
    batch.test_mask = torch.Tensor(np.array(test_mask)).bool()
    
    batch.x = batch.x.float()

    
    return batch

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
    
    return data
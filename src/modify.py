import torch
import random
import numpy as np
def identity(data):
    """Identity function for a data point; returns the data itself
    
    Arguments:
        data: PyTorch dataset
        
    Returns: PyTorch dataset, unmodified
    """
    
    return data


def aggressive_adversary(data, frac):
    """Aggressive adversary for data; returns the noisy data

    As follows the paper "Are Graph Neural Network Explainers
    Robust to Graph Noises?", we randomly choose nodes from the
    set V, then generate edges among them with probability 0.1

    Arguments:
        data: PyTorch dataset
        frac_nodes: Percent of nodes in the graph to modify

    Returns: PyTorch dataset, modified
    """
    vertices = set(torch.flatten(data.edge_index).tolist())

    frac_nodes = random.sample(vertices, int(len(vertices) * frac))

    # loop through nodes and randomly generate edge with prob = 0.1
    for i, node_i in enumerate(frac_nodes):
        for j, node_j in enumerate(frac_nodes):

            if i < j and random.random() < 0.1:
                data.edge_index = np.append(data.edge_index, [[node_i], [node_j]], axis=1)

    data.edge_index = torch.Tensor(data.edge_index).long()

    return data


def conservative_adversary(data, name, frac):
    """Conservative adversary for data; returns the noisy data

    As follows the paper "Are Graph Neural Network Explainers
    Robust to Graph Noises?", we randomly choose nodes from the
    set of unimportant noddes, then generate edges among them with
    probability 0.1

    Arguments:
        data: PyTorch dataset
        name: String name of dataset
        frac_nodes: Percent of nodes to modify

    Returns: PyTorch dataset, modified
    """

    vertices = unimportant_nodes(data, name)

    frac_nodes = random.sample(vertices, int(len(vertices) * frac))

    # loop through nodes and randomly generate edge with prob = 0.1
    for i, node_i in enumerate(frac_nodes):
        for j, node_j in enumerate(frac_nodes):

            if i < j and random.random() < 0.1:
                data.edge_index = np.append(data.edge_index, [[node_i], [node_j]], axis=1)

    data.edge_index = torch.Tensor(data.edge_index).long()

    return data


def unimportant_nodes(data, name):
    """
    Identifies important nodes based on dataset structure.
    BA Shapes - important nodes are the house

    Arguments:
        data: PyTorch dataset
        name: String name of dataset

    Returns: Set of unimportant vertices
    """

    if name == "BAShapes":

        #get indices of nodes of class 0 (nodes not in house motif)
        mask = data.y == 0
        vertices = np.where(mask)[0].tolist()

    if name == "TreeCycles":
        pass

    if name == " Mutagenicity":
        pass

    return vertices

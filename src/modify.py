import torch
import random
import numpy as np
from copy import deepcopy

def identity(data):
    """Identity function for a data point; returns the data itself
    
    Arguments:
        data: PyTorch dataset
        
    Returns: PyTorch dataset, unmodified
    """
    
    return data

def get_adjacency_power(adjacency,degree,power,row):
    """Compute the adjacency matrix to a certain power, weighted by a diagonal matrix degree
    
    Arguments:
        adjacency: NxN adjacency matrix
        degree: NxN diagonal matrix
        power: K, what power to raise the matrix (DAD) to
        row: Which row we care about computing

    Returns:
        (DAD)^K[row]
        
    Time Complexity: O(N^2*K) -> O(EK) ?-> O(two_hop + three_hop...)
        Run a BFS First
    """
    
    num_nodes = len(adjacency)
    dist = [adjacency[row,i]*degree[row,row]*degree[i,i] for i in range(num_nodes)]
    
    for i in range(power-1):
        new_dist = deepcopy(dist)
        
        for j in range(len(new_dist)):
            new_dist[j] = sum([adjacency[i,j]*dist[i]*degree[i,i]*degree[j,j] for i in range(num_nodes)])
            
        dist = new_dist
        
    return np.array(dist)

def get_movement_direction(kmeans_centers,current_prediction,current_location):
    """Get the direction of movement away from a set of kmeans centers, given an activation
    
    Arguments:
        kmeans_centers: Numpy array of kmeans centers
        current_prediction: Which cluster the point is going away from
        current_location: What the current activation is
    
    Returns: numpy array/vector of the direction to travel
        Based on Columb's Law
    """
    
    vec = np.zeros((len(current_location)))
    
    for i in range(len(kmeans_centers)):
        diff = (current_location-kmeans_centers[i])/(np.linalg.norm(kmeans_centers[i]-current_location)**2)
        if i == current_prediction:
            diff *= -1
            
        vec += diff

    return vec 

def targeted_attack(dataset,gce_explainer,model,targeted_node,budget):
    """Targeted attack on GNN based on a budget, 
        Based on https://arxiv.org/abs/1805.07984
        
    Arguments:
        data: PyTorch dataset
        gc_explainer: GC Explainer class object, which has KMeans centers,
            activations 
        targeted_node: Which node we're aiming to target
        budget: Number of modifications to the adjacecny matrix
            we're allowed to make 
    
    Returns: PyTorch dataset, modified"""
    
    W_1 = model.conv1.lin.weight.T
    W_2 = model.conv2.lin.weight.T
    W_3 = model.conv3.lin.weight.T
    
    num_nodes = dataset.num_nodes
    A = np.zeros((num_nodes,num_nodes))
    
    for a,b in dataset.edge_index.T:
        A[a,b] = A[b,a] = 1
        
    identity = np.eye(num_nodes)
    A_squiggle = A+identity
    
    D_squiggle = np.diag(np.sum(A_squiggle, axis=1))
    X = dataset.x

    D_half = np.zeros(D_squiggle.shape)
    np.fill_diagonal(D_half, 1/ (D_squiggle.diagonal()**0.5))
    
    A_hat = (D_half).dot(A_squiggle).dot(D_half)
    
    A_hat_cubed = A_hat.dot(A_hat.dot(A_hat))
        
    return dataset
   

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

    if name == "bashapes":

        #get indices of nodes of class 0 (nodes not in house motif)
        mask = data.y == 0
        vertices = np.where(mask)[0].tolist()

    if name == "bacommunity":
        # get indices of nodes of class 0 or 4 (nodes not in house motif)
        # (BA community is 2 BA shapes graphs appended)

        mask = (data.y == 0) | (data.y == 4)
        vertices = np.where(mask)[0].tolist()

    if name == "TreeCycles":
        pass

    if name == " Mutagenicity":
        pass

    return vertices

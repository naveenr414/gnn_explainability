import torch
import random
import numpy as np
from copy import deepcopy, copy
import time
from src.util import *

def identity(data):
    """Identity function for a data point; returns the data itself
    
    Arguments:
        data: PyTorch dataset
        
    Returns: PyTorch dataset, unmodified
    """
    
    return data

def get_adjacency_power(adjacency,degree,power,row,three_hop_nodes):
    """Compute the adjacency matrix to a certain power, weighted by a diagonal matrix degree
    
    Arguments:
        adjacency: NxN adjacency matrix
        degree: NxN diagonal matrix
        power: K, what power to raise the matrix (DAD) to
        row: Which row we care about computing

    Returns:
        (DAD)^K[row]
        
    Time Complexity: O(N^2*K) -> O(EK) ?-> O(two_hop + three_hop...)
    """
    
    num_nodes = len(adjacency)
    dist = [adjacency[row,i]*degree[row,row]*degree[i,i] for i in range(num_nodes)]
    
    for i in range(power-1):
        new_dist = deepcopy(dist)
        
        for j in three_hop_nodes:
            new_dist[j] = sum([adjacency[i,j]*dist[i]*degree[i,i]*degree[j,j] for i in three_hop_nodes])
            
        dist = new_dist
        
    return np.array(dist)

def get_movement_direction_coloumb(kmeans_centers,current_location):
    """Get the direction of movement away from a set of kmeans centers, given an activation
    
    Arguments:
        kmeans_centers: Numpy array of kmeans centers
        current_prediction: Which cluster the point is going away from
        current_location: What the current activation is
    
    Returns: numpy array/vector of the direction to travel
        Based on Columb's Law
    """
    
    
    distances = get_kmeans_distances(kmeans_centers,current_location)
    current_prediction = np.argmin(distances)
    
    vec = np.zeros((len(current_location)))
    
    for i in range(len(kmeans_centers)):
        diff = (kmeans_centers[i]-current_location)/(np.linalg.norm(kmeans_centers[i]-current_location)**2)
        if i == current_prediction:
            diff *= -1
            
        vec += diff

    return vec 

def get_movement_direction_minmax(kmeans_centers,current_location):
    """Get the direction of movement away from a set of kmeans centers, given an activation
    
    Arguments:
        kmeans_centers: Numpy array of kmeans centers
        current_prediction: Which cluster the point is going away from
        current_location: What the current activation is
    
    Returns: numpy array/vector of the direction to travel
        Find the closest center and go towards that
    """
    
    distances = get_kmeans_distances(kmeans_centers,current_location)    
    distances[np.argmin(distances)] = np.max(distances)+1
    
    min_center = np.argmin(distances)
    vec = kmeans_centers[min_center] - current_location
    
    return vec 

def recompute_multiplication_A(AXW,XW,A_change,A_i,A_j,targeted_node):
    """Compute the product of AXW when modifying A[i,j] by a value A_change
        That is A[i,j] += A_change
        This comes from the fact that 
        (AXW)_{i,j} = \sum_k A_{i,k} (XW)_{k,j}
        If i = A_i and k=A_j, then only j varies 
        
    Arguments:
        AXW: Matrix of size nxd
        XW: Matrix of size nxd
        A_change: float that represents the change in A
        A_i: Row index of where A is being changed
        A_j: Column index of where A is being changed

    Returns:
        Change in the AXW matrix at the targeted node
    """
    
    if targeted_node != A_i:
        return np.zeros(AXW.shape)[targeted_node]
    
    new_AXW = np.zeros(AXW.shape)
    
    for j in range(AXW.shape[-1]):
        new_AXW[A_i,j] += A_change*XW[A_j,j]
        
    return new_AXW[targeted_node]

def recompute_multiplication_X(AXW,A,X,W,X_change,X_i,X_j,targeted_node):
    """Compute the product of AXW when modifying A[i,j] by a value A_change
        That is A[i,j] += A_change
        
        The computation comes from AXW_{i,j} = \sum_{k} A_{i,k} \sum_{r} X_{k,r} W_{r,j}
        This simplifies when only k = X_i and r=X_j are changed
        
    Arguments:
        AXW: Matrix of size nxd
        XW: Matrix of size nxd
        A_change: float that represents the change in A
        A_i: Row index of where A is being changed
        A_j: Column index of where A is being changed

    Returns:
        New AXW matrix
    """
    
    new_AXW = np.zeros(AXW.shape)
    
    for j in range(AXW.shape[-1]):
        new_AXW[targeted_node,j] += X_change*A[targeted_node,X_i]*W[X_j,j]
                
    return new_AXW[targeted_node]
    
def to_unit_vector(v):
    """Given a numpy vector, v, turn it into a unit vector
    
    Arguments:
        v: Numpy vector
        
    Returns: Normalized vector v
    """
    
    if np.linalg.norm(v) == 0:
        return v
    
    return v/np.linalg.norm(v)

def targeted_attack(dataset,gce_explainer,model,targeted_node,budget,
                   score_function, target_direction_function):
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
    
    kmeans_centers = gce_explainer.kmeans.cluster_centers_
    num_nodes = dataset.num_nodes
    node_features = dataset.x.shape[-1]
    num_gcn_layers = sum(1 for module in model.modules() if isinstance(module, GCNConv))
    
    A = np.zeros((num_nodes,num_nodes))
    for a,b in dataset.edge_index.T:
        A[a,b] = A[b,a] = 1
    A_identity = np.eye(num_nodes)
    A_squiggle = A + A_identity
    
    X = copy(dataset.x.detach().numpy())
    W = compute_W(model)

    AXW = compute_AXW(compute_A_hat(A_squiggle),X,W,num_gcn_layers)
    
    distances = get_kmeans_distances(kmeans_centers,AXW[targeted_node])
    initial_prediction = np.argmin(distances)    
    changes = []
    
    for i in range(budget):
        best_A = (0,0,0)
        best_X = (0,0,0)
        
        D_half = compute_D_half(A_squiggle)
        XW = X.dot(W)
        A_hat = compute_A_hat(A_squiggle)
        A_hat_pow = np.linalg.matrix_power(A_hat,num_gcn_layers)
        AXW = compute_AXW(A_hat,X,W,num_gcn_layers)
        current_location = AXW[targeted_node]
        distances = get_kmeans_distances(kmeans_centers,current_location)
        print("Distances {}".format(distances))
        current_prediction = np.argmin(distances)
        
        if current_prediction != initial_prediction:
            break
        
        target_direction = target_direction_function(kmeans_centers,
                                                  current_location)
        
        visitable_nodes = get_n_hop(A_squiggle,targeted_node,num_gcn_layers)
        modifications = []
        
        # Try modifyng each node feature
        for X_i in range(num_nodes):
            for X_j in range(node_features):
                X_change = 1 if X[X_i,X_j]<1 else -1
                change_direction = recompute_multiplication_X(AXW,A_hat_pow,X,W,X_change,X_i,X_j,targeted_node)
                score = score_function(change_direction+current_location,kmeans_centers,current_prediction)            
                modifications.append((score,"X",X_i,X_j,X_change))
                                    
        for j in visitable_nodes:
            for k in visitable_nodes:
                if j<=k:
                    continue
                    
                new_A_squiggle = copy(A_squiggle)
                new_A_squiggle[j,k] = 1-new_A_squiggle[j,k] 
                
                # TODO: Fix the re-computation of A
                """new_A_hat_pow = get_adjacency_power(new_A_squiggle,D_half,num_gcn_layers,targeted_node,visitable_nodes)
                differences = new_A_hat_pow - A_hat_pow
                diff_vector = np.zeros(len(target_direction))
                
                non_zero_indices = np.nonzero(differences[targeted_node])[0]
                A_i = targeted_node
                                
                for A_j in non_zero_indices:
                    A_change = differences[A_i,A_j]
                    diff_vector += recompute_multiplication_A(AXW,XW,A_change,A_i,A_j,targeted_node)"""
                
                AXW = compute_AXW(compute_A_hat(new_A_squiggle),X,W,num_gcn_layers)
                                        
                score = score_function(AXW[targeted_node],kmeans_centers,current_prediction)
                modifications.append((score,"A",j,k,new_A_squiggle[j,k]))
        
        max_score = max(modifications,key=lambda k: k[0])
        print("Max score {}".format(max_score))
        changes.append(max_score[1:])
        if max_score[1] == "A":
            A_squiggle[max_score[2]][max_score[3]] = max_score[4]
        else:
            X[max_score[2]][max_score[3]] = max_score[4]
    
        
    return changes
   

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

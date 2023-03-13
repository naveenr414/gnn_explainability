import numpy as np
from copy import copy
from collections import deque
from torch_geometric.nn import GCNConv

def compute_AXW(A_hat,X,W,L):
    """Compute A~^L*X*W, which is the matrix representing GNNs
    
    Arguments:
        A_hat: Numpy modified adjacency matrix D^-1/2(A+I)D^-1/2 of size NxN
        X: Feature matrix of size NxK
        W: Product of Weight Matrices of size KxD

    Returns:
        Product A^L*X*W of size NxD
    """
    
    A_pow = np.linalg.matrix_power(A_hat,L)
    return A_pow.dot(X).dot(W)

def compute_W(model):
    """For a model, compute the product of the weight matrices for all conv layers
    
    Arguments:
        model: PyTorch model
        
    Returns: PyTorch tensor representing W_1.dot(W_2).dot(...)
    """
    
    W = model.conv1.lin.weight.detach().numpy().T
    num_gcn_layers = sum(1 for module in model.modules() if isinstance(module, GCNConv))
    
    for i in range(2,num_gcn_layers+1):
        W_i = getattr(model,"conv{}".format(i)).lin.weight.detach().numpy().T
        W = W.dot(W_i)
    return W

def compute_D_half(A_squiggle):
    """From A~, compute D^(-1/2)
    
    Arguments:
        A_squiggle: Numpy matrix
        
    Returns: Numpy matrix, D^(-1/2)
    """
    
    D_squiggle = np.diag(np.sum(A_squiggle, axis=1))
    D_half = np.zeros(D_squiggle.shape)
    np.fill_diagonal(D_half, 1/ (D_squiggle.diagonal()**0.5))
    
    return D_half
    
def compute_A_hat(A_squiggle):
    """From A~, compute A^ = D^(-1/2)*A~*D^(-1/2)
    
    Arguments:
        A_squiggle: PyTorch tensor
        
    Returns: A~, PyTorch Tensor
    """
    
    D_half = compute_D_half(A_squiggle)
    A_hat = D_half.dot(A_squiggle).dot(D_half)
    
    return A_hat
    
    
def compute_AXW_simple(model,dataset):
    """Compute the product of A, X, and W from a model + dataset
    
    Arguments:
        model: PyTorch geometric model
        dataset: Dataloader for a geometric dataset
    
    Returns: PyTorch Tensor AXW
    """
    
    num_nodes = dataset.num_nodes
    num_features = dataset.num_features
    num_gcn_layers = sum(1 for module in model.modules() if isinstance(module, GCNConv))

    A = np.zeros((num_nodes,num_nodes))
    for a,b in dataset.edge_index.T:
        A[a,b] = A[b,a] = 1
    A_identity = np.eye(num_nodes)
    A_squiggle = A + A_identity
    A_hat = compute_A_hat(A_squiggle)
    
    X = dataset.x.detach().numpy()
    W = compute_W(model)
    
    return compute_AXW(A_hat,X,W,num_gcn_layers)
    
def get_n_hop(A,node_idx,n_hop):
    """Get the n-hop neighbors for node_idx, given an Adjacency list A
    
    Arguments:
        A: Adjacency matrix, as numpy array
        node_idx: Which node to find neighbors for
        n_hop: Maximum distance we're searching
        
    Returns: List of n_hop neighbors
    """
    
    visited = [False] * len(A)
    queue = deque([(node_idx, 0)])
    nodes_within_n_hop = []
    
    while queue:
        curr_node, curr_hop = queue.popleft()
        visited[curr_node] = True
        
        if curr_hop <= n_hop:
            nodes_within_n_hop.append(curr_node)
        
        if curr_hop < n_hop:
            for neighbor, is_adjacent in enumerate(A[curr_node]):
                if is_adjacent and not visited[neighbor]:
                    queue.append((neighbor, curr_hop+1))
    
    return nodes_within_n_hop
    
def get_kmeans_distances(centers,point):
    """Find the distances to each point from KMeans centers
    
    Arguments:
        centers: numpy array of centers
        point: One point in space, numpy array
        
    Returns: Distances to each center
    """
    
    return np.linalg.norm(centers - point, axis=1)
    
def closest_kmean_center(centers,point):
    """Given a numpy array with KMeans centers, find the closest center to a point
    
    Arguments:
        centers: numpy array of centers
        point: One point in space, numpy array
        
    Returns: Index of the closest center
    """
    
    distances = get_kmeans_distances(centers,point)
    return np.argmin(distances)

def modify_A_X(changes,A_squiggle,X):
    """Based on changes suggested by the targeted adversarial algorithm
        make changes to the A_squiggle and X matrices
        
    Arguments:
        changes: Output from the targeted attack
        A_squiggle: A~ matrix, modified adjacency
        X: X matrix, of features
    
    Returns: New A_squiggle and X"""
    
    A_squiggle_new = copy(A_squiggle)
    X_new = copy(X)

    for matrix,i,j,value in changes:
        if matrix == 'A':
            A_squiggle_new[i,j] = value
        elif matrix == 'X':
            X_new[i,j] = value

    return A_squiggle_new, X_new
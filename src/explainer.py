import torch
from sklearn.cluster import KMeans

def explain_model(explainer,data,node_idx):
    """Run an explainer from PyTorch using some node from a dataset
    
    Arguments: 
        explainer: PyTorch explainer object
        dataset: PyTorch dataset, such as cora
        node_idx: Specific node to explain
        
    Returns: 
        Node and edge masks that explain which nodes/edges are important
    """
    
    x, edge_index = data.x, data.edge_index
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
    
    return node_feat_mask,edge_mask

def add_hook_model(model,layer_name):
    """Add a hook to the model so we can retrieve activations from a layer later
    
    Arguments:
        model: torch_geometric model
        layer_name: String, such as conv2, representing which layer we want activations for

    Returns: 
        hook handle for the model and activations, an empty list of activations
    """
    
    def get_activation_hook(activations):
        def hook(model, input, output):
            activations.append(output.detach())
        return hook
    
    activations = []
    hook_handle = getattr(model,layer_name).register_forward_hook(get_activation_hook(activations))
    
    return hook_handle, activations

def cluster_activations(activations,num_clusters=4):
    """Perform KMeans on the activations from a layer in a GNN
    
    Arguments:
        activations: A list of torch tensors
        
    Returns: 
        kmeans, a KMeans clustering object from sklearn
    """
    
    train_activations = torch.cat(activations, dim=0)

    # Flatten the activations for clustering
    train_activations = train_activations.view(train_activations.size(0), -1)

    # Apply k-means clustering to the activations
    kmeans = KMeans(n_clusters=num_clusters).fit(train_activations)

    return kmeans, train_activations
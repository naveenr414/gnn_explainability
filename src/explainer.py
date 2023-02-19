import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import numpy as np

class PrototypeExplainer:
    def __init__(self):
        self.prototypes = [] 
    
    def learn_prototypes(self,model,data):
        pass 
    
    def get_prediction(self,data):
        pass
    
class GCExplainer(PrototypeExplainer):
    def __init__(self,layer_name="conv2",num_clusters=4):
        self.layer_name = layer_name
        self.num_clusters=num_clusters
        self.initial_activations = []
        self.kmeans = None
    
    def learn_prototypes(self,model,data):
        """Learn the cluster centers for the GCExplainer module
        
        Arguments:
          model: Instance of a torch_geometric model
          data: Graph data
          
        Returns: Nothing
        
        Side Effects: Sets prototypes to be the centers
        """
        
        hook_handle, activations = add_hook_model(model,self.layer_name)
        with torch.no_grad():
            model.eval()
            model(data.x, data.edge_index, data)
        hook_handle.remove()
        kmeans,activations_numpy = cluster_activations(activations,num_clusters = self.num_clusters)
        
        self.kmeans = kmeans
        self.initial_activations = activations_numpy
        
        centers = kmeans.cluster_centers_
        self.prototypes = centers
        
    def get_prediction(self,model,data):
        """Identify which cluster one data point (or multiple) belongs to
        
        Arguments: 
            data: Graph data which we're trying to predict
            
        Returns: Cluster Number
        """
        
        hook_handle, activations = add_hook_model(model,self.layer_name)
        with torch.no_grad():
            model.eval()
            model(data.x, data.edge_index, data)
        hook_handle.remove()
        
        activations = torch.cat(activations, dim=0)

        activations = activations.view(activations.size(0), -1)
        activations = activations.cpu().numpy()
        
        clusters = []
        for new_point in activations:
            closest_center_index = pairwise_distances_argmin(X=[new_point], Y=self.prototypes)
            clusters.append(closest_center_index)

        return np.array(clusters).flatten()
        
class ProtGNNExplainer(PrototypeExplainer):
    def __init__(self):
        pass
    
    def learn_prototypes(self,model,data):
        self.prototypes = model.prototype_vectors
        
    def get_prediction(self,model,data):
        prediction = model(data.x,data.edge_index)
        min_distances = prediction[3].detach().cpu().numpy()
        
        return np.argmin(min_distances,axis=1)


class CDMExplainer(PrototypeExplainer):
    def __init__(self, layer_key="conv3"):
        self.layer_key = layer_key

    def learn_prototypes(self, model, data):
        pass

    def get_prediction(self, model, data):
        concepts, _ = model(data.x, data.edge_index)
        #activation = torch.squeeze(activation_list[layer_key]).detach().numpy()
        return


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
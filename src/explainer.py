import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import numpy as np

import torch_explain as te
from torch_explain.logic.nn import entropy
from torch_explain.logic.metrics import test_explanation, complexity
import torch.nn.functional as F

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
        
        activations = [i for idx, i in enumerate(activations) if data.train_mask[idx]]
        
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
        
        activations = torch.cat(activations, dim=0)[data.test_mask]

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

    def learn_prototypes(self,model,data):
        """Learn the activations from a model 
        
        Arguments:
          model: Instance of a torch_geometric model
          data: Graph data
          
        Returns: Nothing
        
        Side Effects: Sets prototypes to be the activation list
        """
        
        hook_handle, activations = add_hook_model(model,self.layer_key)
        with torch.no_grad():
            model.eval()
            model(data.x, data.edge_index, data)
        hook_handle.remove()
        
        self.activation_list = [i for idx, i in enumerate(activations)]
        self.activation_list = torch.cat(self.activation_list, dim=0)
                        
    def get_prediction(self, model, data):
        concepts, _ = model(data.x, data.edge_index)
        activation = torch.squeeze(self.activation_list).detach().numpy()
        explanations = explain_classes(model, concepts, data.y, data.train_mask, data.test_mask)

        return explanations


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


def explain_classes(model, concepts, y, train_mask, test_mask, max_minterm_complexity=1000, topk_explanations=1000,
                    try_all=False):
    y = F.one_hot(y)
    y1h = F.one_hot(y[train_mask])
    y1h_test = F.one_hot(y[test_mask])

    explanations = {}

    for class_id in range(y1h.shape[1]):
        # explanation, _ = entropy.explain_class(model.lens, concepts.long().detach()[train_mask], y1h,
        #                                       concepts.long().detach()[train_mask], y1h, target_class=class_id,
        #                                       max_minterm_complexity=max_minterm_complexity,
        #                                       topk_explanations=topk_explanations, try_all=try_all)

        explanation, _ = entropy.explain_class(model.lens, concepts.long().detach(), y,
                                               train_mask, test_mask, target_class=class_id,
                                               max_minterm_complexity=max_minterm_complexity,
                                               topk_explanations=topk_explanations, try_all=try_all)

        print(explanation)
        # explanation_accuracy, _ = test_explanation(explanation, concepts[test_mask], y1h_test, target_class=class_id)
        print(len(y.shape))
        explanation_accuracy, _ = test_explanation(explanation, concepts, y, class_id, test_mask)
        explanation_complexity = complexity(explanation)

        explanations[str(class_id)] = {'explanation': explanation,
                                       'explanation_accuracy': explanation_accuracy,
                                       'explanation_complexity': explanation_complexity}

        print(
            f'Explanation class {class_id}: {explanation} - acc. = {explanation_accuracy:.4f} - compl. = {explanation_complexity:.4f}')

    return explanations
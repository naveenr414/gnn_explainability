

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

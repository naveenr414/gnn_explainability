import networkx as nx
import torch_geometric

def plot_subgraph(data,node_feature_mask,edge_mask):
    """Use networkx to visualize a subgraph based on a node feature mask and an edge mask
    
    Arguments:
        Data: PyTorch data object
        node_feature_mask: PyTorch tensor which captures which features are important
        edge_mask: PyTorch tensor (0-1) which captures which edges are important
        
    Returns: Nothing
    
    Side Effects: Plots the graph
    """
    
    all_edges = data.edge_index
    
    edge_index = all_edges[:,edge_mask>0]
    x = data.x[:,node_feature_mask>0]
    
    print(edge_index.shape,x.shape)
    
    g_data = torch_geometric.data.Data(x=x, edge_index=edge_index)
    g = torch_geometric.utils.to_networkx(g_data, to_undirected=True)
    
    nx.draw(g)

    
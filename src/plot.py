import networkx as nx
import torch_geometric
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

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

def plot_kmeans_clusters(kmeans, data):
    """Function that plots kmeans through the PCA dimensionality reduction algorithm
    
    Arguments:
        kmeans: A variable from the Sklearn kmeans
        data: Numpy array or equivalent
    
    Returns:
        Nothing
        
    Side Effects: Plots a 2D PCA plot
    """
    
    # Apply PCA to reduce data to 2 dimensions
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)

    # Get the cluster labels
    labels = kmeans.labels_

    # Define colorblind-friendly colors
    colors = sns.color_palette('colorblind', n_colors=len(np.unique(labels)))

    # Plot the data points, colored by cluster label
    for i, label in enumerate(np.unique(labels)):
        plt.scatter(data_2d[labels == label, 0], data_2d[labels == label, 1], color=colors[i], label=label)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.show()

import networkx as nx
import torch_geometric
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pickle

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

def plot_metric(folder, dataset, metric):
    """Line plots for each explainer + noise method (aggressive/conservative)

        Arguments:
            Data: PyTorch data object
            metric: string - 'completeness' or 'fidelity_plus'
            folder: string - location of text files

        Returns: Nothing

        Side Effects: Plots the graph
        """

    explain_methods = ['gcexplainer', 'protgnn', 'cdm']
    noise_methods = ['aggressive', 'conservative']
    noise_amounts = [0.1, 0.3, 0.5, 0.8]

    fig, ax = plt.subplots()
    for explain_method in explain_methods:
        for noise_method in noise_methods:
            metrics = []
            for noise_amount in noise_amounts:
                file_name = f'{explain_method}_{dataset}_{noise_method}_{noise_amount}.txt'
                file_path = os.path.join(folder, file_name)
                value = read_metric(file_path, metric)
                metrics.append(value)
            #if the metric exists in the file
            if metrics[0] is not None:
                ax.plot(noise_amounts, metrics, label=f'{explain_method} and {noise_method}')

    ax.legend()
    ax.set_xlabel('Fraction')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} For Explainers and Noise Methods in {dataset}')
    plt.savefig('test.png')
    plt.show()

def read_metric(file_path, metric):
    with open(file_path, 'r') as f:
        for line in f:
            if metric in line:
                words = line.split()
                value = words[1]
    try:
        return float(value)
    except:
        return


def plot_difference_metric(folder, dataset, metric):
    """Line plots for each explainer.
     Takes the difference of metrics for aggressive/conservative noise

        Arguments:
            Data: PyTorch data object
            metric: string - 'concepts'
            folder: string - location of text files

        Returns: Nothing

        Side Effects: Plots the graph
        """

    explain_methods = ['cdm']
    noise_methods = ['aggressive', 'conservative']
    noise_amounts = [0.1, 0.3, 0.5, 0.8]

    aggressive = []
    conservative = []

    fig, ax = plt.subplots()
    for explain_method in explain_methods:
        for noise_method in noise_methods:
            metrics = []
            for noise_amount in noise_amounts:
                file_name = f'{explain_method}_{dataset}_{noise_method}_{noise_amount}.pkl'
                file_path = os.path.join(folder, file_name)

                with open(file_path, 'rb') as f:
                    concepts = pickle.load(f)

                if noise_method == 'aggressive':
                    aggressive.append(concepts)
                elif noise_method == 'conservative':
                    conservative.append(concepts)

    # calcualte difference between conservative and
    # aggressive concepts
    avg_lst = []

    for i in range(len(aggressive)):
        difference = aggressive[i].detach().numpy() - conservative[i].detach().numpy()
        avg = np.mean(difference)
        avg_lst.append(avg)

    #Plot
    ax.plot(noise_amounts, avg_lst, label=f'{explain_method} and {noise_method}')

    ax.legend()
    ax.set_xlabel('Fraction')
    ax.set_ylabel('Avg difference in concept vectors')
    ax.set_title('Avg difference in concept vectors for CDM')
    plt.savefig('test.png')
    plt.show()

if __name__ == "__main__":
    plot_difference_metric('results/concepts', 'bashapes', 'concepts')
    #plot_difference_metric('results/prototype_probs', 'bashapes', 'prototype_probs')
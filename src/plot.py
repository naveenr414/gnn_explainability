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
    noise_amounts = [0.0, 0.1, 0.3, 0.5, 0.8]

    fig, ax = plt.subplots()
    for explain_method in explain_methods:
        if explain_method == 'gcexplainer':
            for k_means in ['fixed', 'varied']:
                for noise_method in noise_methods:
                    metrics = []
                    for noise_amount in noise_amounts:
                        file_name = f'{explain_method}_{dataset}_{noise_method}_{noise_amount}_{k_means}.txt'
                        file_path = os.path.join(folder, file_name)
                        value = read_metric(file_path, metric)
                        metrics.append(value)
                    #if the metric exists in the file
                    if metrics[0] is not None:
                        ax.plot(noise_amounts, metrics, label=f'{explain_method} and {noise_method} and {k_means}')

        else:
            for noise_method in noise_methods:
                metrics = []
                for noise_amount in noise_amounts:
                    file_name = f'{explain_method}_{dataset}_{noise_method}_{noise_amount}.txt'
                    file_path = os.path.join(folder, file_name)
                    value = read_metric(file_path, metric)
                    metrics.append(value)
                # if the metric exists in the file
                if metrics[0] is not None:
                    ax.plot(noise_amounts, metrics, label=f'{explain_method} and {noise_method}')

    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_xlabel('Fraction')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} For Explainers and Noise Methods in {dataset}')
    plt.savefig(f'plots/{dataset}_{metric}.png', bbox_inches="tight")
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

    if metric == 'concepts':
        explain_method = 'cdm'
        title = 'Avg difference in concept vectors for CDM'
        y_lab = 'Avg difference in concept vectors'

    elif metric == 'prototype_probs':
        explain_method = 'protgnn'
        title = 'Avg difference in prototype probabilities'
        y_lab = 'Avg difference in prototype probabilities'

    noise_methods = ['aggressive', 'conservative']
    noise_amounts = [0.0, 0.1, 0.3, 0.5, 0.8]

    aggressive = []
    conservative = []

    fig, ax = plt.subplots()
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
        try:
            difference = aggressive[i].detach().numpy() - conservative[i].detach().numpy()
        except: #prototype probs
            difference = aggressive[i] - conservative[i]

        avg = np.mean(difference)
        avg_lst.append(avg)

    #Plot
    ax.plot(noise_amounts, avg_lst, label=f'{explain_method} and {noise_method}')

    ax.set_xlabel('Fraction')
    ax.set_ylabel(y_lab)
    ax.set_title(title)
    plt.savefig(f'plots/{explain_method}_{dataset}.png')
    plt.show()

def plot_changing_nodes(folder, dataset):

    explain_methods = ['gcexplainer']
    noise_methods = ['aggressive', 'conservative']
    noise_amounts = [0.0, 0.1, 0.3, 0.5, 0.8]
    fig, ax = plt.subplots()

    for explain_method in explain_methods:
        for noise_method in noise_methods:
            num_changes_lst = []
            for noise_amount in noise_amounts:
                file_name = f'{explain_method}_{dataset}_{noise_method}_{noise_amount}_fixed.txt'
                file_path = os.path.join(folder, file_name)

                # Open the file for reading
                with open(file_path, "r") as f:
                    # Read the entire file into a list of strings
                    lines = f.readlines()

                # Find the index of the line that separates the two sections
                separator_index = lines.index("Modified Activations\n")

                # Extract the values from each section into separate lists
                baseline_values = [int(x) for x in lines[5:separator_index]]
                modified_values = [int(x) for x in lines[separator_index + 1:-2]]

                # Count the number of values that have changed
                num_changes = sum([1 for i in range(len(baseline_values)) if baseline_values[i] != modified_values[i]])
                num_changes_lst.append(num_changes)

            ax.plot(noise_amounts, num_changes_lst, label=f'{explain_method} and {noise_method}')

    ax.legend()
    ax.set_xlabel('Fraction')
    ax.set_ylabel('Number of Changes')
    ax.set_title('Number of changes with fixed kmeans - GCExplainer')
    plt.savefig(f'plots/{explain_method}_{dataset}_num_changes.png')
    plt.show()


if __name__ == "__main__":
    plot_metric('results', 'bashapes', 'completeness')
    plot_changing_nodes('results', 'bashapes')
    plot_difference_metric('results/concepts', 'bashapes', 'concepts')
    plot_difference_metric('results/prototype_probs', 'bashapes', 'prototype_probs')
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.nn import GCNConv, Set2Set, GNNExplainer
import torch_geometric.transforms as T
import torch
import torch.nn.functional as F
import os

import matplotlib.pyplot as plt
from dgl.data import BACommunityDataset
import networkx as nx
import torch_geometric
import dgl

from src.model import *
from src.data import *
from src.explainer import *
from src.plot import *
from src.modify import *
from src.protgnn import *
from src.metrics import *
import argparse


def evaluate_model(model, dataset, explainer_class, output_location):
    """Evaluate an explainability method by computing results after running over the test-portion of a dataset
    
    Arguments:
        model: Some PyTorch geometric model
        dataset: PyTorch Geometric dataset, such as BAShapes
        explanation_method: Some explanation method such as GCExplainer; 
            An object from the explainer class that has 
            learn_prototypes and get_explaination functions
        output_location: File to which we write our outputs
        
    Returns: Explaination results
    """
    
    explainer = explainer_class()
    explainer.learn_prototypes(model,dataset)
    
    predictions = explainer.get_prediction(model,dataset)
    
    return predictions


def evaluate_model_fixed_kmeans(model, dataset, modified_dataset, explainer_class, output_location):
    """Evaluate an explainability method by computing results after running over the test-portion of a dataset

    Arguments:
        model: Some PyTorch geometric model
        dataset: PyTorch Geometric dataset, such as BAShapes
        modified_dataset: PyTorch Geometric dataset, with noise
        explanation_method: Some explanation method such as GCExplainer;
            An object from the explainer class that has
            learn_prototypes and get_explaination functions
        output_location: File to which we write our outputs

    Returns: Explaination results
    """

    explainer = explainer_class()

    #learn kmeans clusters on clean data
    explainer.learn_prototypes(model, dataset)

    #make predictions on the noisy data with the fixed kmeans clusters
    predictions = explainer.get_prediction(model, modified_dataset)

    return predictions


if __name__ == "__main__":
    explain_methods = ['gcexplainer', 'cdm', 'protgnn']
    noise_methods = ['aggressive','targeted','conservative']
    datasets = ['bashapes','bacommunity']
    noise_amounts = [0.0, 0.1, 0.3, 0.5, 0.8]
    k_means = ['fixed', 'varied']

    parser = argparse.ArgumentParser(description='Evaluate a methodology for a particular dataset and noise method')
    parser.add_argument('--explain_method',type=str, choices=explain_methods,
                        help='Which explainability method to use; either gcexplainer, cdm, or protgnn')
    parser.add_argument('--noise_method', type=str, choices=noise_methods,
                        help='Which algorithm to use to generate noise: either aggressive, targeted, or conservative')
    parser.add_argument('--noise_amount', type=float, choices=noise_amounts,
                        help='Which fraction of nodes to generate edges between: either 0.1, 0.3, 0.5 or 0.8')
    parser.add_argument('--dataset', type=str, choices=datasets,
                        help='Which dataset we\'re testing on; either bashapes or bacommunity')
    parser.add_argument('--model_location', type=str,
                        help='Where to load the model we use for evaluation from')
    parser.add_argument('--output_location',type=str,
                        help='Which file to write the output to')
    
    args = parser.parse_args()
    
    formatted_dataset = args.dataset[:3].upper() + args.dataset[3:]
    dataset = ba_dataset = get_dataset(formatted_dataset)
    
    num_classes = len(set([int(i) for i in ba_dataset.y]))
    num_features = ba_dataset.x.shape[-1]
    dim = 20
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.explain_method == 'gcexplainer':
        model = Net(num_features=num_features, dim=dim, num_classes=num_classes).to(device)
    elif args.explain_method == 'protgnn':
        model = GCNNet_NC(num_features, num_classes, model_args)
    elif args.explain_method == 'cdm':
        model = GCN(num_features=num_features, dim=dim, num_classes=num_classes)

    model.load_state_dict(torch.load(args.model_location))
    
    # Baseline evaluation
    explainer_class_dict = {'protgnn': ProtGNNExplainer, 'cdm': CDMExplainer, 'gcexplainer': GCExplainer}
    explainer_class = explainer_class_dict[args.explain_method]
    
    baseline_activations = evaluate_model(model,dataset,explainer_class,args.output_location)

    modification_dict = {'aggressive': lambda a: aggressive_adversary(a,args.noise_amount),
                         'conservative': lambda a: conservative_adversary(a, args.dataset, args.noise_amount)}
    modified_dataset = modification_dict[args.noise_method](dataset)

    if k_means == 'fixed':
        modified_activations = evaluate_model_fixed_kmeans(model, dataset, modified_dataset, explainer_class, output_location)
    else:
        modified_activations = evaluate_model(model,modified_dataset,explainer_class,args.output_location)

    evaluation_metrics = {
        'cdm': ['completeness', 'concepts'],
        'gcexplainer': ['fidelity_plus', 'completeness'],
        'protgnn': ['fidelity_plus', 'prototype_probs']

    }
    evaluation_names = []
    evaluation_results = []

    if 'fidelity_plus' in evaluation_metrics[args.explain_method]:
        evaluation_results.append(fidelity_plus(baseline_activations, modified_activations))
    if 'completeness' in evaluation_metrics[args.explain_method]:
        evaluation_results.append(completeness(explainer_class, model, modified_dataset))
    if 'concepts' in evaluation_metrics[args.explain_method]:
        concepts(explainer_class, model, modified_dataset, args.output_location)
    if 'prototype_probs' in evaluation_metrics[args.explain_method]:
        prototype_probs(explainer_class, model, modified_dataset, args.output_location)

    w = open(f'results/{args.output_location}.txt',"w")
    w.write("Modification Function: {}\n".format(args.noise_method))
    w.write("Modification Noise Amount: {}\n".format(args.noise_amount))
    w.write("Dataset: {}\n".format(args.dataset))
    w.write("Explaination method: {}\n".format(args.explain_method))
    w.write("Baseline Activations\n")
    for datapoint in baseline_activations:
        w.write(str(datapoint))
        w.write("\n")

    w.write("Modified Activations\n")
    for datapoint in modified_activations:
        w.write(str(datapoint))
        w.write("\n")

    for metric_name, metric_value in zip(evaluation_metrics[args.explain_method], evaluation_results):
        w.write("{}: {}\n".format(metric_name,metric_value))
        
    w.close()
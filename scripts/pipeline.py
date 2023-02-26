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
    
if __name__ == "__main__":
    explain_methods = ['gcexplainer', 'cdm', 'protgnn']
    noise_methods = ['aggressive','targeted','conservative']
    datasets = ['bashapes','bacommunity']
    
    parser = argparse.ArgumentParser(description='Evaluate a methodology for a particular dataset and noise method')
    parser.add_argument('--explain_method',type=str, values=explain_methods,
                        help='Which explainability method to use; either gcexplainer, cdm, or protgnn')
    parser.add_argument('--noise_method', type=str, values=noise_methods,
                        help='Which algorithm to use to generate noise: either aggressive, targeted, or conservative')
    parser.add_argument('--dataset', type=str, values=datasets,
                        help='Which dataset we\'re testing on; either bashapes or bacommunity')
    parser.add_argument('--model_location', type=str,
                        help='Where to load the model we use for evaluation from')
    parser.add_argument('--output_location',type=str,
                        help='Which file to write the output to')
    
    args = parser.parse_args()
    
    formatted_dataset = args.dataset[:3].capitalize() + args.dataset[3:]
    dataset = ba_dataset = get_dataset(formatted_dataset)
    
    num_classes = len(set([int(i) for i in ba_dataset.y]))
    num_features = ba_dataset.x.shape[-1]
    dim = 16
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(num_features=num_features, dim=dim, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.model_location))
    
    # Baseline evaluation
    explainer_class_dict = {'protgnn': ProtGNNExplainer, 'cdm': CDMExplainer, 'gcexplainer': GCExplainer}
    explainer_class = explainer_class_dict[args.explain_method]
    
    baseline_activations = evaluate_model(model,dataset,explainer_class)
    
    modification_dict = {'aggressive': lambda a: aggressive_adversary(a,0.1)}
    modified_dataset = modification_dict[args.noise_method](dataset)
    modified_activations = evaluate_model(model,modified_dataset,explainer_class)
    
    evaluation_metrics = [fidelity_plus]
    evaluation_names = ["Fidelity"]
    evaluation_results = [func(baseline_activations,modified_activations) for func in evaluation_metrics]
    
    w = open(args.output_location,"w")
    w.write("Modification Function: {}\n".format(args.noise_method))
    w.write("Dataset: {}\n".format(args.dataset))
    w.write("Explaination method: {}\n".format(args.explanation_method))
    w.write("Baseline Activations\n")
    for datapoint in baseline_activations:
        w.write(datapoint)
        w.write("\n")

    w.write("Modified Activations\n")
    for datapoint in modified_activations:
        w.write(datapoint)
        w.write("\n")

    for metric_name, metric_value in zip(evaluation_names, evaluation_results):
        w.write("{}: {}\n".format(metric_name,metric_value))
        
    w.close()
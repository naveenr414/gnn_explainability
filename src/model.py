from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.nn import GCNConv, Set2Set, GNNExplainer
import torch_geometric.transforms as T
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
import torch_explain as te


class Net(torch.nn.Module):
    def __init__(self, num_features, dim=16, num_classes=1):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, dim)
        self.conv2 = GCNConv(dim, num_classes)

    def forward(self, x, edge_index, data=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class SimpleConv(torch.nn.Module):
    def __init__(self,num_features,dim=16,hidden_size=16):
        super(SimpleConv, self).__init__()
        self.conv1 = GCNConv(num_features, dim)
        self.conv2 = GCNConv(dim, hidden_size)

    def forward(self, x, edge_index, data=None):
        x = F.relu(self.conv1(x, edge_index))
        print(x.shape)
        x = F.relu(self.conv2(x, edge_index))
        return x

class CDM(torch.nn.Module):
    def __init__(self, encoding_model, cluster_encoding_size, dim,hidden_size):
        super(CDM, self).__init__()
        
        self.encoding_model = encoding_model

        # linear layers
        self.lens = torch.nn.Sequential(te.nn.EntropyLinear(cluster_encoding_size, dim, n_classes=hidden_size),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.Linear(dim, 1))

    def forward(self, x, edge_index):
        x = self.encoding_model(x,edge_index)
                        
        self.gnn_embedding = x
        
        x = F.softmax(x, dim=-1)
        x = torch.div(x, torch.max(x, dim=-1)[0].unsqueeze(1))
        concepts = x
        
        x = self.lens(x)
                
        return concepts, x.squeeze(-1)

def run_experiment(data, seed, path):
    config = {'seed': seed,
                       'dataset_name': DATASET_NAME,
                       'model_name': MODEL_NAME,
                       'num_classes': NUM_CLASSES,
                       'train_test_split': TRAIN_TEST_SPLIT,
                       'num_hidden_units': NUM_HIDDEN_UNITS,
                       'cluster_encoding_size': CLUSTER_ENCODING_SIZE,
                       'epochs': EPOCHS,
                       'lr': LR,
                       'num_nodes_view': NUM_NODES_VIEW,
                       'num_expansions': NUM_EXPANSIONS,
                       'layer_num': LAYER_NUM,
                       'layer_key': LAYER_KEY
                      }
    persistence_utils.persist_experiment(config, path, 'config.z')
    persistence_utils.persist_experiment(data, path,'data.z')

    # model training
    model = GCN(data["x"].shape[1], NUM_HIDDEN_UNITS, CLUSTER_ENCODING_SIZE, NUM_CLASSES)
    
    # register hooks to track activation
    model = model_utils.register_hooks(model)

    # train 
    train_acc, test_acc, train_loss, test_loss = model_utils.train(model, data, EPOCHS, LR)
    persistence_utils.persist_model(model, path, 'model.z')
        
    visualisation_utils.plot_model_accuracy(train_acc, test_acc, MODEL_NAME, path)
    visualisation_utils.plot_model_loss(train_loss, test_loss, MODEL_NAME, path)
    
    x = data["x"]
    edges = data['edges']
    edges_t = data['edge_list'].numpy()
    y = data["y"]
    train_mask = data["train_mask"]
    test_mask = data["test_mask"]
    
    # get model activations for complete dataset
    concepts, _ = model(x, edges)
    activation = torch.squeeze(model_utils.activation_list[LAYER_KEY]).detach().numpy()
    persistence_utils.persist_experiment(concepts, path, 'concepts.z')
    persistence_utils.persist_experiment(model_utils.activation_list, path, 'activation_list.z')


    # find centroids
    centroids, centroid_labels, used_centroid_labels = clustering_utils.find_centroids(activation, concepts, y)
    print(f"Number of cenroids: {len(centroids)}")
    persistence_utils.persist_experiment(centroids, path, 'centroids.z')
    persistence_utils.persist_experiment(centroid_labels, path, 'centroid_labels.z')
    persistence_utils.persist_experiment(used_centroid_labels, path, 'used_centroid_labels.z')
    
    # plot concept heatmaps
    visualisation_utils.plot_concept_heatmap(centroids, concepts, y, used_centroid_labels, MODEL_NAME, LAYER_NUM, path)
    
    # concept alignment
    homogeneity = homogeneity_score(y, used_centroid_labels)
    print(f"Concept homogeneity score: {homogeneity}")

    # clustering efficency
    completeness = completeness_score(y, used_centroid_labels)
    print(f"Concept completeness score: {completeness}")
    
    # calculate cluster sizing
    cluster_counts = visualisation_utils.print_cluster_counts(used_centroid_labels)
    concept_metrics = [('homogeneity', homogeneity), ('completeness', completeness), ('cluster_count', cluster_counts)]
    persistence_utils.persist_experiment(concept_metrics, path, 'concept_metrics.z')
    
    # generate explanations
    explanations = lens_utils.explain_classes(model, concepts, y, train_mask, test_mask, max_minterm_complexity=10, topk_explanations=3)
    persistence_utils.persist_experiment(explanations, path, 'explanations.z')

    # plot clustering
    visualisation_utils.plot_clustering(seed, activation, y, centroids, centroid_labels, used_centroid_labels, MODEL_NAME, LAYER_NUM, path)

    # plot samples
    sample_graphs, sample_feat = visualisation_utils.plot_samples(None, activation, y, LAYER_NUM, len(centroids), "Differential Clustering", "Raw", NUM_NODES_VIEW, edges_t, NUM_EXPANSIONS, path, concepts=centroids)
    persistence_utils.persist_experiment(sample_graphs, path, 'sample_graphs.z')
    persistence_utils.persist_experiment(sample_feat, path, 'sample_feat.z')
    
    # clean up
    plt.close()

    
def test_model(model, data):
    """Helper function for model training, which takes in a model and data, and evaluates it
    
    Arguments:
        model: PyTorch Module
        data: PyTorch dataloader
        
    Returns: Accuracies from evalaution
    """
    
    model.eval()
    logits, accs = model(data.x, data.edge_index, data), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

    
def train_model(epochs,model,device,data,optimizer,test_function):
    """Train a model for a set number of epochs
    
    Arguments:
         epochs: Number of epochs to train for
         model: Which model to train, object from Module Class
         device: String such as 'cuda' or 'cpu'
         data: PyTorch data loader 
         optimizer: PyTorch optimizer
         test_function: Function such as test_model
         
    Returns: Trained model
    """
    
    loss = 999.0
    train_acc = 0.0
    test_acc = 0.0

    t = trange(epochs, desc="Stats: ", position=0)

    for epoch in t:

        model.train()

        loss = 0

        data = data.to(device)
        optimizer.zero_grad()
        log_logits = model(data.x, data.edge_index, data)

        # Since the data is a single huge graph, training on the training set is done by masking the nodes that are not in the training set.
        loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # validate
        train_acc, test_acc = test_function(model, data)
        train_loss = loss

        t.set_description('[Train_loss:{:.6f} Train_acc: {:.4f}, Test_acc: {:.4f}]'.format(loss, train_acc, test_acc))
        
    return model

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

    
def test_model(model, data,protgnn=False):
    """Helper function for model training, which takes in a model and data, and evaluates it
    
    Arguments:
        model: PyTorch Module
        data: PyTorch dataloader
        
    Returns: Accuracies from evalaution
    """
    
    model.eval()
    logits, accs = model(data.x, data.edge_index, data), []

    if protgnn:
        logits = logits[0]
    
    for _, mask in data('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

    
def train_model(epochs,model,device,data,optimizer,test_function,protgnn=False):
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

        if protgnn:
            log_logits = log_logits[0]
            
        
        # Since the data is a single huge graph, training on the training set is done by masking the nodes that are not in the training set.
        loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # validate
        train_acc, test_acc = test_function(model, data,protgnn=protgnn)
        train_loss = loss

        t.set_description('[Train_loss:{:.6f} Train_acc: {:.4f}, Test_acc: {:.4f}]'.format(loss, train_acc, test_acc))
        
    return model

import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, GRUCell, CrossEntropyLoss
from torch_geometric.data import Data

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv,SAGEConv,GATv2Conv, GINConv, Linear

import torch
import numpy as np

from torch_geometric_temporal.nn.recurrent import GConvGRU,  EvolveGCNH, EvolveGCNO
import torch.nn as nn

dataset_dim = {
    'bitcoinotc': {
        'input_dim': 1,
        'hidden_dim': 100
    },
    'reddit-title': {
        'input_dim': 1,
        'hidden_dim': 152,
    },
    'email-eu': {
        'input_dim': 1,
        'hidden_dim': 128
    }
}

def load_model(model_str, dataset):
    emb = None
    if model_str == 'evolvegcn':
        model = EvolveGCN(input_dim = dataset_dim[dataset]['input_dim'], hidden_dim=dataset_dim[dataset]['hidden_dim'])
    elif model_str == 'gcrngru':
        model = GCRNGRU(input_dim = dataset_dim[dataset]['input_dim'], hidden_dim=dataset_dim[dataset]['hidden_dim'])
    elif model_str == 'roland':
        model = ROLAND(input_dim = dataset_dim[dataset]['input_dim'], num_gnn_layers=2, hidden_dim=dataset_dim[dataset]['hidden_dim'])
    return model

def save_trained_model(model, model_str, dataset_str):
    if model_str == 'roland': 
        torch.save(model.get_memory(), f'trained_models/{model_str}_{dataset_str}_mem.pt')
    torch.save(model.state_dict(), f'trained_models/{model_str}_{dataset_str}.pt')

def load_trained_model(model_str, path_param, dataset_str):
    model = load_model(model_str, dataset_str)
    model.load_state_dict(torch.load(path_param, weights_only=False))
    memory = None
    if model_str == 'roland':
        memory = torch.load(f'trained_models/{model_str}_{dataset_str}_mem.pt')
        model.set_memory(memory)
    return model

class EvolveGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, loss_fn=BCEWithLogitsLoss):
        super(EvolveGCN, self).__init__()
        self.pre = torch.nn.Linear(input_dim,hidden_dim)
        self.evolve = EvolveGCNO(hidden_dim)
        self.post = torch.nn.Linear(hidden_dim, 2)
        
        self.loss_fn = loss_fn()
        self.memory = None
        
    def reset_parameters(self):
        self.post.reset_parameters()

    def forward(self, x, edge_index, edge_label_index, *args, **kwargs):
        h = self.pre(x.float())
        h = self.evolve(h, edge_index)
        h = F.relu(h)
        
        #HADAMARD MLP
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        h_hadamard = torch.mul(h_src, h_dst) #hadamard product
        h = self.post(h_hadamard.float())
        h = torch.sum(h.clone(), dim=-1).clone()
        
        return h

    def node_embed(self, x, edge_index, node_index):
        #Method used by PGExplainer to create its input features
        h = self.pre(x.float())
        h = self.evolve(h, edge_index)
        h = F.relu(h)
        h = h[node_index]
        return h
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)

class GCRNGRU(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, loss_fn=BCEWithLogitsLoss):
        super(GCRNGRU, self).__init__()
        self.pre = torch.nn.Linear(input_dim,hidden_dim)
        self.evolve = GConvGRU(hidden_dim, hidden_dim, 2)
        self.post = torch.nn.Linear(hidden_dim, 2)
        
        self.loss_fn = loss_fn()
        
    def reset_parameters(self):
        self.post.reset_parameters()

    def forward(self, x, edge_index, edge_label_index, *args, **kwargs):
        h = self.pre(x)
        h = self.evolve(h, edge_index)
        h = F.relu(h)
        
        #HADAMARD MLP
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        h_hadamard = torch.mul(h_src, h_dst) #hadamard product
        h = self.post(h_hadamard.float())
        h = torch.sum(h.clone(), dim=-1).clone()
        
        return h

    def node_embed(self, x, edge_index, node_index):
        h = self.pre(x.float())
        h = self.evolve(h, edge_index)
        h = F.relu(h)
        h = h[node_index]
        return h
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)

class ROLAND(torch.nn.Module):
    def __init__(self, input_dim, num_gnn_layers, hidden_dim, dropout=0.0,\
                 update='gru', loss=BCEWithLogitsLoss):
        
        super(ROLAND, self).__init__()
        self.preprocess1 = Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.postprocess1 = Linear(hidden_dim, 2)
        
        #Initialize the loss function to BCEWithLogitsLoss
        self.loss_fn = loss()

        self.update = update
        self.num_gnn_layers = num_gnn_layers
        self.dropout = dropout
        
        self.updates = nn.ModuleList()
        if update=='avg':
            self.tau0 = torch.nn.Parameter(torch.Tensor([0.2]))
        else:
            for _ in range(num_gnn_layers):
                if update=='gru':
                    self.updates.append(GRUCell(hidden_dim, hidden_dim))
                elif update=='mlp':
                    self.updates.append(Linear(hidden_dim*2, hidden_dim))
        self.previous_embeddings = None
        self.device = torch.device('cuda')
        #self.device = 'cpu'
                                    
        
    def reset_loss(self,loss=BCEWithLogitsLoss):
        self.loss_fn = loss()
        
    def reset_parameters(self):
        self.preprocess1.reset_parameters()
        for i in range(self.num_gnn_layers):
            self.convs[i].reset_parameters()
            self.updates[i].reset_parameters()
        self.postprocess1.reset_parameters()

    def set_memory(self, embeddings):
        self.previous_embeddings = [embeddings[i].clone().detach() for i in range(self.num_gnn_layers)]

    def get_memory(self):
        return self.previous_embeddings

    def node_embed(self, x, edge_index, node_index):
        current_embeddings = [torch.Tensor([]) for i in range(self.num_gnn_layers)]
        
        #Preprocess node repr
        h = self.preprocess1(x)
        h = h.relu()
        
        #ROLAND forward
        for z in range(self.num_gnn_layers):
            h = self.convs[z](h, edge_index)
            h = h.relu()
            #Embedding Update after first layer
            if self.update=='gru':
                h = self.updates[z](h.float(), self.previous_embeddings[z].float())
            elif self.update=='mlp':
                hin = torch.cat((h,self.previous_embeddings[z].clone()),dim=1)
                h = self.updates[z](hin)
            else:
                h = torch.Tensor((self.tau0 * self.previous_embeddings[z].clone() + (1-self.tau0) * h.clone()).detach().cpu().numpy()).to(self.device)
            current_embeddings[z] = h.clone()

        self.set_memory(current_embeddings)
        
        return h[node_index]

    def forward(self, x, edge_index, edge_label_index=None, isnap=1, previous_embeddings=None, explain=False, *args, **kwargs):
        #You do not need all the parameters to be different to None in test phase
        #You can just use the saved previous embeddings and tau
        if previous_embeddings is not None and isnap > 0: #None if test
            if isinstance(previous_embeddings, list):
                self.previous_embeddings = [previous_embeddings[i].clone() for i in range(self.num_gnn_layers)]
            
        current_embeddings = [torch.Tensor([]) for i in range(self.num_gnn_layers)]
        
        #Preprocess node repr
        h = self.preprocess1(x)
        h = h.relu()
        
        #ROLAND forward
        for z in range(self.num_gnn_layers):
            h = self.convs[z](h, edge_index)
            h = h.relu()
            #Embedding Update after first layer
            if isnap > 0:
                if self.update=='gru':
                    if explain:
                        h = self.updates[z](h.float(), self.previous_embeddings[z].float())
                    else:
                        #h = h.to(self.device)
                        #self.previous_embeddings[z] = self.previous_embeddings[z].to(self.device)
                        h = torch.Tensor(self.updates[z](h, self.previous_embeddings[z].clone()).detach().cpu().numpy()).to(self.device)
                elif self.update=='mlp':
                    hin = torch.cat((h,self.previous_embeddings[z].clone()),dim=1)
                    if explain:
                        h = self.updates[z](hin)
                    else:
                        h = torch.Tensor(self.updates[z](hin).detach().cpu().numpy()).to(self.device)
                else:
                    h = torch.Tensor((self.tau0 * self.previous_embeddings[z].clone() + (1-self.tau0) * h.clone()).detach().cpu().numpy()).to(self.device)
            current_embeddings[z] = h.clone()

        self.set_memory(current_embeddings)
            
        #HADAMARD MLP
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        h_hadamard = torch.mul(h_src, h_dst) #hadamard product
        h = self.postprocess1(h_hadamard)
        h = torch.sum(h.clone(), dim=-1).clone()
        
        #return both 
        #i)the predictions for the current snapshot 
        #ii) the embeddings of current snapshot

        if explain:
            return h #return tuple object may lead to errors for torch_geometric.explain
                
        return h, current_embeddings
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
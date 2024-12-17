from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import average_precision_score
import torch_geometric

from dataset import load_dataset
from models import load_model, dataset_dim

import torch
import copy

import numpy as np
import random

dataset_split_point = {
    'bitcoinotc': {
        'train_split': 95,
        'val_split': 95+14,
        'test_split': 95+14+28
    },
    'reddit-title':{
        'train_split': 122,
        'val_split': 122+18,
        'test_split': 122+18+34
    },
    'email-eu':{
        'train_split': 361,
        'val_split': 361+52,
        'test_split': 361+52+104
    }
}

def train_val_test_split(dataset, dataset_str):
    train_split = dataset_split_point[dataset_str]['train_split']
    val_split = dataset_split_point[dataset_str]['val_split']
    test_split = dataset_split_point[dataset_str]['test_split']

    train_data = dataset[:train_split]
    val_data = dataset[train_split:val_split]
    test_data = dataset[val_split:test_split]

    return train_data, val_data, test_data


lrs = {
    'bitcoinotc': 0.01,
    'reddit-title': 0.085,
    'email-eu': 1e-3
}

def train_evolvegcn(model_str, dataset_str, device='cpu'):
    """
        Train EvolveGCN model in the live-update setting from the first snapshot until the last train set one.
    """
    evo = load_model(model_str, dataset_str)
    dataset = load_dataset(dataset_str)
    snapshots, _, _ = train_val_test_split(dataset, dataset_str)
    num_snap = len(snapshots) 
    
    evo_avgpr_test_singles = []
    evh_avgpr_test_singles = []
    
    evopt = torch.optim.Adam(params=evo.parameters(), lr=lrs[dataset_str])
    evo.reset_parameters()
    
    for i in range(num_snap):
        #CREATE TRAIN + VAL + TEST SET FOR THE CURRENT SNAP
        snapshot = copy.deepcopy(snapshots[i])
        if snapshot.x is None:
            snapshot.x = torch.ones(snapshot.num_nodes,1)
        num_current_edges = len(snapshot.edge_index[0])
        transform = RandomLinkSplit(num_val=0.0,num_test=0.25)
        train_data, _, val_data = transform(snapshot)
        test_data = copy.deepcopy(snapshots[i+1]) if i!=(num_snap-1) else dataset[dataset_split_point[dataset_str]['train_split']]
        future_neg_edge_index = negative_sampling(
            edge_index=test_data.edge_index, #positive edges
            num_nodes=test_data.num_nodes, # number of nodes
            num_neg_samples=test_data.edge_index.size(1)) # number of neg_sample equal to number of pos_edges
        #edge index ok, edge_label concat, edge_label_index concat
        num_pos_edge = test_data.edge_index.size(1)
        test_data.edge_label = torch.Tensor(np.array([1 for i in range(num_pos_edge)] + [0 for i in range(num_pos_edge)]))
        test_data.edge_label_index = torch.cat([test_data.edge_index, future_neg_edge_index], dim=-1)
        
        #TRAIN AND TEST THE MODELS FOR THE CURRENT SNAP
        evo, evo_avgpr_test, evopt =\
            ev_train_single_snapshot(evo, snapshot, dataset_str, train_data, val_data, test_data, evopt, device=device)
        
        #SAVE AND DISPLAY EVALUATION
        print(f'Snapshot: {i}\n')
        print(f'\tEvolveGCN AUPRC Test: {evo_avgpr_test}')
        
        evo_avgpr_test_singles.append(evo_avgpr_test)
        
    evo_avgpr_test_all = sum(evo_avgpr_test_singles)/len(evo_avgpr_test_singles)
    
    print(f'EvolveGCN AUPRC over time (training): Test: {evo_avgpr_test_all}')
    
    return evo, evo_avgpr_test_singles

def train_gcrngru(model_str, dataset_str, device='cpu'):
    """
        Train GCRNGRU model in the live-update setting from the first snapshot until the last train set one.
    """
    gcgru = load_model(model_str, dataset_str)
    dataset = load_dataset(dataset_str)
    snapshots, _, _ = train_val_test_split(dataset, dataset_str)
    num_snap = len(snapshots) 
    
    gcgru_avgpr_test_singles = []
    
    gcgruopt = torch.optim.Adam(params=gcgru.parameters(), lr=lrs[dataset_str])
    gcgru.reset_parameters()
    
    for i in range(num_snap):
        #CREATE TRAIN + VAL + TEST SET FOR THE CURRENT SNAP
        snapshot = copy.deepcopy(snapshots[i])
        if snapshot.x is None:
            snapshot.x = torch.ones(snapshot.num_nodes,1)
        num_current_edges = len(snapshot.edge_index[0])
        transform = RandomLinkSplit(num_val=0.0,num_test=0.25)
        train_data, _, val_data = transform(snapshot)
        test_data = copy.deepcopy(snapshots[i+1]) if i!=(num_snap-1) else dataset[dataset_split_point[dataset_str]['train_split']]
        future_neg_edge_index = negative_sampling(
            edge_index=test_data.edge_index, #positive edges
            num_nodes=test_data.num_nodes, # number of nodes
            num_neg_samples=test_data.edge_index.size(1)) # number of neg_sample equal to number of pos_edges
        #edge index ok, edge_label concat, edge_label_index concat
        num_pos_edge = test_data.edge_index.size(1)
        test_data.edge_label = torch.Tensor(np.array([1 for i in range(num_pos_edge)] + [0 for i in range(num_pos_edge)]))
        test_data.edge_label_index = torch.cat([test_data.edge_index, future_neg_edge_index], dim=-1)
        
        #TRAIN AND TEST THE MODELS FOR THE CURRENT SNAP
        gcgru, gcgru_avgpr_test, gcgruopt =\
            ev_train_single_snapshot(gcgru, snapshot, dataset_str, train_data, val_data, test_data, gcgruopt, device=device)
        
        #SAVE AND DISPLAY EVALUATION
        print(f'Snapshot: {i}\n')
        print(f'\tGCRN-GRU AUPRC Test: {gcgru_avgpr_test}')
        
        gcgru_avgpr_test_singles.append(gcgru_avgpr_test)
        
    gcgru_avgpr_test_all = sum(gcgru_avgpr_test_singles)/len(gcgru_avgpr_test_singles)
    
    print(f'EvolveGCN AUPRC over time (training): Test: {gcgru_avgpr_test_all}')
    
    return gcgru, gcgru_avgpr_test_singles

def train_roland(model_str, dataset_str, device='cpu'):
    """
        Train and evaluate ROLAND in the live update setting
    """

    roland = load_model(model_str, dataset_str)
    dataset = load_dataset(dataset_str)
    snapshots, _, _ = train_val_test_split(dataset, dataset_str)
    
    avgpr_test_singles = []
    
    rolopt = torch.optim.Adam(params=roland.parameters(), lr=lrs[dataset_str])
    roland.reset_parameters()

    
    num_snap = len(snapshots)
    input_channels = dataset_dim[dataset_str]['input_dim']
    num_nodes = snapshots[0].num_nodes
    hidden_dim = dataset_dim[dataset_str]['hidden_dim']
    last_embeddings = [torch.Tensor([[0 for i in range(hidden_dim)] for j in range(num_nodes)]),\
                                    torch.Tensor([[0 for i in range(hidden_dim)] for j in range(num_nodes)])]
    
    for i in range(num_snap):
        #CREATE TRAIN + VAL + TEST SET FOR THE CURRENT SNAP
        snapshot = copy.deepcopy(snapshots[i])
        if snapshot.x is None:
            snapshot.x = torch.ones(snapshot.num_nodes,1)
        num_current_edges = len(snapshot.edge_index[0])
        transform = RandomLinkSplit(num_val=0.0,num_test=0.25)
        train_data, _, val_data = transform(snapshot)
        test_data = copy.deepcopy(snapshots[i+1]) if i!=(num_snap-1) else dataset[dataset_split_point[dataset_str]['train_split']]
        future_neg_edge_index = negative_sampling(
            edge_index=test_data.edge_index, #positive edges
            num_nodes=test_data.num_nodes, # number of nodes
            num_neg_samples=test_data.edge_index.size(1)) # number of neg_sample equal to number of pos_edges
        #edge index ok, edge_label concat, edge_label_index concat
        num_pos_edge = test_data.edge_index.size(1)
        test_data.edge_label = torch.Tensor(np.array([1 for i in range(num_pos_edge)] + [0 for i in range(num_pos_edge)]))
        test_data.edge_label_index = torch.cat([test_data.edge_index, future_neg_edge_index], dim=-1)

        
        #TRAIN AND TEST THE MODEL FOR THE CURRENT SNAP
        roland, rolopt, avgpr_test, last_embeddings =\
            roland_train_single_snapshot(roland, snapshot, dataset_str, train_data, val_data, test_data, i,\
                                  last_embeddings, rolopt, device)
        
        
        #SAVE AND DISPLAY EVALUATION
        print(f'Snapshot: {i}\n\tROLAND AVGPR Test: {avgpr_test}')
        avgpr_test_singles.append(avgpr_test)
        
    avgpr_test_all = sum(avgpr_test_singles)/len(avgpr_test_singles)
    
    print(f'ROLAND AVGPR over time Test: {avgpr_test_all}')
    
    return roland, avgpr_test_singles

epochs = {
    'bitcoinotc': 500,
    'reddit-title': 500,
    'email-eu': 1000
}

def ev_train_single_snapshot(model, data, dataset_str, train_data, val_data, test_data,\
                          optimizer, device='cpu', verbose=False):
    
    avgpr_val_max = 0
    best_model = model
    train_data = train_data.to(device)
    best_epoch = -1
    
    tol = 5e-2

    num_epochs = epochs[dataset_str]

    model.to(device)
    train_data.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        ## Note
        ## 1. Zero grad the optimizer
        ## 2. Compute loss and backpropagate
        ## 3. Update the model parameters
        optimizer.zero_grad()
            
        pred = model(train_data.x, train_data.edge_index, train_data.edge_label_index)
        
        loss = model.loss(pred, train_data.edge_label.type_as(pred)) #loss to fine tune on current snapshot

        loss.backward(retain_graph=True)  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        ##########################################

        log = 'Epoch: {:03d}\n AVGPR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n MRR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n F1-Score Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n Loss: {}'
        avgpr_score_val = ev_test_live(model, train_data, val_data, data, device)
        
        if avgpr_val_max-tol <= avgpr_score_val:
            avgpr_val_max = avgpr_score_val
            best_epoch = epoch
            best_model = model
        else:
            break
        
    avgpr_score_test = ev_test_live(model, train_data, test_data, data, device)
            
    if verbose:
        print(f'Best Epoch: {best_epoch}')
    #print(f'Best Epoch: {best_epoch}')
    
    return best_model, avgpr_score_test, optimizer

def roland_train_single_snapshot(model, data, dataset_str, train_data, val_data, test_data, isnap,\
                          last_embeddings, optimizer, device='cpu', verbose=False):
    
    avgpr_val_max = 0
    best_model = model
    train_data = train_data.to(device)
    best_epoch = -1
    
    tol = 5e-2

    num_epochs = epochs[dataset_str]

    model.to(device)
    train_data.to(device)
    for tnsr in last_embeddings:
        tnsr.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        ## Note
        ## 1. Zero grad the optimizer
        ## 2. Compute loss and backpropagate
        ## 3. Update the model parameters
        optimizer.zero_grad()

        pred,\
        current_embeddings =\
            model(train_data.x, train_data.edge_index, edge_label_index = train_data.edge_label_index,\
                  isnap=isnap, previous_embeddings = last_embeddings)
        
        loss = model.loss(pred, train_data.edge_label.type_as(pred)) #loss to fine tune on current snapshot

        loss.backward(retain_graph=True)  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        ##########################################

        log = 'Epoch: {:03d}\n AVGPR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n MRR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n F1-Score Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n Loss: {}'
        avgpr_score_val  = roland_test_live(model, train_data, val_data, data, isnap, device)
        
        if avgpr_val_max-tol <= avgpr_score_val:
            avgpr_val_max = avgpr_score_val
            best_epoch = epoch
            best_current_embeddings = current_embeddings
            best_model = model
        else:
            break
        
        
    #avgpr_score_train = roland_test(model, train_data, data, isnap, device)
    avgpr_score_test = roland_test_live(model, train_data, test_data, data, isnap, device)
            
    if verbose:
        print(f'Best Epoch: {best_epoch}')
    #print(f'Best Epoch: {best_epoch}')
    
    return best_model, optimizer, avgpr_score_test, best_current_embeddings

def ev_test_live(model, previous_snap, test_data, data, device='cpu'): 
    model.eval()
    test_data = test_data.to(device)
    previous_snap = previous_snap.to(device)
    h = model(previous_snap.x, previous_snap.edge_index, test_data.edge_label_index)
    pred_cont = torch.sigmoid(h).cpu().detach().numpy()
    label = test_data.edge_label.cpu().detach().numpy()
    avgpr_score = average_precision_score(label, pred_cont)
    return avgpr_score

def roland_test_live(model, previous_snap, test_data, data, isnap, device='cpu'):
    model.eval()
    test_data = test_data.to(device)
    h, _ = model(previous_snap.x, previous_snap.edge_index, edge_label_index = test_data.edge_label_index, isnap=isnap)
    pred_cont_link = torch.sigmoid(h).cpu().detach().numpy()
    label_link = test_data.edge_label.cpu().detach().numpy()
    avgpr_score_link = average_precision_score(label_link, pred_cont_link)
    return avgpr_score_link

def ev_test_streaming(model, dataset, dataset_str, device='cpu'):
    model.eval()
    model.to(device)
    _, val_data, test_data = train_val_test_split(dataset, dataset_str)
    for i in range(len(val_data)):
        snap = copy.deepcopy(val_data[i])
        if snap.x is None:
            snap.x = torch.ones(snap.num_nodes,1)
        snap.edge_label_index = torch.tensor([[0],[0]]).int()
        snap.to(device)
        h = model(snap.x, snap.edge_index, snap.edge_label_index)

    avgpr_test = []
    for i in range(len(test_data)):
        test_snap = copy.deepcopy(test_data[i])
        if test_snap.x is None:
            test_snap.x = torch.ones(test_snap.num_nodes, 1)
        future_neg_edge_index = negative_sampling(
            edge_index=test_snap.edge_index, #positive edges
            num_nodes=test_snap.num_nodes, # number of nodes
            num_neg_samples=test_snap.edge_index.size(1)) # number of neg_sample equal to number of pos_edges
        #edge index ok, edge_label concat, edge_label_index concat
        num_pos_edge = test_snap.edge_index.size(1)
        test_snap.edge_label = torch.Tensor(np.array([1 for i in range(num_pos_edge)] + [0 for i in range(num_pos_edge)]))
        test_snap.edge_label_index = torch.cat([test_snap.edge_index, future_neg_edge_index], dim=-1)
        test_snap.to(device)
        h = model(snap.x, snap.edge_index, test_snap.edge_label_index)

        pred_cont = torch.sigmoid(h).cpu().detach().numpy()
        label = test_snap.edge_label.cpu().detach().numpy()
        avgpr_score = average_precision_score(label, pred_cont)

        snap = copy.deepcopy(test_snap)

        print(f'test snap {i} AUPRC: {avgpr_score}')
        avgpr_test.append(avgpr_score)

    print(f'AUPRC over test set: {sum(avgpr_test)/len(test_data)}')
    return avgpr_test

def roland_test_streaming(model, dataset, dataset_str, device='cpu'):
    model.eval()
    model.to(device)
    _, val_data, test_data = train_val_test_split(dataset, dataset_str)
    for i in range(len(val_data)):
        snap = copy.deepcopy(val_data[i])
        if snap.x is None:
            snap.x = torch.ones(snap.num_nodes,1)
        snap.edge_label_index = torch.tensor([[0],[0]]).int()
        snap.to(device)
        if i == 0:
            h, emb = model(snap.x, snap.edge_index, snap.edge_label_index, isnap=i)
        else:
            h, emb = model(snap.x, snap.edge_index, snap.edge_label_index, previous_embeddings=emb)

    avgpr_test = []
    for i in range(len(test_data)):
        test_snap = copy.deepcopy(test_data[i])
        if test_snap.x is None:
            test_snap.x = torch.ones(test_snap.num_nodes, 1)
        future_neg_edge_index = negative_sampling(
            edge_index=test_snap.edge_index, #positive edges
            num_nodes=test_snap.num_nodes, # number of nodes
            num_neg_samples=test_snap.edge_index.size(1)) # number of neg_sample equal to number of pos_edges
        #edge index ok, edge_label concat, edge_label_index concat
        num_pos_edge = test_snap.edge_index.size(1)
        test_snap.edge_label = torch.Tensor(np.array([1 for i in range(num_pos_edge)] + [0 for i in range(num_pos_edge)]))
        test_snap.edge_label_index = torch.cat([test_snap.edge_index, future_neg_edge_index], dim=-1)
        test_snap.to(device)
        h, emb = model(snap.x, snap.edge_index, test_snap.edge_label_index, previous_embeddings=emb)

        pred_cont = torch.sigmoid(h).cpu().detach().numpy()
        label = test_snap.edge_label.cpu().detach().numpy()
        avgpr_score = average_precision_score(label, pred_cont)

        snap = copy.deepcopy(test_snap)

        print(f'test snap {i} AUPRC: {avgpr_score}')
        avgpr_test.append(avgpr_score)

    print(f'AUPRC over test set: {sum(avgpr_test)/len(test_data)}')
    return avgpr_test
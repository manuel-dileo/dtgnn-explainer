from dataset import load_dataset
from models import load_trained_model
from train_models import train_val_test_split
import copy

from xaimodels import PGExplainer, EdgeGNNExplainer, LastSnapshotExplainer, KHopSubgraphExplainer, GNNExplainer
from torch_geometric.explain import ModelConfig, Explainer, ThresholdConfig, Explanation, ExplainerAlgorithm
from torch_geometric.explain import DummyExplainer, AttentionExplainer, GraphMaskExplainer, CaptumExplainer
from torch_geometric.explain.metric import fidelity, characterization_score
from torch_geometric.data import Data
import time
from torch_geometric.utils import negative_sampling, k_hop_subgraph, erdos_renyi_graph, to_networkx
from train_models import train_val_test_split, dataset_split_point
import captum

import torch
import numpy as np
import pandas as pd
import networkx as nx

import csv
import os

import re

xai_algorithm = {
    'gnnexplainer': GNNExplainer(epochs=200),
    'gnnexplainer-edge': EdgeGNNExplainer(epochs=200),
    'pg': PGExplainer(epochs=200),
    'dummy': DummyExplainer(),
    'ig': CaptumExplainer(attribution_method=captum.attr.IntegratedGradients),
    'sa': CaptumExplainer(attribution_method=captum.attr.Saliency),
    'last': LastSnapshotExplainer(),
    'khop': KHopSubgraphExplainer()
}    

def sample_edges_to_explain(test_snapshots, edges_per_snap=50):
    """
    Samples target events from a sequence of graph snapshots. The method will sample half existing target events (positive edges) and half non existing ones (negative edges).

    Args:
        test_snapshots (list): A list of test-set graph snapshots
        edges_per_snap (int, optional): The number of edges to sample as target events from each snapshot. Defaults to 50.

    Returns:
        list: A list of graph snapshots, each containing:
            - `edge_label`: A tensor containing labels for positive and negative edges.
            - `edge_label_index`: A tensor containing indices for positive and negative edges.
            - `edge_label_explain`: A tensor of sampled edge indices for explanation.
            - `target_votes`: A tensor of edge attributes corresponding to the sampled edges.
    """
    test_datas = []
    for i in range(len(test_snapshots)):
        test_data = copy.deepcopy(test_snapshots[i])
        if test_data.x is None:
            test_data.x = torch.ones(test_data.num_nodes, 1)
    
        future_neg_edge_index = negative_sampling(
            edge_index=test_data.edge_index, #positive edges
            num_nodes=test_data.num_nodes, # number of nodes
            num_neg_samples=test_data.edge_index.size(1)
        ) # number of neg_sample equal to number of pos_edges
    
        #edge index ok, edge_label concat, edge_label_index concat
        num_pos_edge = test_data.edge_index.size(1)
        test_data.edge_label = torch.Tensor(np.array([1 for i in range(num_pos_edge)] + [0 for i in range(num_pos_edge)]))
        test_data.edge_label_index = torch.cat([test_data.edge_index, future_neg_edge_index], dim=-1)

        # Assuming test_data.edge_label_index is the edge index matrix with shape [2, num_edges]
        num_edges = test_data.edge_label_index.size(1)  # total number of edges (columns)

        # Randomly sample 50 unique indices for columns
        sample_indices = torch.randperm(num_edges)[:edges_per_snap]

        # Sample 50 columns (edges) from edge_label_index
        sampled_edges = test_data.edge_label_index[:, sample_indices]
        test_data.edge_label_explain = sampled_edges

        #Sample target_votes NB: only real edge have vote!
        # Get the valid range (0 to len(edge_attr) - 1)
        max_index = test_data.edge_attr.size(0) - 1
        # Filter sample_indices to only keep values within the valid range
        valid_indices = sample_indices[sample_indices <= max_index]
        test_data.target_votes = test_data.edge_attr[valid_indices]
        test_datas.append(test_data)
    return test_datas

def explain_eval_fid(model_str, dataset_str, xai_str, edges_per_snap = 50, time_window_percentage=10, topk=20, device='cpu'):
    """
    Train an explainer for a certain base-model and dataset and evaluate the fidelity using the setting described in our paper.
    """
    
    data = load_dataset(dataset_str)
    model = load_trained_model(model_str, f"trained_models/{model_str.lower()}_{dataset_str.lower()}.pt",\
                               dataset_str)

    model.train()

    model.to(device)

    return_type = 'probs' if xai_str == 'ig' or xai_str == 'sa' else 'raw'

    model_config = ModelConfig(
        mode='binary_classification',
        task_level='edge',
        return_type=return_type
    )

    threshold_config = None if xai_str == 'last' else ThresholdConfig(threshold_type='topk', value=topk)
    #threshold_config = None

    #GNNExplainer(epochs=200), DummyExplainer(), GraphMaskExplainer(1, epochs=200), 
    #CaptumExplainer(attribution_method=captum.attr.IntegratedGradients)
    #CaptumExplainer(attribution_method=captum.attr.Saliency)
    
    explainer = Explainer(
        model=model,
        explanation_type='model',
        algorithm= xai_algorithm[xai_str],
        node_mask_type=None,
        edge_mask_type='object',
        model_config=model_config,
        threshold_config = threshold_config
    )

    train_data, val_data, test_data = train_val_test_split(data, dataset_str)
    test_explain = sample_edges_to_explain(test_data, edges_per_snap=edges_per_snap)
        
    window_snap = int((dataset_split_point[dataset_str]['test_split'] * time_window_percentage) / 100)

    neg_fids = []
    for i in range(len(test_explain)):
        edge_to_explain = copy.deepcopy(test_explain[i])
        #Candidate events are all the ones in the time window before the target event
        #The window size is specified using time_window_percentage parameter
        i_current = dataset_split_point[dataset_str]['val_split'] + i
        first_snap = i_current - window_snap if i_current - window_snap >= 0 else 0
        first_snap = 0 if xai_str == 'last' else first_snap
        snap_to_consider = data[first_snap:i_current]
        candidate_data = Data()
        candidate_data.edge_index = torch.Tensor([[],[]])
        candidate_data.ts = torch.Tensor([])
        if model_str == 'roland':
            #If base-model is roland we need to evolve the node embedding snapshot by snapshot
            for j in range(len(snap_to_consider)):
                snap = copy.deepcopy(snap_to_consider[j])
                edge_index = snap.edge_index
                ts = torch.Tensor([j for _ in range(len(edge_index[0]))])
                candidate_data.edge_index = torch.cat([candidate_data.edge_index, edge_index], dim=1)
                candidate_data.ts = torch.cat([candidate_data.ts, ts])
                if snap.x is None:
                    snap.x = torch.ones(snap.num_nodes,1).float()
                snap.to(device)
                _ = model(snap.x, snap.edge_index, edge_label_index=torch.Tensor([[0],[0]]).long().to(device))
                if j == len(snap_to_consider)-1:
                    candidate_data.x = snap.x.float()
        else:
            #Otherwise we update the model parameter only through this loop, or the internalized memory for GCRNGRU
            for j in range(len(snap_to_consider)):
                snap = copy.deepcopy(snap_to_consider[j])
                edge_index = snap.edge_index
                ts = torch.Tensor([j for _ in range(len(edge_index[0]))])
                candidate_data.edge_index = torch.cat([candidate_data.edge_index, edge_index], dim=1)
                candidate_data.ts = torch.cat([candidate_data.ts, ts])
                if j == len(snap_to_consider)-1:
                    if snap.x is None:
                        candidate_data.x = torch.ones(snap.num_nodes,1).float()
                    else:
                        candidate_data.x = snap.x.float()

        edge_to_explain.to(device)
        candidate_data.to(device)
        start = time.time()
        srcs = edge_to_explain.edge_label_explain[0]
        dsts = edge_to_explain.edge_label_explain[1]
        explanations = []
        for src,dst in zip(srcs,dsts):
            target = torch.Tensor([[src],[dst]]).long()
            target.to(device)
            if model_str == 'roland':
                explanation = explainer(
                    x=candidate_data.x.float(),
                    edge_index=candidate_data.edge_index.long(),
                    edge_label_index = target,
                    isnap = 1,
                    previous_embeddings = None,
                    explain = True,
                    ts = candidate_data.ts
                )
                explanation._model_args = [e for e in explanation._model_args if e!="previous_embeddings"]
            else:
                explanation = explainer(
                    x=candidate_data.x.float(),
                    edge_index=candidate_data.edge_index.long(),
                    edge_label_index = target,
                    ts = candidate_data.ts
                )
            explanations.append(explanation)
        end = time.time()
        execution_time = end-start
        pos_fid_snap = 0
        neg_fid_snap = 0
        for explanation in explanations:
            explanation.to(device)
            _, neg_fid = fidelity(explainer, explanation)
            neg_fid_inv = 1-neg_fid #Fidelity sufficiency for tgnn papers
            neg_fid_snap += neg_fid_inv
        neg_fid_snap /= edges_per_snap
        neg_fids.append(neg_fid_snap)

        print(f'Test snap {i}\n')
        print(f'\t Fidelity: {neg_fid_snap}')

    return neg_fids, execution_time

def eval_pos_explain(model_str, dataset_str, xai_str, edges_per_snap = 50, time_window_percentage=10, topk=20, device='cpu'):
    """
    Evaluation of explanations for temporal positive edges using cohesiveness, edge recurrency, edge reciprocity, homophily
    """
    data = load_dataset(dataset_str)
    model = load_trained_model(model_str, f"trained_models/{model_str.lower()}_{dataset_str.lower()}.pt",\
                               dataset_str)

    model.train()

    model.to(device)

    return_type = 'probs' if xai_str == 'ig' or xai_str == 'sa' else 'raw'

    model_config = ModelConfig(
        mode='binary_classification',
        task_level='edge',
        return_type=return_type
    )

    threshold_config = None if xai_str == 'last' else ThresholdConfig(threshold_type='topk', value=topk)
    
    explainer = Explainer(
        model=model,
        explanation_type='model',
        algorithm= xai_algorithm[xai_str],
        node_mask_type=None,
        edge_mask_type='object',
        model_config=model_config,
        threshold_config = threshold_config
    )

    train_data, val_data, test_data = train_val_test_split(data, dataset_str)
    test_explain = sample_pos_edges_to_explain(test_data, edges_per_snap=edges_per_snap) #only positive edges
        

    window_snap = int((dataset_split_point[dataset_str]['test_split'] * time_window_percentage) / 100)

    cohs = []
    recs = []
    recips = []
    jaccards = []
    num_recs = 0 #We keep track of the number of recurrence and reciprocal edge in the target events.
    num_recips = 0
    for i in range(len(test_explain)):
        edge_to_explain = copy.deepcopy(test_explain[i])
        i_current = dataset_split_point[dataset_str]['val_split'] + i
        first_snap = i_current - window_snap if i_current - window_snap >= 0 else 0
        first_snap = 0 if xai_str == 'last' else first_snap
        snap_to_consider = data[first_snap:i_current]
        candidate_data = Data()
        candidate_data.edge_index = torch.Tensor([[],[]])
        candidate_data.ts = torch.Tensor([])
        if model_str == 'roland':
            for j in range(len(snap_to_consider)):
                snap = copy.deepcopy(snap_to_consider[j])
                edge_index = snap.edge_index
                ts = torch.Tensor([j for _ in range(len(edge_index[0]))])
                candidate_data.edge_index = torch.cat([candidate_data.edge_index, edge_index], dim=1)
                candidate_data.ts = torch.cat([candidate_data.ts, ts])
                if snap.x is None:
                    snap.x = torch.ones(snap.num_nodes,1).float()
                snap.to(device)
                _ = model(snap.x, snap.edge_index, edge_label_index=torch.Tensor([[0],[0]]).long().to(device))
                if j == len(snap_to_consider)-1:
                    candidate_data.x = snap.x.float()
        else:
            for j in range(len(snap_to_consider)):
                snap = copy.deepcopy(snap_to_consider[j])
                edge_index = snap.edge_index
                ts = torch.Tensor([j for _ in range(len(edge_index[0]))])
                candidate_data.edge_index = torch.cat([candidate_data.edge_index, edge_index], dim=1)
                candidate_data.ts = torch.cat([candidate_data.ts, ts])
                if j == len(snap_to_consider)-1:
                    if snap.x is None:
                        candidate_data.x = torch.ones(snap.num_nodes,1).float()
                    else:
                        candidate_data.x = snap.x.float()

        edge_to_explain.to(device)
        candidate_data.to(device)
        srcs = edge_to_explain.edge_label_explain[0]
        dsts = edge_to_explain.edge_label_explain[1]
        explanations = []
        rec_snap = []
        recip_snap = []
        jaccard_snap = []
        count = 0
        for src,dst in zip(srcs,dsts):
            target = torch.Tensor([[src],[dst]]).long()
            target.to(device)
            if model_str == 'roland':
                explanation = explainer(
                    x=candidate_data.x.float(),
                    edge_index=candidate_data.edge_index.long(),
                    edge_label_index = target,
                    isnap = 1,
                    previous_embeddings = None,
                    explain = True,
                    ts = candidate_data.ts
                )
                explanation._model_args = [e for e in explanation._model_args if e!="previous_embeddings"]
            else:
                explanation = explainer(
                    x=candidate_data.x.float(),
                    edge_index=candidate_data.edge_index.long(),
                    edge_label_index = target,
                    ts = candidate_data.ts
                )
            comp_tgraph = candidate_data.edge_index.long()
            rec, check = eval_recurrence(explanation, target, comp_tgraph, device=device)
            if check:
                num_recs+=1
            recip, check = eval_reciprocity(explanation, target, comp_tgraph, device=device)
            if check:
                num_recips+=1
            jac = eval_homophily(explanation, target, comp_tgraph, device=device)
            jaccard_snap.append(jac)
            recip_snap.append(recip)
            rec_snap.append(rec)
            explanations.append(explanation)
            #print(count)
            #count+=1
        coh = eval_cohesive_explanations(explanations, device=device)
        coh_avg = np.mean(coh)
        rec_avg = np.mean(rec_snap)
        recip_avg = np.mean(recip_snap)
        jac_avg = np.mean(jaccard_snap)
        recs.append(rec_avg)        
        cohs.append(coh_avg)
        recips.append(recip_avg)
        jaccards.append(jac_avg)

        print(f'Test snap {i}\n')
        print(f'\t Cohesiveness: {coh_avg}')
        print(f'\t Edge recurrency: {rec_avg}')
        print(f'\t Edge reciprocity: {recip_avg}')
        print(f'\t Structural homophily: {jac_avg}')

    return cohs, recs, recips, jaccards, num_recs, num_recips

def eval_homophily(explanation, target, comp_tgraph, device='cpu'):
    """
    Compute the temporal common neighbors as defined in the paper for the nodes involved in a certain target event, on its explanatory subgraph.
    """
    # Create adjacency list from edge_index
    expl_edge_index = explanation.edge_index[:, (explanation.edge_mask > 0)].long()
    if expl_edge_index.numel() == 0:
        return 0
    num_nodes = max(torch.max(expl_edge_index[0]).item(), torch.max(expl_edge_index[1]).item()) + 1
    neighbors = {i: set() for i in range(num_nodes)}
    for u, v in expl_edge_index.t():
        neighbors[u.item()].add(v.item())
        neighbors[v.item()].add(u.item())
    
    # Extract the target edge
    u, v = target[:, 0].tolist()
    
    # Handle the case where nodes in the target are not in the edge_index
    if u not in neighbors or v not in neighbors:
        return 0 # Jaccard similarity is zero if nodes are not in edge_index
    
    # Compute Jaccard similarity
    intersection = neighbors[u].intersection(neighbors[v])
    #union = neighbors[u].union(neighbors[v])
    #jaccard_sim = len(intersection) / len(union) if len(union) > 0 else 0.0
    denom = min(len(neighbors[u]), len(neighbors[v])) - 1
    jaccard_sim = len(intersection) / denom if denom > 0 else 0.0
    
    return jaccard_sim

def eval_recurrence(explanation, target, candidate_data, device='cpu'):
    def is_edge_recurrent(target, candidates, device='cpu'):
        """
        Check if any edge in `target` is present in `candidates`.
        
        Parameters:
        - target: Tensor of shape [2, num_edges], representing edges as pairs of nodes.
        - candidates: Tensor of shape [2, num_edges], representing edges as pairs of nodes.
        - device: Device to perform computation on.
        
        Returns:
        - True if any edge in `target` is present in `candidates`; False otherwise.
        """
        # Move tensors to the specified device
        target = target.to(device)
        candidates = candidates.to(device)
    
        # Convert edges to sets of tuples for comparison
        target_edges = set(map(tuple, target.T.tolist()))  # Convert target edges to a set
        candidate_edges = set(map(tuple, candidates.T.tolist()))  # Convert candidates edges to a set
    
        # Check if there's any overlap
        return len(target_edges & candidate_edges) > 0
        
    if is_edge_recurrent(target, candidate_data, device=device):
        expl_edge_index = explanation.edge_index[:, (explanation.edge_mask > 0)].long()
        if expl_edge_index.numel() == 0:
            return 0, True
        aux = 1 if is_edge_recurrent(target, expl_edge_index, device=device) else 0
        return aux, True
    else:
        return 1, False


def eval_reciprocity(explanation, target, candidate_data, device='cpu'):
    def has_reciprocal(target, candidates, device='cpu'):
        """
        Check if any edge in `target` is present in `candidates`.
        
        Parameters:
        - target: Tensor of shape [2, num_edges], representing edges as pairs of nodes.
        - candidates: Tensor of shape [2, num_edges], representing edges as pairs of nodes.
        - device: Device to perform computation on.
        
        Returns:
        - True if any edge in `target` is present in `candidates`; False otherwise.
        """
        # Move tensors to the specified device
        target_rec = target.to(device)
        candidates = candidates.to(device)

        target_rec = target_rec.flip(0)
    
        # Convert edges to sets of tuples for comparison
        target_edges = set(map(tuple, target_rec.T.tolist()))  # Convert target edges to a set
        candidate_edges = set(map(tuple, candidates.T.tolist()))  # Convert candidates edges to a set
    
        # Check if there's any overlap
        return len(target_edges & candidate_edges) > 0
        
    if has_reciprocal(target, candidate_data, device=device):
        expl_edge_index = explanation.edge_index[:, (explanation.edge_mask > 0)].long()
        if expl_edge_index.numel() == 0:
            return 0, True
        aux = 1 if has_reciprocal(target, expl_edge_index, device=device) else 0
        return aux, True
    else:
        return 1, False

def explain_case_study(model_str, dataset_str, xai_str, time_window_percentage=10, topk=20, device='cpu'):
    """
    Methods to obtain explanatory subgraphs and evaluate consensus and authority on the case-study presented in the paper: explainin the decisions behind total distrust in BitcoinOTC.
    """
    
    data = load_dataset(dataset_str)
    model = load_trained_model(model_str, f"trained_models/{model_str.lower()}_{dataset_str.lower()}.pt",\
                               dataset_str)

    model.train()

    model.to(device)

    return_type = 'probs' if xai_str == 'ig' or xai_str == 'sa' else 'raw'

    model_config = ModelConfig(
        mode='binary_classification',
        task_level='edge',
        return_type=return_type
    )

    threshold_config = None if xai_str == 'last' else ThresholdConfig(threshold_type='topk', value=topk)
    
    explainer = Explainer(
        model=model,
        explanation_type='model',
        algorithm= xai_algorithm[xai_str],
        node_mask_type=None,
        edge_mask_type='object',
        model_config=model_config,
        threshold_config = threshold_config
    )

    train_data, val_data, test_data = train_val_test_split(data, dataset_str)
    edge_to_explain, i_current = load_case_study_edges(dataset_str)

    window_snap = int((dataset_split_point[dataset_str]['test_split'] * time_window_percentage) / 100)

    pos_fids = []
    neg_fids = []
    char_scores = []
    first_snap = i_current - window_snap if i_current - window_snap >= 0 else 0
    first_snap = 0 if xai_str == 'last' else first_snap
    snap_to_consider = data[first_snap:i_current]
    candidate_data = Data()
    candidate_data.edge_index = torch.Tensor([[],[]])
    candidate_data.ts = torch.Tensor([])
    candidate_data.edge_attr = torch.Tensor([])
    if model_str == 'roland':
        for j in range(len(snap_to_consider)):
            snap = copy.deepcopy(snap_to_consider[j])
            edge_index = snap.edge_index
            ts = torch.Tensor([j for _ in range(len(edge_index[0]))])
            candidate_data.edge_index = torch.cat([candidate_data.edge_index, edge_index], dim=1)
            candidate_data.ts = torch.cat([candidate_data.ts, ts])
            edge_attr = snap.edge_attr
            candidate_data.edge_attr = torch.cat([candidate_data.edge_attr, edge_attr])
            if snap.x is None:
                snap.x = torch.ones(snap.num_nodes,1).float()
            snap.to(device)
            _ = model(snap.x, snap.edge_index, edge_label_index=torch.Tensor([[0],[0]]).long().to(device))
            if j == len(snap_to_consider)-1:
                candidate_data.x = snap.x.float()
    else:
        for j in range(len(snap_to_consider)):
            snap = copy.deepcopy(snap_to_consider[j])
            edge_index = snap.edge_index
            ts = torch.Tensor([j for _ in range(len(edge_index[0]))])
            candidate_data.edge_index = torch.cat([candidate_data.edge_index, edge_index], dim=1)
            candidate_data.ts = torch.cat([candidate_data.ts, ts])
            edge_attr = snap.edge_attr
            candidate_data.edge_attr = torch.cat([candidate_data.edge_attr, edge_attr])
            if j == len(snap_to_consider)-1:
                if snap.x is None:
                    candidate_data.x = torch.ones(snap.num_nodes,1).float()
                else:
                    candidate_data.x = snap.x.float()

    edge_to_explain.to(device)
    candidate_data.to(device)
    srcs = edge_to_explain[0]
    dsts = edge_to_explain[1]
    subgraphs = []
    cons = []
    auths = []
    cons_cand = []
    cons_rand = []
    auths_cand = []
    auths_rand = []
    for src,dst in zip(srcs,dsts):
        target = torch.Tensor([[src],[dst]]).long()
        target.to(device)
        if model_str == 'roland':
            explanation = explainer(
                x=candidate_data.x.float(),
                edge_index=candidate_data.edge_index.long(),
                edge_label_index = target,
                isnap = 1,
                previous_embeddings = None,
                explain = True,
                ts = candidate_data.ts
            )
            explanation._model_args = [e for e in explanation._model_args if e!="previous_embeddings"]
        else:
            explanation = explainer(
                x=candidate_data.x.float(),
                edge_index=candidate_data.edge_index.long(),
                edge_label_index = target,
                ts = candidate_data.ts
            )
        explanation.votes = candidate_data.edge_attr
        subgraph = Data(edge_index = explanation.edge_index[:, (explanation.edge_mask > 0)], target = target, ts = candidate_data.ts[(explanation.edge_mask > 0)], votes = explanation.votes[(explanation.edge_mask > 0)])
        subgraphs.append(subgraph)
        consensus, authority, consensus_cand, authority_cand, consensus_rand, authority_rand = case_study_metrics(explanation, device=device)
        cons.append(consensus)
        auths.append(authority)
        cons_cand.append(consensus_cand)
        cons_rand.append(consensus_rand)
        auths_cand.append(authority_cand)
        auths_rand.append(authority_rand)
    return subgraphs, cons, auths, cons_cand, auths_cand, cons_rand, auths_rand

def case_study_metrics(explanation, device='cpu'):

    def compute_consensus(destination_node, edge_index, votes):
        # CONSENSUS: Average vote received by the destination node
        # Get all edges pointing to the destination node
        destination_edges_mask = (edge_index[1] == destination_node)
        destination_votes = votes[destination_edges_mask]
        
        # Compute the average vote for the destination node
        consensus = destination_votes.mean().item() if len(destination_votes) > 0 else 0.0

        return consensus

    def compute_authority(source_node, edge_index, votes):
        # AUTHORITY: In-degree centrality of the source node on vote positive edges
        total_edges = edge_index.size(1)
        if total_edges == 0:
            authority = 0
        edge_index_pos = edge_index[:, (votes > 0)]
        source_edges_mask = (edge_index[1] == source_node)
        authority = source_edges_mask.sum().item() / total_edges
        
        return authority
    
    explanation = explanation.to(device)
    expl_edge_index = explanation.edge_index[:, (explanation.edge_mask > 0)].long()
    expl_votes = explanation.votes[(explanation.edge_mask > 0)]
    target = explanation.edge_label_index

    # Extract the target source and destination from the target tensor
    source_node = target[0].item()
    destination_node = target[1].item()

    #Metrics for explanation
    consensus = compute_consensus(destination_node, expl_edge_index, expl_votes)
    authority = compute_authority(source_node, expl_edge_index, expl_votes)

    #Metrics for candidate graph (computational graph)
    consensus_candidate = compute_consensus(destination_node, explanation.edge_index, explanation.votes)
    authority_candidate = compute_authority(source_node, explanation.edge_index, explanation.votes)
    
    #Random network generation and its metrics
    G = to_networkx(Data(edge_index=expl_edge_index))
    G.remove_nodes_from(nx.isolates(G.copy()))
    num_nodes = G.order()
    p = nx.density(G)
    index_random = erdos_renyi_graph(num_nodes, p, directed=True)
    # Generate random votes between -10 and 10 (inclusive)
    num_edges = index_random.size(1)
    votes_random = torch.randint(-10, 11, (num_edges,)).float()  # Random integers between -10 and 10
    # Randomly select an index of the edge
    random_edge_index = torch.randint(0, index_random.size(1), (1,)).item()
    # Retrieve source and destination for the selected edge
    source = index_random[0, random_edge_index].item()
    destination = index_random[1, random_edge_index].item()
    consensus_random = compute_consensus(destination, index_random, votes_random)
    authority_random = compute_authority(source, index_random, votes_random)

    return consensus, authority, consensus_candidate, authority_candidate, consensus_random, authority_random

def eval_cohesive_explanations(explanations, device='cpu'):
    cohs = []
    for explanation in explanations:
        explanation.to(device)
        coh = cohesiveness(explanation)
        cohs.append(coh)
    return cohs

def save_case_study_edges(dataset_str, dataset, snap):
    #TODO: parametrize saving options
    case_study_edges = dataset[snap].edge_index[:,(dataset[snap].edge_attr == -10)]
    torch.save(case_study_edges, f'dataset/{dataset_str}/case_study_edges.pt')
    
    # Transpose to get edges as rows (source, target)
    edge_list = case_study_edges.t()
    edge_list_lst = [(edge[0].item(), edge[1].item()) for edge in edge_list]
    ts = snap
    df = pd.read_csv(f"dataset/{dataset_str}/ml_{dataset_str}.csv")
    filtered_df = df[df['ts']==snap]
    event_idx = filtered_df[filtered_df.apply(lambda row: (row['u'], row['i']) in edge_list_lst, axis=1)]['idx'].tolist()
    with open(f"dataset/{dataset_str}/ml_{dataset_str}_casestudy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "ts", "event_idx"])
        for idx,edge in zip(event_idx, edge_list):
            writer.writerow([edge[0].item(), edge[1].item(), ts, idx])

    return case_study_edges

import math
from itertools import combinations

def cohesiveness(explanation):
    """
    Computes the cohesiveness of a set of events based on the paper TempME (Chen et al. 2023)
    """

    def enumerate_ordered_pairs(edges,ts):
        """
        Enumerates all ordered pairs of edges excluding self-pairs.
    
        Parameters:
        - edges: A list of edges.
        - ts: A list of timestamps associated to edges.
    
        Returns:
        - A list of ordered pairs of edges (tuples).
        """
        ordered_pairs = []
        
        for e_i, t_i in zip(edges,ts):
            for e_j, t_j in zip(edges,ts):
                if torch.equal(e_i, e_j) and torch.equal(t_i, t_j): continue  # Exclude self-pairs
                ordered_pairs.append(((e_i, e_j),(t_i,t_j)))
        return ordered_pairs

    
    def are_edges_adjacent(e_i, e_j):
        """
        Checks if two edges are spatially adjacent.
    
        Parameters:
        - e_i: Tuple representing the first edge (src_i, dst_i).
        - e_j: Tuple representing the second edge (src_j, dst_j).
    
        Returns:
        - True if the edges are adjacent, False otherwise.
        """
        src_i, dst_i = e_i[0].item(), e_i[1].item()
        src_j, dst_j = e_j[0].item(), e_j[1].item()
    
        # Check for shared nodes between the two edges
        return (src_i == src_j or src_i == dst_j or
                dst_i == src_j or dst_i == dst_j)

    def adjacent_edges(edge_index):
        """
        Calculates the number of adjacent edges using node degrees.
        
        Parameters:
            edge_index (torch.Tensor): A [2, num_edges] tensor representing the edges of the graph.
            
        Returns:
            int: The total number of adjacent edge pairs in the graph.
        """
        # Compute node degrees
        num_nodes = max(torch.max(edge_index[0]).item(), torch.max(edge_index[1]).item()) + 1
        degrees = torch.zeros(num_nodes, dtype=torch.int32).cuda()
        #degrees = torch.zeros(num_nodes, dtype=torch.int32)
        degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.int32).cuda())
        degrees.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), dtype=torch.int32).cuda())
        #degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.int32))
        #degrees.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), dtype=torch.int32))
        # Calculate adjacent edge pairs for each node
        adjacent_edges = torch.sum((degrees * (degrees - 1)) // 2).item()
    
        return adjacent_edges
    
    binary_mask = (explanation.edge_mask > 0).float()  # Example threshold
    n = binary_mask.sum().item()
    if n < 2:
        return 1  # Cohesiveness is not defined for fewer than 2 events

    # Compute the denominator
    denominator = (n ** 2) - n  # |G_exp_e|^2 - |G_exp_e|

    #Compute delta_T
    delta_T = (torch.max(explanation.ts[(explanation.edge_mask > 0)]) - torch.min(explanation.ts[(explanation.edge_mask > 0)])).item()
    
    # Initialize the numerator
    numerator = 0

    # Iterate over all pairs of events
    candidate_edges = explanation.edge_index[:, (explanation.edge_mask > 0)].t().long()
    
    if delta_T == 0: #all edges share the same timestamp (snapshots), count just adjacency
        numerator = adjacent_edges(candidate_edges)
        return numerator/denominator
        
    candidate_ts = explanation.ts[explanation.edge_mask.bool()]
    pairs = enumerate_ordered_pairs(candidate_edges, candidate_ts)
    #assert len(pairs) == denominator
    for (e_i, e_j), (t_i, t_j) in pairs:
        if are_edges_adjacent(e_i, e_j):  # Check if the events are related
            time_difference = abs(t_i - t_j)
            cosine_term = math.cos(time_difference / delta_T)
            numerator += cosine_term

    # Compute cohesiveness
    cohesiveness_value = numerator / denominator
    return cohesiveness_value


def load_case_study_edges(dataset_str):
    edge_index = torch.load(f'dataset/{dataset_str}/case_study_edges.pt')
    ts = int(pd.read_csv(f'dataset/{dataset_str}/case_study_edges.csv')['ts'][0])
    return edge_index, ts


def sample_pos_edges_to_explain(test_snapshots, edges_per_snap=50):
    """
    Methods equal to sample_edges_to_explain but it samples only positive edges (existing future events). 
    Useful for human readability evaluation.
    """
    test_datas = []
    for i in range(len(test_snapshots)):
        test_data = copy.deepcopy(test_snapshots[i])
        if test_data.x is None:
            test_data.x = torch.ones(test_data.num_nodes, 1)
    
        num_pos_edge = test_data.edge_index.size(1)
        test_data.edge_label = torch.Tensor(np.array([1 for i in range(num_pos_edge)]))
        test_data.edge_label_index = test_data.edge_index

        # Assuming test_data.edge_label_index is the edge index matrix with shape [2, num_edges]
        num_edges = test_data.edge_label_index.size(1)  # total number of edges (columns)

        # Randomly sample 50 unique indices for columns
        sample_indices = torch.randperm(num_edges)[:edges_per_snap]

        # Sample 50 columns (edges) from edge_label_index
        sampled_edges = test_data.edge_label_index[:, sample_indices]
        test_data.edge_label_explain = sampled_edges
        test_datas.append(test_data)
    return test_datas
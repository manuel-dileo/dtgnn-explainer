from math import sqrt
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter


from torch_geometric.explain import ExplainerConfig, Explanation, ModelConfig
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel

import torch.nn as nn



import logging
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import ReLU, Sequential

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import (
    ExplanationType,
    ModelMode,
    ModelTaskLevel,
)
from torch_geometric.nn import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.utils import get_embeddings

from torch_geometric.utils import k_hop_subgraph

class LastSnapshotExplainer(ExplainerAlgorithm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def supports(self):
        return True

    def forward(self, model, x, edge_index, *, target, index = None, **kwargs):
        max_ts = torch.max(kwargs.get('ts'))
        edge_mask = (kwargs.get('ts') == max_ts).long()
        explanation = Explanation(x=x, edge_index=edge_index, edge_mask = edge_mask, node_mask = torch.ones(x.size(0), 1))
        return explanation

class KHopSubgraphExplainer(ExplainerAlgorithm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def supports(self):
        return True

    def forward(self, model, x, edge_index, *, target, index = None, **kwargs):
        target = kwargs.get('edge_label_index')
        node_idx = [int(target[0][0]),int(target[1][0])]
        try:
            _, _, _, edge_mask = k_hop_subgraph(node_idx = node_idx, num_hops=2, edge_index=edge_index)
        except IndexError:
            edge_mask = torch.zeros(len(edge_index[0]))
        explanation = Explanation(x=x, edge_index=edge_index, edge_mask = edge_mask.int(), node_mask = torch.ones(x.size(0), 1))
        return explanation

class PGExplainer(ExplainerAlgorithm):
    r"""The PGExplainer model from the `"Parameterized Explainer for Graph
    Neural Network" <https://arxiv.org/abs/2011.04573>`_ paper.

    Internally, it utilizes a neural network to identify subgraph structures
    that play a crucial role in the predictions made by a GNN.
    Importantly, the :class:`PGExplainer` needs to be trained via
    :meth:`~PGExplainer.train` before being able to generate explanations:

    .. code-block:: python

        explainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=30, lr=0.003),
            explanation_type='phenomenon',
            edge_mask_type='object',
            model_config=ModelConfig(...),
        )

        # Train against a variety of node-level or graph-level predictions:
        for epoch in range(30):
            for index in [...]:  # Indices to train against.
                loss = explainer.algorithm.train(epoch, model, x, edge_index,
                                                 target=target, index=index)

        # Get the final explanations:
        explanation = explainer(x, edge_index, target=target, index=0)

    Args:
        epochs (int): The number of epochs to train.
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.003`).
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.PGExplainer.coeffs`.
    """

    coeffs = {
        'edge_size': 0.05,
        'edge_ent': 1.0,
        'temp': [5.0, 2.0],
        'bias': 0.01,
    }

    def __init__(self, epochs: int, lr: float = 0.003, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.mlp = Sequential(
            Linear(-1, 64),
            ReLU(),
            Linear(64, 1),
        )
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        self._curr_epoch = -1

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.mlp)

    def train(
        self,
        epoch: int,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        r"""Trains the underlying explainer model.
        Needs to be called before being able to make predictions.

        Args:
            epoch (int): The current epoch of the training phase.
            model (torch.nn.Module): The model to explain.
            x (torch.Tensor): The input node features of a
                homogeneous graph.
            edge_index (torch.Tensor): The input edge indices of a homogeneous
                graph.
            target (torch.Tensor): The target of the model.
            index (int or torch.Tensor, optional): The index of the model
                output to explain. Needs to be a single index.
                (default: :obj:`None`)
            **kwargs (optional): Additional keyword arguments passed to
                :obj:`model`.
        """
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        if self.model_config.task_level == ModelTaskLevel.node:
            if index is None:
                raise ValueError(f"The 'index' argument needs to be provided "
                                 f"in '{self.__class__.__name__}' for "
                                 f"node-level explanations")
            if isinstance(index, Tensor) and index.numel() > 1:
                raise ValueError(f"Only scalars are supported for the 'index' "
                                 f"argument in '{self.__class__.__name__}'")

        model.train()

        z = get_embeddings(model, x, edge_index, **kwargs)[-1]

        self.optimizer.zero_grad()
        temperature = self._get_temperature(epoch)

        inputs = self._get_inputs(z, edge_index, index)
        logits = self.mlp(inputs).view(-1)
        edge_mask = self._concrete_sample(logits, temperature)
        set_masks(model, edge_mask, edge_index, apply_sigmoid=True)

        if self.model_config.task_level == ModelTaskLevel.node:
            _, hard_edge_mask = self._get_hard_masks(model, index, edge_index,
                                                     num_nodes=x.size(0))
            edge_mask = edge_mask[hard_edge_mask]

        y_hat, y = model(x, edge_index, **kwargs), target

        if index is not None:
            y_hat, y = y_hat[index], y[index]

        loss = self._loss(y_hat, y, edge_mask)
        loss.backward()
        self.optimizer.step()

        clear_masks(model)
        self._curr_epoch = epoch

        return float(loss)

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        self.mlp.to(x.device)

        for epoch in range(self.epochs):
            self.train(epoch, model, x, edge_index, target=target, index=index, **kwargs)

        if self._curr_epoch < self.epochs - 1:  # Safety check:
            raise ValueError(f"'{self.__class__.__name__}' is not yet fully "
                             f"trained (got {self._curr_epoch + 1} epochs "
                             f"from {self.epochs} epochs). Please first train "
                             f"the underlying explainer model by running "
                             f"`explainer.algorithm.train(...)`.")

        hard_edge_mask = None
        if self.model_config.task_level == ModelTaskLevel.node:
            if index is None:
                raise ValueError(f"The 'index' argument needs to be provided "
                                 f"in '{self.__class__.__name__}' for "
                                 f"node-level explanations")
            if isinstance(index, Tensor) and index.numel() > 1:
                raise ValueError(f"Only scalars are supported for the 'index' "
                                 f"argument in '{self.__class__.__name__}'")

            # We need to compute hard masks to properly clean up edges and
            # nodes attributions not involved during message passing:
            _, hard_edge_mask = self._get_hard_masks(model, index, edge_index,
                                                     num_nodes=x.size(0))

        z = get_embeddings(model, x, edge_index, **kwargs)[-1]

        inputs = self._get_inputs(z, edge_index, index)

        #inputs = self._get_inputs(z, edge_index, index)
        logits = self.mlp(inputs).view(-1)

        edge_mask = self._post_process_mask(logits, hard_edge_mask,
                                            apply_sigmoid=True)

        return Explanation(edge_mask=edge_mask)

    def supports(self) -> bool:
        explanation_type = self.explainer_config.explanation_type
        task_level = self.model_config.task_level
        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type is not None:
            logging.error(f"'{self.__class__.__name__}' does not support "
                          f"explaining input node features "
                          f"got (`node_mask_type={node_mask_type.value}`)")
            return False

        return True

    ###########################################################################

    def _get_inputs(self, embedding: Tensor, edge_index: Tensor,
                    index: Optional[int] = None) -> Tensor:
        
        zs = [embedding[edge_index[0]], embedding[edge_index[1]]]
        if self.model_config.task_level == ModelTaskLevel.node:
            assert index is not None
            zs.append(embedding[index].view(1, -1).repeat(zs[0].size(0), 1))
        return torch.cat(zs, dim=-1)

    def _get_temperature(self, epoch: int) -> float:
        temp = self.coeffs['temp']
        return temp[0] * pow(temp[1] / temp[0], epoch / self.epochs)

    def _concrete_sample(self, logits: Tensor,
                         temperature: float = 1.0) -> Tensor:
        bias = self.coeffs['bias']
        eps = (1 - 2 * bias) * torch.rand_like(logits) + bias
        return (eps.log() - (1 - eps).log() + logits) / temperature

    def _loss(self, y_hat: Tensor, y: Tensor, edge_mask: Tensor) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)

        # Regularization loss:
        mask = edge_mask.sigmoid()
        size_loss = mask.sum() * self.coeffs['edge_size']
        mask = 0.99 * mask + 0.005
        mask_ent = -mask * mask.log() - (1 - mask) * (1 - mask).log()
        mask_ent_loss = mask_ent.mean() * self.coeffs['edge_ent']

        return loss + size_loss + mask_ent_loss

from math import sqrt
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from torch_geometric.explain import ExplainerConfig, Explanation, ModelConfig
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel


class EdgeGNNExplainer(ExplainerAlgorithm):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and node features that play a crucial role in the predictions
    made by a GNN.

    .. note::

        For an example of using :class:`GNNExplainer`, see
        `examples/explain/gnn_explainer.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer.py>`_,
        `examples/explain/gnn_explainer_ba_shapes.py <https://github.com/
        pyg-team/pytorch_geometric/blob/master/examples/
        explain/gnn_explainer_ba_shapes.py>`_, and `examples/explain/
        gnn_explainer_link_pred.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer_link_pred.py>`_.

    .. note::

        The :obj:`edge_size` coefficient is multiplied by the number of nodes
        in the explanation at every iteration, and the resulting value is added
        to the loss as a regularization term, with the goal of producing
        compact explanations.
        A higher value will push the algorithm towards explanations with less
        elements.
        Consider adjusting the :obj:`edge_size` coefficient according to the
        average node degree in the dataset, especially if this value is bigger
        than in the datasets used in the original paper.

    Args:
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.GNNExplainer.coeffs`.
    """

    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    def __init__(self, epochs: int = 100, lr: float = 0.01, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        self._train(model, x, edge_index, target=target, index=index, **kwargs)

        node_mask = self._post_process_mask(
            self.node_mask,
            self.hard_node_mask,
            apply_sigmoid=True,
        )
        edge_mask = self._post_process_mask(
            self.edge_mask,
            self.hard_edge_mask,
            apply_sigmoid=True,
        )

        edge_mask = self.event_to_edge_mask(edge_mask, edge_index)

        self._clean_model(model)

        return Explanation(node_mask=node_mask, edge_mask=edge_mask)

    def supports(self) -> bool:
        return True

    def _train(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        self._initialize_masks(x, edge_index)

        parameters = []
        if self.node_mask is not None:
            parameters.append(self.node_mask)
        if self.edge_mask is not None:
            set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            parameters.append(self.edge_mask)

        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for i in range(self.epochs):
            optimizer.zero_grad()

            h = x if self.node_mask is None else x * self.node_mask.sigmoid()
            y_hat, y = model(h, edge_index, **kwargs), target

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y)

            loss.backward()
            optimizer.step()

            # In the first iteration, we collect the nodes and edges that are
            # involved into making the prediction. These are all the nodes and
            # edges with gradient != 0 (without regularization applied).
            if i == 0 and self.node_mask is not None:
                if self.node_mask.grad is None:
                    raise ValueError("Could not compute gradients for node "
                                     "features. Please make sure that node "
                                     "features are used inside the model or "
                                     "disable it via `node_mask_type=None`.")
                self.hard_node_mask = self.node_mask.grad != 0.0
            if i == 0 and self.edge_mask is not None:
                if self.edge_mask.grad is None:
                    raise ValueError("Could not compute gradients for edges. "
                                     "Please make sure that edges are used "
                                     "via message passing inside the model or "
                                     "disable it via `edge_mask_type=None`.")
                self.hard_edge_mask = self.edge_mask.grad != 0.0

    def _initialize_masks(self, x: Tensor, edge_index: Tensor):
        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type

        device = x.device
        #device='cpu'
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        if node_mask_type is None:
            self.node_mask = None
        elif node_mask_type == MaskType.object:
            self.node_mask = Parameter(torch.randn(N, 1, device=device) * std)
        elif node_mask_type == MaskType.attributes:
            self.node_mask = Parameter(torch.randn(N, F, device=device) * std)
        elif node_mask_type == MaskType.common_attributes:
            self.node_mask = Parameter(torch.randn(1, F, device=device) * std)
        else:
            assert False

        if edge_mask_type is None:
            self.edge_mask = None
        elif edge_mask_type == MaskType.object:
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            self.edge_mask = Parameter(torch.randn(E, device=device) * std)
        else:
            assert False

    def _loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)
        else:
            assert False

        if self.hard_edge_mask is not None:
            assert self.edge_mask is not None
            m = self.edge_mask[self.hard_edge_mask].sigmoid()
            edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
            loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
            ent = -m * torch.log(m + self.coeffs['EPS']) - (
                1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['edge_ent'] * ent.mean()

        if self.hard_node_mask is not None:
            assert self.node_mask is not None
            m = self.node_mask[self.hard_node_mask].sigmoid()
            node_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
            loss = loss + self.coeffs['node_feat_size'] * node_reduce(m)
            ent = -m * torch.log(m + self.coeffs['EPS']) - (
                1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def event_to_edge_mask(self, edge_mask, edge_index):
        # Get the number of edges
        num_edges = edge_index.size(1)
        
        # Create a dictionary to store the product for each edge
        product_appearance_values = {}
        
        # Iterate through edges in order
        for i in range(num_edges):
            edge = (edge_index[0, i].item(), edge_index[1, i].item())
            # Update the product value for the edge
            if edge not in product_appearance_values:
                product_appearance_values[edge] = edge_mask[i].item()
            else:
                product_appearance_values[edge] *= edge_mask[i].item()
        
        # Create a new edge mask with replaced values
        new_edge_mask = torch.empty_like(edge_mask)
        
        # Replace each edge's value with the product value found
        for i in range(num_edges):
            edge = (edge_index[0, i].item(), edge_index[1, i].item())
            new_edge_mask[i] = product_appearance_values[edge]
        
        return new_edge_mask

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None


from math import sqrt
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from torch_geometric.explain import ExplainerConfig, Explanation, ModelConfig
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel


class GNNExplainer(ExplainerAlgorithm):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and node features that play a crucial role in the predictions
    made by a GNN.

    .. note::

        For an example of using :class:`GNNExplainer`, see
        `examples/explain/gnn_explainer.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer.py>`_,
        `examples/explain/gnn_explainer_ba_shapes.py <https://github.com/
        pyg-team/pytorch_geometric/blob/master/examples/
        explain/gnn_explainer_ba_shapes.py>`_, and `examples/explain/
        gnn_explainer_link_pred.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer_link_pred.py>`_.

    .. note::

        The :obj:`edge_size` coefficient is multiplied by the number of nodes
        in the explanation at every iteration, and the resulting value is added
        to the loss as a regularization term, with the goal of producing
        compact explanations.
        A higher value will push the algorithm towards explanations with less
        elements.
        Consider adjusting the :obj:`edge_size` coefficient according to the
        average node degree in the dataset, especially if this value is bigger
        than in the datasets used in the original paper.

    Args:
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.GNNExplainer.coeffs`.
    """

    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    def __init__(self, epochs: int = 100, lr: float = 0.01, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        self._train(model, x, edge_index, target=target, index=index, **kwargs)

        node_mask = self._post_process_mask(
            self.node_mask,
            self.hard_node_mask,
            apply_sigmoid=True,
        )
        edge_mask = self._post_process_mask(
            self.edge_mask,
            self.hard_edge_mask,
            apply_sigmoid=True,
        )

        self._clean_model(model)

        return Explanation(node_mask=node_mask, edge_mask=edge_mask)

    def supports(self) -> bool:
        return True

    def _train(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        self._initialize_masks(x, edge_index)

        parameters = []
        if self.node_mask is not None:
            parameters.append(self.node_mask)
        if self.edge_mask is not None:
            set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            parameters.append(self.edge_mask)

        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        model.train()

        for i in range(self.epochs):
            optimizer.zero_grad()

            h = x if self.node_mask is None else x * self.node_mask.sigmoid()
            y_hat, y = model(h, edge_index, **kwargs), target

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y)

            loss.backward()
            optimizer.step()

            # In the first iteration, we collect the nodes and edges that are
            # involved into making the prediction. These are all the nodes and
            # edges with gradient != 0 (without regularization applied).
            if i == 0 and self.node_mask is not None:
                if self.node_mask.grad is None:
                    raise ValueError("Could not compute gradients for node "
                                     "features. Please make sure that node "
                                     "features are used inside the model or "
                                     "disable it via `node_mask_type=None`.")
                self.hard_node_mask = self.node_mask.grad != 0.0
            if i == 0 and self.edge_mask is not None:
                if self.edge_mask.grad is None:
                    raise ValueError("Could not compute gradients for edges. "
                                     "Please make sure that edges are used "
                                     "via message passing inside the model or "
                                     "disable it via `edge_mask_type=None`.")
                self.hard_edge_mask = self.edge_mask.grad != 0.0

    def _initialize_masks(self, x: Tensor, edge_index: Tensor):
        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type

        device = x.device
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        if node_mask_type is None:
            self.node_mask = None
        elif node_mask_type == MaskType.object:
            self.node_mask = Parameter(torch.randn(N, 1, device=device) * std)
        elif node_mask_type == MaskType.attributes:
            self.node_mask = Parameter(torch.randn(N, F, device=device) * std)
        elif node_mask_type == MaskType.common_attributes:
            self.node_mask = Parameter(torch.randn(1, F, device=device) * std)
        else:
            assert False

        if edge_mask_type is None:
            self.edge_mask = None
        elif edge_mask_type == MaskType.object:
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            self.edge_mask = Parameter(torch.randn(E, device=device) * std)
        else:
            assert False

    def _loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)
        else:
            assert False

        if self.hard_edge_mask is not None:
            assert self.edge_mask is not None
            m = self.edge_mask[self.hard_edge_mask].sigmoid()
            edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
            loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
            ent = -m * torch.log(m + self.coeffs['EPS']) - (
                1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['edge_ent'] * ent.mean()

        if self.hard_node_mask is not None:
            assert self.node_mask is not None
            m = self.node_mask[self.hard_node_mask].sigmoid()
            node_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
            loss = loss + self.coeffs['node_feat_size'] * node_reduce(m)
            ent = -m * torch.log(m + self.coeffs['EPS']) - (
                1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None
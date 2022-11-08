import abc
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch_geometric.utils import to_undirected
import torch_sparse
from tqdm import tqdm

from src.attacks.base_attack import LocalAttack

_LOSS_TYPE = Callable[[Tensor, Tensor, Optional[Tensor]], Tensor]


def tanh_margin_loss(log_logits: Tensor, labels: Tensor,
                     idx_mask: Optional[Tensor] = None) -> Tensor:
    """Node-classification loss that focuses on nodes next to dec. boundary.

    Closely related to the margin in probability space. See paper for details."""
    if idx_mask is not None:
        log_logits = log_logits[idx_mask]
        labels = labels[idx_mask]

    sorted = log_logits.argsort(-1)
    best_non_target_class = sorted[sorted != labels[:, None]].reshape(
        log_logits.size(0), -1)[:, -1]
    margin = (
        log_logits[np.arange(log_logits.size(0)), labels]
        - log_logits[np.arange(log_logits.size(0)), best_non_target_class]
    )
    loss = torch.tanh(-margin).mean()
    return loss


class Attack(abc.ABC):
    """Abstract class for an adversarial attack that perturbs edges."""

    @abc.abstractmethod
    def attack(self, x: Tensor, edge_index: Tensor, labels: Tensor,
               budget: int, idx_attack: Optional[Tensor] = None,
               **kwargs) -> Tuple[Tensor, Tensor]:
        """Attack model.

        Args:
            x (Tensor): The node feature matrix. We assume that `x` is located
                on the target device.
            edge_index (LongTensor): The edge indices.
            labels (Tensor): Tensor containing the labels.
            budget (int): The number of allowed perturbations (i.e. 
                number of edges that are flipped at most).
            idx_attack (Tensor, optional): Filter for predictions/labels. 
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """
        raise NotImplementedError('Abstractmethod needs to be implemented')

    def __call__(self, *args, **kwargs):
        return self.attack(*args, **kwargs)


class RBCDAttack(Attack):
    """Abstract class for an adversarial attack covering a sparse greedy and
    projected gradient attack from the `Robustness of Graph Neural Networks 
    at Scale <https://www.cs.cit.tum.de/daml/robustness-of-gnns-at-scale/>`
    paper.
    """

    coeffs = {
        'max_final_samples': 20,
        'max_trials_sampling': 20,
        'eps': 1e-7
    }

    def __init__(self,
                 model: torch.nn.Module,
                 block_size: int = 1_000_000,
                 # Target class has lowest score
                 loss: _LOSS_TYPE = tanh_margin_loss,
                 is_undirected_graph: bool = True,
                 log: bool = True,
                 **kwargs) -> None:
        self.model = model
        self.block_size = block_size
        self.loss = loss
        self.is_undirected_graph = is_undirected_graph
        self.log = log

        self.coeffs.update(kwargs)

        # Need to contain the perturbations (shape [2, # perts.])
        self.flipped_edges = None
        # A sequence the attack is iteration over (values are passed to `step`)
        self.step_sequence = tuple()

    def attack(self, x: Tensor, edge_index: Tensor, labels: Tensor,
               budget: int, idx_attack: Optional[Tensor] = None,
               **kwargs) -> Tuple[Tensor, Tensor]:
        assert self.block_size > budget, (
            f'The search space size ({self.block_size}) must be '
            f'greater than the number of permutations ({budget})')

        self.model.eval()

        self.device = x.device
        assert kwargs.get('edge_weight', None) is None, \
            '`edge_weight` is not supported'
        edge_weight = torch.ones(edge_index.size(1), device=self.device)
        self.edge_index = edge_index.cpu()
        self.edge_weight = edge_weight.cpu()
        self.n = x.size(0)

        # For collecting attack statistics
        self.attack_statistics = defaultdict(list)

        # Prepare attack and define `self.iterable` to iterate over
        self.prepare(x, edge_index, labels, budget, idx_attack, **kwargs)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=len(self.step_sequence))
            pbar.set_description('Attack model')

        # Loop over the epochs (Algorithm 1, line 5)
        for step in self.step_sequence:
            scalars = self.step(step, x, edge_index, labels,
                                budget, idx_attack, **kwargs)
            self.append_statistics(scalars)

            if self.log:  # pragma: no cover
                pbar.update(1)

        if self.log:  # pragma: no cover
            pbar.close()

        ret = self.close(x, edge_index, labels, budget, idx_attack, **kwargs)

        assert self.flipped_edges.shape[1] <= budget, (
            f'# perturbed edges {self.flipped_edges.shape[1]} '
            f'exceeds budget {budget}')

        return ret

    @abc.abstractmethod
    def prepare(self, x: Tensor, edge_index: Tensor, labels: Tensor,
                budget: int, idx_attack: Optional[Tensor] = None, **kwargs):
        """Prepare attack."""
        pass

    @abc.abstractmethod
    def step(self, step: Any, x: Tensor, edge_index: Tensor, labels: Tensor,
             budget: int, idx_attack: Optional[Tensor] = None,
             **kwargs) -> Dict[str, Any]:
        """Step attack. Returned dict is added to statistics."""
        pass

    @abc.abstractmethod
    def close(self, x: Tensor, edge_index: Tensor, labels: Tensor,
              budget: int, idx_attack: Optional[Tensor] = None,
              **kwargs) -> Tuple[Tensor, Tensor]:
        """Clean up and prepare return argument."""
        pass

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                **kwargs) -> Tensor:
        """Forward model."""
        # return self.model(x, (edge_index, edge_weight), **kwargs)
        adj = torch_sparse.SparseTensor.from_edge_index(
            edge_index, edge_weight, (self.n, self.n), is_sorted=True)
        return self.model(x, adj, **kwargs)

    def _sample_random_block(self, budget: int = 0) -> None:
        for _ in range(self.coeffs['max_trials_sampling']):
            self.current_block = torch.randint(
                RBCDAttack.num_possible_edges(
                    self.n, self.is_undirected_graph),
                (self.block_size,), device=self.device)
            self.current_block = torch.unique(self.current_block, sorted=True)
            if self.is_undirected_graph:
                self.block_edge_index = RBCDAttack.linear_to_triu_idx(
                    self.n, self.current_block)
            else:
                self.block_edge_index = RBCDAttack.linear_to_full_idx(
                    self.n, self.current_block)
                self._filter_self_loops(with_weight=False)

            self.block_edge_weight = torch.full_like(
                self.current_block, self.coeffs['eps'], dtype=torch.float32)
            if self.current_block.size(0) >= budget:
                return
        raise RuntimeError(
            'Sampling random block was not successful. '
            'Please decrease `budget`.')

    def _resample_random_block(self, budget: int) -> None:
        # Keep at most half of the block (i.e. resample low weights)
        sorted_idx = torch.argsort(self.block_edge_weight)
        keep_above = (self.block_edge_weight <=
                      self.coeffs['eps']).sum().long()
        if keep_above < sorted_idx.size(0) // 2:
            keep_above = sorted_idx.size(0) // 2
        sorted_idx = sorted_idx[keep_above:]

        self.current_block = self.current_block[sorted_idx]
        self.block_edge_index = self.block_edge_index[:, sorted_idx]
        self.block_edge_weight = self.block_edge_weight[sorted_idx]

        # Sample until enough edges were drawn
        for _ in range(self.coeffs['max_trials_sampling']):
            n_edges_resample = self.block_size - self.current_block.size(0)
            lin_index = torch.randint(
                RBCDAttack.num_possible_edges(
                    self.n, self.is_undirected_graph),
                (n_edges_resample,), device=self.device)

            current_block = torch.cat((self.current_block, lin_index))
            self.current_block, unique_idx = torch.unique(
                current_block, sorted=True, return_inverse=True)

            if self.is_undirected_graph:
                self.block_edge_index = RBCDAttack.linear_to_triu_idx(
                    self.n, self.current_block)
            else:
                self.block_edge_index = RBCDAttack.linear_to_full_idx(
                    self.n, self.current_block)

            # Merge existing weights with new edge weights
            block_edge_weight_prev = self.block_edge_weight.clone()
            self.block_edge_weight = torch.full(
                self.current_block.shape, self.coeffs['eps'],
                device=self.device)
            self.block_edge_weight[
                unique_idx[:sorted_idx.size(0)]] = block_edge_weight_prev

            if not self.is_undirected_graph:
                self._filter_self_loops(with_weight=True)

            if self.current_block.size(0) > budget:
                return
        raise RuntimeError(
            'Sampling random block was not successful.'
            'Please decrease `budget`.')

    def _get_modified_adj(self) -> Tuple[Tensor, Tensor]:
        if self.is_undirected_graph:
            modified_edge_idx, modified_edge_weight = to_undirected(
                self.block_edge_index, self.block_edge_weight,
                self.n, reduce='mean')
        else:
            modified_edge_idx = self.block_edge_index
            modified_edge_weight = self.block_edge_weight
        edge_index = torch.cat(
            (self.edge_index.to(self.device), modified_edge_idx), dim=-1)
        edge_weight = torch.cat(
            (self.edge_weight.to(self.device), modified_edge_weight))

        edge_index, edge_weight = torch_sparse.coalesce(
            edge_index, edge_weight, m=self.n, n=self.n, op='sum')

        # Allow (soft) removal of edges
        edge_weight[edge_weight > 1] = 2 - edge_weight[edge_weight > 1]

        return edge_index, edge_weight

    def append_statistics(self, mapping: Dict[str, Any]):
        for key, value in mapping.items():
            self.attack_statistics[key].append(value)

    @staticmethod
    def num_possible_edges(n: int, is_undirected_graph: bool) -> int:
        """Determine number of possible edges for graph."""
        if is_undirected_graph:
            return n * (n - 1) // 2
        else:
            return int(n ** 2)  # We filter self-loops later

    @staticmethod
    def linear_to_triu_idx(n: int, lin_idx: Tensor) -> Tensor:
        """Convert a linear index to index of upper triangular matrix."""
        row_idx = (
            n
            - 2
            - torch.floor(torch.sqrt(-8 * lin_idx.double() +
                                     4 * n * (n - 1) - 7) / 2.0 - 0.5)
        ).long()
        col_idx = (
            lin_idx
            + row_idx
            + 1 - n * (n - 1) // 2
            + torch.div((n - row_idx) * ((n - row_idx) - 1),
                        2, rounding_mode='floor')
        )
        return torch.stack((row_idx, col_idx))

    @staticmethod
    def linear_to_full_idx(n: int, lin_idx: Tensor) -> Tensor:
        """Convert a linear index to index of matrix."""
        row_idx = lin_idx // n
        col_idx = lin_idx % n
        return torch.stack((row_idx, col_idx))


class GRBCDAttack(RBCDAttack):
    """Greedy Randomized Block Coordinate Descent (PRBCD) as appeared in
    Geisler et al., Robustness of Graph Neural Networks at Scale, NeurIPS 2021.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 block_size: int = 1_000_000,
                 # Target class has lowest score
                 loss: _LOSS_TYPE = tanh_margin_loss,
                 is_undirected_graph: bool = True,
                 epochs: int = 100,
                 log: bool = True,
                 **kwargs):
        super().__init__(
            model, block_size, loss, is_undirected_graph, log, **kwargs)
        self.epochs = epochs

    def prepare(self, x: Tensor, edge_index: Tensor, labels: Tensor,
                budget: int, idx_attack: Optional[Tensor] = None,
                **kwargs) -> None:
        """Prepare attack."""
        self.flipped_edges = torch.empty(
            (2, 0), dtype=edge_index.dtype, device=self.device)

        # Determine the number of edges to be flipped in each attach step / epoch
        step_size = budget // self.epochs
        if step_size > 0:
            steps = self.epochs * [step_size]
            for i in range(budget % self.epochs):
                steps[i] += 1
        else:
            steps = [1] * budget

        self.step_sequence = steps

    def step(self, step_size: int, x: Tensor, edge_index: Tensor,
             labels: Tensor, budget: int, idx_attack: Optional[Tensor] = None,
             **kwargs) -> Dict[str, Any]:
        """A single step of the attack."""
        # Sample initial search space (Algorithm 2, line 3-4)
        self._sample_random_block(step_size)
        self.block_edge_weight.requires_grad = True

        # Retrieve sparse perturbed adjacency matrix `A \oplus p_{t-1}`
        # (Algorithm 2, line 7)
        edge_index, edge_weight = self._get_modified_adj()

        # Get predictions (Algorithm 2, line 7)
        predictions = self.forward(x, edge_index, edge_weight, **kwargs)
        # Calculate loss combining all each node (Algorithm 2, line 8)
        loss = self.loss(predictions, labels, idx_attack)
        # Retrieve gradient towards the current block (Algorithm 2, line 8)
        gradient = torch.autograd.grad(loss, self.block_edge_weight)[0]

        # Greedy update of edges (Algorithm 2, line 8)
        self._greedy_update(step_size, gradient)

        return dict(loss=loss.item())

    def close(self, *args, **kwargs) -> Any:
        """Clean up and prepare return argument."""
        return self.edge_index, self.flipped_edges

    @torch.no_grad()
    def _greedy_update(self, step_size: int, gradient: torch.Tensor):
        _, topk_edge_index = torch.topk(gradient, step_size)

        add_edge_index = self.block_edge_index[:, topk_edge_index]
        add_edge_weight = torch.ones_like(
            add_edge_index[0], dtype=torch.float32)

        self.flipped_edges = torch.cat(
            (self.flipped_edges, add_edge_index), axis=-1)

        if self.is_undirected_graph:
            add_edge_index, add_edge_weight = to_undirected(
                add_edge_index, add_edge_weight, self.n, reduce='mean')
        edge_index = torch.cat(
            (self.edge_index.to(self.device), add_edge_index.to(self.device)),
            dim=-1)
        edge_weight = torch.cat(
            (self.edge_weight.to(self.device), add_edge_weight.to(self.device)))
        edge_index, edge_weight = torch_sparse.coalesce(
            edge_index, edge_weight, m=self.n, n=self.n, op='sum'
        )

        is_one_mask = torch.isclose(edge_weight, torch.tensor(1.))
        self.edge_index = edge_index[:, is_one_mask]
        self.edge_weight = edge_weight[is_one_mask]
        # self.edge_weight = torch.ones_like(self.edge_weight)
        assert self.edge_index.size(1) == self.edge_weight.size(0)


class PRBCDAttack(RBCDAttack):
    """Projected Randomized Block Coordinate Descent (PRBCD) as appeared in
    Geisler et al., Robustness of Graph Neural Networks at Scale, NeurIPS 2021.
    """

    coeffs = {
        'max_final_samples': 20,
        'max_trials_sampling': 20,
        'with_early_stopping': True,
        'eps': 1e-7
    }

    def __init__(self,
                 model: torch.nn.Module,
                 block_size: int = 1_000_000,
                 epochs_resampling: int = 100,
                 epochs_finetuning: int = 25,
                 # Target class has lowest score
                 loss: _LOSS_TYPE = tanh_margin_loss,
                 metric: _LOSS_TYPE = tanh_margin_loss,
                 lr: float = 100,
                 is_undirected_graph: bool = True,
                 log: bool = True,
                 **kwargs) -> None:
        super().__init__(
            model, block_size, loss, is_undirected_graph, log, **kwargs)

        self.epochs_resampling = epochs_resampling
        self.epochs = epochs_resampling + epochs_finetuning
        if metric is not None:
            self.metric = metric
        else:
            self.metric = loss
        self.lr = lr

        self.coeffs.update(kwargs)

        self.budget = 0

    def prepare(self, x: Tensor, edge_index: Tensor, labels: Tensor,
                budget: int, idx_attack: Optional[Tensor] = None, **kwargs):
        """Prepare attack."""
        self.step_sequence = list(range(self.epochs))

        # For early stopping (not explicitly covered by pseudo code)
        self.best_metric = float('-Inf')

        # Sample initial search space (Algorithm 1, line 3-4)
        self._sample_random_block(budget)

    def step(self, epoch: int, x: Tensor, edge_index: Tensor, labels: Tensor,
             budget: int, idx_attack: Optional[Tensor] = None,
             **kwargs) -> Dict[str, Any]:
        """A single step of the attack."""
        self.block_edge_weight.requires_grad = True

        # Retrieve sparse perturbed adjacency matrix `A \oplus p_{t-1}`
        # (Algorithm 1, line 6)
        edge_index, edge_weight = self._get_modified_adj()

        # Get prediction (Algorithm 1, line 6)
        prediction = self.forward(x, edge_index, edge_weight, **kwargs)
        # Calculate loss combining all each node (Algorithm 1, line 7)
        loss = self.loss(prediction, labels, idx_attack)
        # Retrieve gradient towards the current block (Algorithm 1, line 7)
        gradient = torch.autograd.grad(loss, self.block_edge_weight)[0]

        with torch.no_grad():
            # Gradient update step (Algorithm 1, line 7)
            edge_weight = self.update_edge_weights(
                budget, epoch, gradient)[1]
            # For monitoring
            pmass_update = torch.clamp(
                self.block_edge_weight, 0, 1).sum().item()
            # Projection to stay within relaxed `L_0` budget
            # (Algorithm 1, line 8)
            self.block_edge_weight = PRBCDAttack.project(
                budget, self.block_edge_weight, self.coeffs['eps'])
            # For monitoring
            pmass_projected = self.block_edge_weight.sum().item()

            # Calculate metric after the current epoch (overhead
            # for monitoring and early stopping)
            edge_index, edge_weight = self._get_modified_adj()
            prediction = self.forward(x, edge_index, edge_weight, **kwargs)
            metric = self.metric(prediction, labels, idx_attack)
            del edge_index, edge_weight, prediction

            # Save best epoch for early stopping
            # (not explicitly covered by pseudo code)
            if self.coeffs['with_early_stopping'] and self.best_metric < metric:
                self.best_metric = metric
                self.best_block = self.current_block.cpu()
                self.best_edge_index = self.block_edge_index.cpu()
                best_pert_edge_weight = self.block_edge_weight.detach()
                self.best_pert_edge_weight = best_pert_edge_weight.cpu()

            # Resampling of search space (Algorithm 1, line 9-14)
            if epoch < self.epochs_resampling - 1:
                self._resample_random_block(budget)
            elif (self.coeffs['with_early_stopping']
                    and epoch == self.epochs_resampling - 1):
                # Retrieve best epoch if early stopping is active
                # (not explicitly covered by pseudo code)
                self.current_block = self.best_block.to(self.device)
                self.block_edge_index = self.best_edge_index.to(self.device)
                self.block_edge_weight = self.best_pert_edge_weight.to(
                    self.device)

        return dict(loss=loss.item(),
                    metric=metric.item(),
                    prob_mass_after_update=pmass_update,
                    prob_mass_after_projection=pmass_projected)

    def close(self, x: Tensor, edge_index: Tensor, labels: Tensor,
              budget: int, idx_attack: Optional[Tensor] = None,
              **kwargs) -> Any:
        """Clean up and prepare return argument."""
        # Retrieve best epoch if early stopping is active
        # (not explicitly covered by pseudo code)
        if self.coeffs['with_early_stopping']:
            self.current_block = self.best_block.to(self.device)
            self.block_edge_index = self.best_edge_index.to(self.device)
            self.block_edge_weight = self.best_pert_edge_weight.to(self.device)

        # Sample final discrete graph (Algorithm 1, line 16)
        edge_index, flipped_edges = self.sample_final_edges(
            x, labels, budget, idx_attack=idx_attack, **kwargs)

        return edge_index, flipped_edges

    @torch.no_grad()
    def sample_final_edges(self, x: Tensor, labels: Tensor, budget: int,
                           idx_attack: Optional[Tensor] = None,
                           **kwargs) -> Tuple[Tensor, Tensor]:
        best_metric = float('-Inf')
        block_edge_weight = self.block_edge_weight
        block_edge_weight[block_edge_weight <= self.coeffs['eps']] = 0

        for i in range(self.coeffs['max_final_samples']):
            if i == 0:
                # In first iteration employ top k heuristic instead of sampling
                sampled_edges = torch.zeros_like(block_edge_weight)
                sampled_edges[torch.topk(
                    block_edge_weight, budget).indices] = 1
            else:
                sampled_edges = torch.bernoulli(block_edge_weight).float()

            if sampled_edges.sum() > budget:
                # Allowed budget is exceeded
                continue
            self.block_edge_weight = sampled_edges

            edge_index, edge_weight = self._get_modified_adj()
            prediction = self.forward(x, edge_index, edge_weight, **kwargs)
            metric = self.metric(prediction, labels, idx_attack)

            # Save best sample
            if best_metric < metric:
                best_metric = metric
                best_edge_weight = self.block_edge_weight.clone().cpu()

        # Recover best sample
        self.block_edge_weight = best_edge_weight.to(self.device)
        self.flipped_edges = self.block_edge_index[
            :, torch.where(best_edge_weight)[0]]

        edge_index, edge_weight = self._get_modified_adj()
        edge_mask = edge_weight == 1

        assert self.flipped_edges.shape[1] <= budget, (
            f'# perturbed edges {self.flipped_edges.shape[1]} '
            f'exceeds budget {budget}')

        self.edge_index = edge_index[:, edge_mask]
        self.edge_weight = edge_weight[edge_mask]

        return self.edge_index, self.flipped_edges

    def update_edge_weights(self, budget: int, epoch: int,
                            gradient: Tensor) -> Tuple[Tensor, Tensor]:
        """Update the edge weights and adaptively, heuristically refined the learning rate such that (1) it is
        independent of the number of perturbations (assuming an undirected adjacency matrix) and (2) to decay learning
        rate during fine-tuning (i.e. fixed search space).
        Parameters
        ----------
        budget : int
            Number of perturbations.
        epoch : int
            Number of epochs until fine tuning.
        gradient : Tensor
            The current gradient.
        Returns
        -------
        Tuple[Tensor, Tensor]
            Updated edge indices and weights.
        """

        lr = (budget / self.n * self.lr
              / np.sqrt(max(0, epoch - self.epochs_resampling) + 1))
        self.block_edge_weight.data.add_(lr * gradient)

        return self._get_modified_adj()

    def _filter_self_loops(self, with_weight: bool):
        is_not_sl = self.block_edge_index[0] != self.block_edge_index[1]
        self.current_block = self.current_block[is_not_sl]
        self.block_edge_index = self.block_edge_index[:, is_not_sl]
        if with_weight:
            self.block_edge_weight = self.block_edge_weight[is_not_sl]

    @staticmethod
    def project(budget: int, values: Tensor, eps: float = 1e-7):
        r"""Projects `values`: $budget \ge \sum \Pi_{[0, 1]}(\text{values})$."""
        if torch.clamp(values, 0, 1).sum() > budget:
            left = (values - 1).min()
            right = values.max()
            miu = PRBCDAttack.bisection(values, left, right, budget)
            values = values - miu
        return torch.clamp(values, min=eps, max=1 - eps)

    @staticmethod
    def bisection(edge_weights, a, b, n_pert, eps=1e-5, max_iter=1e3):
        """Bisection search for projection."""
        def shift(offset):
            return (torch.clamp(edge_weights - offset, 0, 1).sum() - n_pert)

        miu = a
        for i in range(int(max_iter)):
            miu = (a + b) / 2
            # Check if middle point is root
            if (shift(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (shift(miu) * shift(a) < 0):
                b = miu
            else:
                a = miu
            if ((b - a) <= eps):
                break
        return miu


class RBCDWrapper(LocalAttack):

    def __init__(self, attack: str, n_idx: int, X: np.ndarray, A: np.ndarray,
                 y: np.ndarray, model: torch.nn.Module, **kwargs):
        self.idx_attack = torch.tensor((n_idx,))
        self.X = torch.tensor(X)
        self.A = torch.tensor(A)
        self.y = torch.tensor(y)
        self.model = model

        if attack == 'PRBCD':
            self.attack = PRBCDAttack(model, log=False, **kwargs)
        else:
            self.attack = GRBCDAttack(model, epochs=1, log=False, **kwargs)

        self.edge_index = None
        self.budget = 0

    def create_adversarial_pert(self) -> Union[Tuple[int, int],
                                               Sequence[Tuple[int, int]]]:
        """Add an adversarial edge to the stored graph.

        Returns:
            Tuple[int, int]: Node-index-tuple of newly added edge or None if no
                perturbation possible anymore.
        """
        self.budget += 1

        device = next(self.model.parameters()).device
        X = self.X.to(device=device, dtype=torch.float32)

        if isinstance(self.attack, PRBCDAttack):
            A = self.A.to_sparse()
            A = A.to(device=device)
            self.edge_index, flipped_edges = self.attack(
                x=X, edge_index=A.indices(), labels=self.y.to(device),
                idx_attack=self.idx_attack, budget=self.budget)
            return [(u.item(), v.item()) for u, v in flipped_edges.T][0]
        else:
            if self.edge_index is None:
                A = self.A.to_sparse()
                A = A.to(device=device)
                self.edge_index = A.indices()

            self.edge_index, flipped_edges = self.attack(
                x=X, edge_index=self.edge_index, labels=self.y.to(device),
                idx_attack=self.idx_attack, budget=1)
            return flipped_edges[0].item(), flipped_edges[1].item()

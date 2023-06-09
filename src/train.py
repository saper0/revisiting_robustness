import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sacred.run import Run
import torch
import torch.nn as nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from src.models.lp import LP
from src.utils import accuracy


patch_typeguard()


class TrainingTracker():
    """Log statistics (losses & accuracies) of a model during training. 
    
    Class Invariant: Always holds parameters of best performing model so far.
    """
    def __init__(self, model: Optional[nn.Module], 
                 verbosity_params: Dict[str, Any], minimization = True, 
                 _run: Optional[Run]=None) -> None:
        """Initializes the TrainingTracker to track the given model.

        Args:
            model (Optional[nn.Module]): Model to track.
            verbosity_params (Dict[str, Any]): Has to include key "display_steps"
                to define at which training iteration statistics should be 
                sent to the standard output.
            minimization (bool, optional): Is the loss minimized? Defaults to True.
            _run (Optional[Run], optional): If given, also tracks training 
                statistics using sacred. Defaults to None.
        """
        self.model = model
        self.display_steps = verbosity_params["display_steps"]
        self.minimization = minimization
        self._run = _run
        # Statistics
        self.loss_trn = []
        self.loss_val = []
        self.acc_trn = []
        self.acc_val = []
        self.epoch = -1
        self.best_epoch = -1
        # Invariant
        self.best_state = self.get_current_model_state()

    def get_current_model_state(self):
        """Return a copy of the state dictionary of the current model."""
        if self.model is None:
            return None
        return {key: value.cpu() for key, value in self.model.state_dict().items()}

    def get_best_model_state(self):
        return self.best_state

    def get_best_epoch(self) -> int:
        """Return best training epoch (1-based)."""
        return self.best_epoch + 1

    def get_training_epochs(self) -> int:
        """Return 1-based training epochs."""
        return self.epoch + 1

    def get_training_loss(self) -> List[float]:
        return self.loss_trn

    def get_validation_loss(self) -> List[float]:
        return self.loss_val

    def get_training_accuracy(self) -> List[float]:
        return self.acc_trn

    def get_validation_accuracy(self) -> List[float]:
        return self.acc_val

    def is_better_loss(self, loss_val):
        if self.minimization:
            return self.loss_val[self.best_epoch] > loss_val
        else:
            return self.loss_val[self.best_epoch] < loss_val

    def log(self, epoch):
        if self._run is not None:
            self._run.log_scalar("loss_trn", self.loss_trn[epoch])
            self._run.log_scalar("loss_val", self.loss_val[epoch])
            self._run.log_scalar("acc_trn", self.acc_trn[epoch])
            self._run.log_scalar("acc_val", self.acc_val[epoch])

    def log_current_epoch(self):
        """Log most recently added statistics."""
        self.log(self.epoch)

    def log_best_epoch(self):
        self.log(self.best_epoch)

    def _add(self, loss_trn, loss_val, acc_trn, acc_val):
        self.loss_trn.append(loss_trn)
        self.loss_val.append(loss_val)
        self.acc_trn.append(acc_trn)
        self.acc_val.append(acc_val)
        self.epoch = self.epoch + 1
        self.log_current_epoch()
    
    def update_best_state(self):
        if self.best_epoch == -1 \
           or self.is_better_loss(self.loss_val[self.epoch]):
            self.best_epoch = self.epoch
            self.best_state = self.get_current_model_state()

    def print(self, epoch):
        logging.info(f"\nEpoch {epoch:4}: loss_train: {self.loss_trn[epoch]:.5f}"
                     f", loss_val: {self.loss_val[epoch]:.5f}, "
                     f"acc_train: {self.acc_trn[epoch]:.5f}, "
                     f"acc_val: {self.acc_val[epoch]:.5f}")
    
    def print_current_epoch(self):
        self.print(self.epoch)

    def print_best_epoch(self):
        self.print(self.best_epoch)

    def update(self, loss_trn: float, loss_val: float, acc_trn: float, 
               acc_val: float) -> None:
        """Call to add an iteration to the training tracker."""
        self._add(loss_trn, loss_val, acc_trn, acc_val)
        self.update_best_state() # Ensure class invariant
        if self.epoch % self.display_steps == 0:
            self.print_current_epoch()

    def get_statistics(self) -> Tuple[List[float], List[float], List[float],
                                      List[float], int]:
        """Return statistics including training epochs (1-based).
        
        Return order: loss_trn, loss_val, acc_trn, acc_val, training epochs
        """
        return self.loss_trn, self.loss_val, self.acc_trn, self.acc_val, \
               self.epoch + 1


@typechecked
def train_inductive(
          model: Optional[nn.Module], 
          label_prop: Optional[LP],
          X: TensorType["n", "d"], 
          A: TensorType["n", "n"], 
          y: TensorType["n"], 
          split_trn: np.ndarray, 
          split_val: np.ndarray,
          train_params: Dict[str, Any], 
          verbosity_params: Dict[str, Any], 
          _run: Optional[Run]=None
) -> TrainingTracker:
    """Train a model on a given graph inductively using a given trn/val split.
    
    Training is not batched but uses the whole graph in one forward pass.
    Inductively means that the validation nodes will be removed from the graph
    for training.

    Args:
        model (Optional[nn.Module]): Initialized model to train. If None 
            assumes (only) a label propagation model is provided.
        label_prop (Optional[nn.Module]): Label Propagation Module applied on 
            top of model-predictions. Can be disabled by setting to None or 
            used as stand-alone if model is set to None.
        X (TensorType["n", "d"]): Graph feature matrix.
        A (TensorType["n", "n"]): Adjacency matrix.
        y (TensorType["n"]): Node labels.
        split_trn (np.ndarray): Ids of training nodes.
        split_val (np.ndarray): Ids of validation nodes.
        train_params (Dict[str, Any]): Define training hyperparameters:
            - "lr": Learning Rate,
            - "weight_decay": Weight Decay,
            - "patience": #iteration when to stop after no improvement on 
                          validation data
            - "max_epochs": Maximial #iterations for training.
        verbosity_params (Dict[str, Any]): Has to include key "display_steps"
            to define at which training iteration statistics should be sent 
            to the standard output.
        _run (Optional[Run], optional): If set will be used to log statistics
            using sacred. Defaults to None.

    Returns:
        Tuple[List[float], List[float], List[float], List[float], int]: 
            Elements:
                loss_trn [epochs], 
                loss_val [epochs], 
                acc_trn [epochs], 
                acc_val [epochs], 
                epochs: training epochs (1-based)
    """
    if model is None:
        return train_transductive(None, label_prop, X, A, y, split_trn, 
                                  split_val, train_params, verbosity_params, 
                                  _run)

    train_tracker = TrainingTracker(model, verbosity_params, _run=_run)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params["lr"], 
                                    weight_decay=train_params["weight_decay"])
    loss = nn.CrossEntropyLoss()

    X_trn = X[split_trn, :]
    A_trn = A[split_trn, :]
    A_trn = A_trn[:,split_trn]
    y_trn = y[split_trn]

    for epoch in range(train_params["max_epochs"]):
        model.train()
        optimizer.zero_grad()

        logits = model(X_trn, A_trn)
        loss_train = loss(logits, y_trn)
        acc_trn = accuracy(logits, y_trn)

        with torch.no_grad():
            model.eval()
            logits = model(X, A)
            if label_prop is not None:
                logits = label_prop.smooth(logits, y_trn, split_trn, A)
            loss_val = loss(logits[split_val], y[split_val])
            acc_val = accuracy(logits, y, split_val)

        loss_train.backward()
        optimizer.step()
        
        train_tracker.update(loss_train.detach().item(), 
                            loss_val.detach().item(),
                            acc_trn, acc_val)
                            
        if epoch >= train_tracker.best_epoch + train_params["patience"]:
            break
                
    train_tracker.log_best_epoch()

    train_tracker.print_best_epoch()

    model.load_state_dict(train_tracker.get_best_model_state())
    return train_tracker


@typechecked
def train_transductive(
        model: Optional[nn.Module], 
        label_prop: Optional[LP],
        X: TensorType["n", "d"], 
        A: TensorType["n", "n"], 
        y: TensorType["n"], 
        split_trn: np.ndarray, 
        split_val: np.ndarray,
        train_params: Dict[str, Any], 
        verbosity_params: Dict[str, Any], 
        _run: Optional[Run]=None
) -> TrainingTracker:
    """Train a model on a given graph transductively using a given trn/val split.
    
    Training is not batched but uses the whole graph in one forward pass.

    Args:
        model (nn.Module): Initialized model to train.
        label_prop (Optional[nn.Module]): Label Propagation Module applied on 
            top of model-predictions. Can be disabled by setting to None or 
            used as stand-alone if model is set to None.
        X (TensorType["n", "d"]): Graph feature matrix.
        A (TensorType["n", "n"]): Adjacency matrix.
        y (TensorType["n"]): Node labels.
        split_trn (np.ndarray): Ids of training nodes.
        split_val (np.ndarray): Ids of validation nodes.
        train_params (Dict[str, Any]): Define training hyperparameters:
            - "lr": Learning Rate,
            - "weight_decay": Weight Decay,
            - "patience": #iteration when to stop after no improvement on 
                          validation data
            - "max_epochs": Maximial #iterations for training.
        verbosity_params (Dict[str, Any]): Has to include key "display_steps"
            to define at which training iteration statistics should be sent 
            to the standard output.
        _run (Optional[Run], optional): If set will be used to log statistics
            using sacred. Defaults to None.

    Returns:
        Tuple[List[float], List[float], List[float], List[float], int]: 
            Elements:
                loss_trn [epochs], 
                loss_val [epochs], 
                acc_trn [epochs], 
                acc_val [epochs], 
                epochs: training epochs (1-based)
    """
    assert model is not None or label_prop is not None

    train_tracker = TrainingTracker(model, verbosity_params, _run=_run)

    loss = nn.CrossEntropyLoss()

    if model is None:
        # Label Propagation achieves best result after one epoch (no training)
        logits = label_prop.smooth(None, y[split_trn], split_trn, A, 
                                   normalize=False)
        loss_train = loss(logits[split_trn], y[split_trn])
        loss_val = loss(logits[split_val], y[split_val])
        acc_trn = accuracy(logits, y, split_trn)
        acc_val = accuracy(logits, y, split_val)
        #train_tracker.update(-1, loss_val.detach().item(), -1, acc_val)
        train_tracker.update(loss_train.detach().item(), 
                             loss_val.detach().item(), acc_trn, acc_val)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=train_params["lr"], 
                                    weight_decay=train_params["weight_decay"])

        for epoch in range(train_params["max_epochs"]):
            model.train()
            optimizer.zero_grad()

            logits = model(X, A)
            if label_prop is not None:
                logits = label_prop.smooth(logits, y[split_trn], split_trn, A)
            loss_train = loss(logits[split_trn], y[split_trn])
            loss_val = loss(logits[split_val], y[split_val])

            loss_train.backward()
            optimizer.step()

            acc_trn = accuracy(logits, y, split_trn)
            acc_val = accuracy(logits, y, split_val)

            train_tracker.update(loss_train.detach().item(), 
                                loss_val.detach().item(),
                                acc_trn, acc_val)
                                
            if epoch >= train_tracker.best_epoch + train_params["patience"]:
                break
                
    train_tracker.log_best_epoch()

    train_tracker.print_best_epoch()

    if model is not None:
        model.load_state_dict(train_tracker.get_best_model_state())
    return train_tracker


@typechecked
def test(
        model: Optional[nn.Module], 
        label_prop: Optional[LP],
        X: TensorType["n", "d"], 
        A: TensorType["n", "n"], 
        y: TensorType["n"], 
        split_trn: np.ndarray, 
        split_val: np.ndarray,
        _run: Optional[Run]=None
) -> TrainingTracker:
    """Test and log given model and/or label propagation on given graph.

    Args:
        model (nn.Module): Initialized model to train.
        label_prop (Optional[nn.Module]): Label Propagation Module applied on 
            top of model-predictions. Can be disabled by setting to None or 
            used as stand-alone if model is set to None.
        X (TensorType["n", "d"]): Graph feature matrix.
        A (TensorType["n", "n"]): Adjacency matrix.
        y (TensorType["n"]): Node labels.
        split_trn (np.ndarray): Ids of training nodes.
        split_val (np.ndarray): Ids of validation nodes.
        _run (Optional[Run], optional): If set will be used to log statistics
            using sacred. Defaults to None.

    Returns:
        Tuple[List[float], List[float], List[float], List[float], int]: 
            Elements:
                loss_trn [epochs]: [-1] because test() does not train
                loss_val [epochs]: Validation loss on validation nodes.
                acc_trn [epochs]: [-1] because test() does not train
                acc_val [epochs]: Validation accuracy on validation nodes.
                epochs: 1 because no training is performed
    """
    assert model is not None or label_prop is not None

    train_tracker = TrainingTracker(model, {"display_steps": 1}, _run=_run)

    loss = nn.CrossEntropyLoss()

    if model is None:
        logits = label_prop.smooth(None, y[split_trn], split_trn, A, 
                                   normalize=False)
        loss_train = loss(logits[split_trn], y[split_trn])
        loss_val = loss(logits[split_val], y[split_val])
        acc_trn = accuracy(logits, y, split_trn)
        acc_val = accuracy(logits, y, split_val)
    else:
        model.eval()
        logits = model(X, A)
        if label_prop is not None:
            logits = label_prop.smooth(logits, y[split_trn], split_trn, A)
        loss_train = loss(logits[split_trn], y[split_trn])
        loss_val = loss(logits[split_val], y[split_val])
        acc_trn = accuracy(logits, y, split_trn)
        acc_val = accuracy(logits, y, split_val)

    train_tracker.update(loss_train.detach().item(), loss_val.detach().item(), 
                         acc_trn, acc_val)
    return train_tracker
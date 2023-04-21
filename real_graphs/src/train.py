# This file has been mostly taken from the work bei Geisler et al. 
# "Robustness of Graph Neural Networks at Scale" (NeurIPS, 2021) and adapted
# for this work: https://github.com/sigeisler/robustness_of_gnns_at_scale

import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from rgnn_at_scale.helper.utils import accuracy





def train_inductive(model, attr, adj_training, adj_validation, labels, idx_train, idx_val,
          lr, weight_decay, patience, max_epochs, display_step=50):
    """Train a model using inductive training.
    Parameters
    ----------
    model: torch.nn.Module
        Model which we want to train.
    attr: torch.Tensor [n, d]
        Dense attribute matrix.
    adj: torch.Tensor [n, n]
        Dense adjacency matrix.
    labels: torch.Tensor [n]
        Ground-truth labels of all nodes,
    idx_train: array-like [?]
        Indices of the training nodes.
    idx_val: array-like [?]
        Indices of the validation nodes.
    lr: float
        Learning rate.
    weight_decay : float
        Weight decay.
    patience: int
        The number of epochs to wait for the validation loss to improve before stopping early.
    max_epochs: int
        Maximum number of epochs for training.
    display_step : int
        How often to print information.
    Returns
    -------
    train_val, trace_val, trace_acc_train, trace_acc_val: list
        A tuple of lists of values of traning/validation loss and accuracy during training.
    """
    trace_loss_train = []
    trace_loss_val = []
    trace_acc_train = []
    trace_acc_val = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = np.inf

    model.train()
    for it in tqdm(range(max_epochs), desc='Training...'):
        #### Train step ####
        optimizer.zero_grad()
        model.train()
        logits = model(attr, adj_training)
        loss_train = F.cross_entropy(logits[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        #### Validation ####
        with torch.no_grad():
            model.eval()
            logits_val = model(attr, adj_validation)
            loss_val = F.cross_entropy(logits_val[idx_val], labels[idx_val])
            
        trace_loss_train.append(loss_train.detach().item())
        trace_loss_val.append(loss_val.detach().item())

        train_acc = accuracy(logits, labels, idx_train)
        val_acc = accuracy(logits_val, labels, idx_val)

        trace_acc_train.append(train_acc)
        trace_acc_val.append(val_acc)

        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = it
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
        else:
            if it >= best_epoch + patience:
                break

        if it % display_step == 0:
            logging.info(f'\nEpoch {it:4}: loss_train: {loss_train.item():.5f}, loss_val: {loss_val.item():.5f}, '
                         f'acc_train: {train_acc:.5f}, acc_val: {val_acc:.5f} ')

    # restore the best validation state
    model.load_state_dict(best_state)

    return trace_loss_val, trace_loss_train, trace_acc_val, trace_acc_train


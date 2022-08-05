from collections import Counter
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn

from src.graph_models import GRAPH_MODEL_TYPE
from src.utils import accuracy

def evaluate_robustness(model: nn.Module, 
                        graph_model: GRAPH_MODEL_TYPE, 
                        X_np: np.ndarray, 
                        A_np: np.ndarray, 
                        y_np: np.ndarray,
                        inductive_samples: int,
                        #attack_params: Dict[str, Any],
                        device: Union[torch.device, str]) -> Dict[str, Any]:
    """Evaluate the robustness of a given model on a synthetic graph.

    Evaluates different robustness metrics of a node-classifier trained on a 
    given graph generated from a specific generative graph model. Robustness
    metrics are calculated over repeated sampling of an additional node in an
    inductive manner and include:
        - General Robustness to edge insertions
        - Robustness w.r.t. Bayes Classifier

    Args:
        model (nn.Model): Node-classifier to investigation.
        graph_model (GRAPH_MODEL_TYPE): Generative graph model.
        X_np (np.ndarray, [n x d]): Feature matrix (assumed to be known during 
            training).
        A_np (np.ndarray, [n x n]): Adjacency matrix (assumed to be known 
            during training).
        y_np (np.ndarray, [n, ]): Labels of nodes (assumed to be known)
        inductive_samples (int): How often an additional node should be 
            inductively sampled.
        device: Calculation device for predictions.

    Returns:
        Dict[str, Any]: Robstness statistics
    """
    # Statistics Regarding Bayes & GNN Predictions
    c_acc_bayes = 0 # Count nodes correctly classified by bayes classifier
    c_acc_bayes_deg = Counter()  # Above but for each degree
    c_acc_bayes_structure = 0 # Count nodes separable by structure alone
    c_acc_bayes_structure_deg = Counter() # Above but for each degree
    c_acc_bayes_feature = 0 # Count nodes separable by features alone (degree 
                            # dependent doesn't make sense as features 
                            # independent of connections)
    c_acc_bayes_not_gnn = 0 # Decisions where BC correct but GNN wrong
    c_acc_bayes_not_gnn_deg = Counter() # Above but for each degree
    c_acc_gnn = 0 # Count nodes correctly classified by gnn
    c_acc_gnn_deg = Counter() # Above but for each degree
    c_acc_gnn_not_bayes = 0 # Decisions where GNN correctly says true even 
                            # though BC violated
    c_acc_gnn_not_bayes_deg = Counter() # Above but for each degree
    c_acc_bayes_gnn = 0 # Count nodes correctly classified by bc & gnn
    c_acc_bayes_gnn_deg = Counter() # Above but for each degree
    c_degree_total = Counter() # Count degrees of all generated nodes
    # Statistics Regarding Bayes & GNN Robustness
    c_bayes_robust = dict() # Degree-dependend robustness BC
    c_gnn_robust = dict() # Degree-dependend robustness GNN
    c_bayes_gnn_robust = dict() # Degree-dependend robustness of GNN w.r.t. BC
    c_bayes_robust_when_both = dict() # Degree-dependend robustness of Bayes on 
                                      # GNN w.r.t. Bayes Nodes
    c_gnn_robust_when_both = dict() # Degree-dependend robustness of GNN on GNN 
                                    # w.r.t. Bayes Nodes
    c_bayes_higher_robust = 0 # Number of times BC is more robust than GNN
    c_gnn_higher_robust = 0 # Number of times GNN is "overly robust"
    c_bayes_gnn_equal_robust = 0 # Number of times GNN has perfect robustness 
                                 # w.r.t. BC

    n = y_np.size
    model.eval()
    for i in range(inductive_samples):
        # ToDo: Create empty X_, A_, y_ templates & always only fill last row
        X, A, y = graph_model.sample_conditional(1, X_np, A_np, y_np)
        deg_n = np.sum(A[:,n])
        c_degree_total[deg_n] += 1
        # Statistics Bayes Classifier
        feature_separable, _ = graph_model.feature_separability(X, y, [n])
        structure_separable, _ = graph_model.structure_separability(A, y, [n])
        bayes_separable, _ = graph_model.likelihood_separability(X, A, y, [n])
        if bayes_separable:
            c_acc_bayes += 1
            c_acc_bayes_deg[deg_n] += 1
        if structure_separable:
            c_acc_bayes_structure += 1
            c_acc_bayes_structure_deg[deg_n] += 1
        if feature_separable:
            c_acc_bayes_feature += 1
        # Calculate GNN-prediction
        X_gpu = torch.tensor(X, dtype=torch.float32, device=device)
        A_gpu = torch.tensor(A, dtype=torch.float32, device=device)
        y_gpu = torch.tensor(y, device=device)
        logits = model(X_gpu, A_gpu)
        gnn_separable = round(accuracy(logits, y_gpu, n))
        # Statistics Prediction
        if gnn_separable:
            c_acc_gnn += 1
            c_acc_gnn_deg[deg_n] += 1
            if bayes_separable:
                c_acc_bayes_gnn += 1
                c_acc_bayes_gnn_deg[deg_n] += 1
            else:
                c_acc_gnn_not_bayes += 1
                c_acc_gnn_not_bayes_deg[deg_n] += 1
        elif bayes_separable:
            c_acc_bayes_not_gnn += 1
            c_acc_bayes_not_gnn_deg[deg_n] += 1
        # Investigate Robustness
        """
        c_robustness = 0
        bayes_separable_new = 0
        gnn_separable_new = 0
        # gnn w.r.t. bayes count possible?
        gnn_wrt_bayes_setting = False
        if bayes_separable and gnn_separable:
            gnn_wrt_bayes_setting = True
        attack = LocalAttack(n, X_, A_, y_, method = attack_method)
        while bayes_separable or gnn_separable:
            j = attack.add_adversarial_edge()
            if j != -1:
                A_gpu[n, j] = 1
                A_gpu[j, n] = 1
            else:
                assert not bayes_separable
                assert gnn_separable
            # Robustness of BC
            if bayes_separable:
                bayes_separable_new, _ = csbm.likelihood_separability(X_, A_, y_, [n])
                if not bayes_separable_new:
                    if deg_n not in c_bayes_robust:
                        c_bayes_robust[deg_n] = []
                    c_bayes_robust[deg_n].append(c_robustness)
                    if gnn_wrt_bayes_setting:
                        if deg_n not in c_bayes_robust_when_both:
                            c_bayes_robust_when_both[deg_n] = []
                        c_bayes_robust_when_both[deg_n].append(c_robustness)
            # Robustness of GNN
            if gnn_separable:
                logits = model(X_gpu, A_gpu)
                gnn_separable_new = round(accuracy(logits, y_gpu, n))
                if not gnn_separable_new or j == -1:
                    if deg_n not in c_gnn_robust:
                        c_gnn_robust[deg_n] = []
                    c_gnn_robust[deg_n].append(c_robustness)
                    if gnn_wrt_bayes_setting:
                        if deg_n not in c_gnn_robust_when_both:
                            c_gnn_robust_when_both[deg_n] = []
                        c_gnn_robust_when_both[deg_n].append(c_robustness)
            # Robustness of GNN w.r.t. BC
            if bayes_separable and gnn_separable:
                if deg_n not in c_bayes_gnn_robust:
                    c_bayes_gnn_robust[deg_n] = []
                if not bayes_separable_new and not gnn_separable_new:
                    c_bayes_gnn_equal_robust += 1
                    c_bayes_gnn_robust[deg_n].append(c_robustness)
                if bayes_separable_new and not gnn_separable_new:
                    c_bayes_higher_robust += 1
                    c_bayes_gnn_robust[deg_n].append(c_robustness)
                if not bayes_separable_new and gnn_separable_new:
                    c_gnn_higher_robust += 1
                    c_bayes_gnn_robust[deg_n].append(c_robustness)

            bayes_separable = bayes_separable_new
            gnn_separable = gnn_separable_new if j != -1 else False
            c_robustness += 1
        """
    # Postprocess robustness counts to averages
    """avg_bayes_robust = {}
    med_bayes_robust = {}
    std_bayes_robust = {}
    max_bayes_robust = {}
    for degree in c_acc_bayes_deg:
        avg_bayes_robust[degree] = np.mean(c_bayes_robust[degree])
        med_bayes_robust[degree] = np.median(c_bayes_robust[degree])
        std_bayes_robust[degree] = np.std(c_bayes_robust[degree])
        max_bayes_robust[degree] = np.max(c_bayes_robust[degree])
    avg_gnn_robust = {}
    med_gnn_robust = {}
    std_gnn_robust = {}
    max_gnn_robust = {}
    for degree in c_acc_gnn_deg:
        avg_gnn_robust[degree] = np.mean(c_gnn_robust[degree])
        med_gnn_robust[degree] = np.median(c_gnn_robust[degree])
        std_gnn_robust[degree] = np.std(c_gnn_robust[degree])
        max_gnn_robust[degree] = np.max(c_gnn_robust[degree])
    avg_bayes_gnn_robust = {}
    for degree in c_acc_bayes_gnn_deg:
        avg_bayes_gnn_robust[degree] = np.mean(c_bayes_gnn_robust[degree])
    avg_bayes_robust_when_both = {}
    for degree in c_acc_bayes_gnn_deg:
        avg_bayes_robust_when_both[degree] = np.mean(c_bayes_robust_when_both[degree])
    avg_gnn_robust_when_both = {}
    for degree in c_acc_bayes_gnn_deg:
        avg_gnn_robust_when_both[degree] =  np.mean(c_gnn_robust_when_both[degree])"""
    
    return [(c_acc_bayes, c_acc_gnn, c_acc_bayes_structure, c_acc_bayes_feature, 
            c_acc_bayes_gnn, c_acc_bayes_not_gnn, c_acc_gnn_not_bayes)]
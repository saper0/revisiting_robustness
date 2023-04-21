import torch
from torch_sparse import SparseTensor
import numpy as np


def add_edges_and_evaluate(adj, attr, model, lp, lp_input, edges, node):
    '''Add edges to the graph's adjacency and evaluate the model under this perturbed graph.
    Parameters
    ----------
    adj: torch_sparse.SparseTensor [n,n]
    attr: torch.Tensor [n, d]
        Dense attribute matrix.
    model: torch.nn.Module
        Model to evaluate.
    lp: src.models.LP or None
        Label propaghation (if not None)
    lp_input: dict[str] 
        keys are 'y_true' and 'mask' and provide known (training) labels and a binary mask indicating their node's location
    node: int 
        target node 
    edges: torch.Tensor [?]
        nodes to connect the target node to
    Returns
    -------
    pred: int
        prediction of the model for the target node when considering the perturbed adjacency
    '''
    col = torch.cat([edges, node*torch.ones_like(edges)])
    row = torch.cat([node*torch.ones_like(edges), edges])
    edge_mat = SparseTensor(row=row, col=col, value=torch.ones_like(row), sparse_sizes=(adj.sparse_size(dim=0),adj.sparse_size(dim=1)))
    adj_pert = edge_mat + adj
    with torch.no_grad():
        logits = model(attr, adj_pert)
        if not lp is None:
            logits = lp.smooth(y_soft = logits, A = adj_pert, **lp_input)
    pred = logits[node].argmax()
    return pred

def get_candidate_edges(adj, attr, labels, node, heuristic, projection = None):
    '''For each class (except the node's class) retrieve 128 candidate nodes to connect this node to.
    ----------
    adj: torch_sparse.SparseTensor [n,n]
    attr: torch.Tensor [n, d]
        Dense attribute matrix.
    labels: torch.Tensor [n]
        Ground-truth labels of all nodes
    node: int 
        target node 
    heuristic: 'l1' or 'l2 
        heuristic for selecting edge perturbations (min l1 or l2 distance between (projected) node attributes implemented)
    projection: lambda function (optional)
        projection to apply to the attributes before evaluating the heuristic
    Returns
    -------
    classes: list[int]
        classes 
    candidate_edges: list[torch.Tensor [128]]
        list of node idx from class in classes to connect the class to
    average_distance: list[float]
        average distance of node attributes from candidate edges to node per class'''
    # init results
    candidate_edges = []
    average_distance = []
    with torch.no_grad():
        if not projection is None:
            attr = projection(attr)
        if heuristic == "l2":
            dist = torch.linalg.norm(attr-attr[node], dim=1, ord=2)
        elif heuristic == "l1":
            dist = torch.linalg.norm(attr-attr[node], dim=1, ord=1)
    # all classes that have more than 128 nodes 
    classes = [x for x in labels.unique() if (x != labels[node]) and ((labels == x).sum()>=128)]
    # current neighbours
    neighbors, _, _ = adj[int(node)].t().coo()
    is_not_neighbour = torch.ones(adj.sparse_size(dim=0),device=adj.device()).bool()
    is_not_neighbour[neighbors]=False
    # iterate over classes
    for target_class in classes:
        target_idx = ((labels == target_class) & is_not_neighbour).nonzero().flatten()
        class_dist = dist[target_idx]
        nns_for_class = torch.topk(class_dist, largest=False, k=128)
        # save idx of nearest class members
        idx_nns = target_idx[nns_for_class.indices]
        candidate_edges.append(idx_nns)
        # evaluate average distance of topk values (or less than all topk??)
        m = 3*len(neighbors)
        distance = nns_for_class.values[:m].mean() # choose m something like 2*degree???
        average_distance.append(distance.detach().cpu().numpy())
    return classes, candidate_edges, average_distance

def evaluate_node_binary(adj, attr, model, lp, lp_input, labels, node, candidate_edges, m = 7):
    '''Binary search for the largest number of edge insertions for which the model's prediction for node does not change.
    Parameters
    ----------
    adj: torch_sparse.SparseTensor [n,n]
    attr: torch.Tensor [n, d]
        Dense attribute matrix.
    model: torch.nn.Module
        Model to evaluate.
    lp: src.models.LP or None
        Label propaghation (if not None)
    lp_input: dict[str] 
        keys are 'y_true' and 'mask' and provide known (training) labels and a binary mask indicating their node's location
    labels: torch.Tensor [n]
        Ground-truth labels of all nodes
    node: int 
        target node 
    candidate_edges: torch.Tensor [128]
        nodes to connect the target node to
    m: int
        size of search space 2**m
    Returns
    -------
    pred: int
        largest observed number of edge insertions for which the model's prediction for node did not change
    '''
    n = 2**m
    # original pred
    pred = labels[node]
    best_pert = 0
    # get candidate edges
    current_pert = n/2
    # binary search
    for i in range(m):
        edges = candidate_edges[:int(current_pert)]
        new_pred = add_edges_and_evaluate(adj, attr, model, lp, lp_input, edges, node)
        if new_pred == pred:
            best_pert = current_pert
            current_pert += n / 2**(i+2)
        if new_pred != pred:
            current_pert -= n / 2**(i+2)
    return best_pert

def evaluate_node(adj, attr, model, lp, lp_input, labels, node, candidate_edges, m = 7):
    '''Linear search for the largest number of edge insertions for which the model's prediction for node does not change.
    Parameters
    ----------
    adj: torch_sparse.SparseTensor [n,n]
    attr: torch.Tensor [n, d]
        Dense attribute matrix.
    model: torch.nn.Module
        Model to evaluate.
    lp: src.models.LP or None
        Label propaghation (if not None)
    lp_input: dict[str] 
        keys are 'y_true' and 'mask' and provide known (training) labels and a binary mask indicating their node's location
    labels: torch.Tensor [n]
        Ground-truth labels of all nodes
    node: int 
        target node 
    candidate_edges: torch.Tensor [128]
        nodes to connect the target node to
    m: int
        size of search space 2**m
    Returns
    -------
    pred: int
        largest observed number of edge insertions for which the model's prediction for node did not change
    '''
    n = 2**m
    # original pred
    pred = labels[node]
    # binary search
    for i in range(n):
        edges = candidate_edges[:i+1]
        new_pred = add_edges_and_evaluate(adj, attr, model, lp, lp_input, edges, node)
        if new_pred != pred:
            return i
    return n-1

def evaluate_node_all_classes(adj, attr, model, lp, lp_input, labels, node, heuristic='l2', projection=None):
    '''evaluate a model's robustness w.r.t a single node and edge insertions connecting this node to nodes from a different class. 
    This is repeated for all classes
    Parameters
    ----------
    adj: torch_sparse.SparseTensor [n,n]
    attr: torch.Tensor [n, d]
        Dense attribute matrix.
    model: torch.nn.Module
        Model to evaluate.
    lp: src.models.LP or None
        Label propaghation (if not None)
    lp_input: dict[str] 
        keys are 'y_true' and 'mask' and provide known (training) labels and a binary mask indicating their node's location
    labels: torch.Tensor [n]
        Ground-truth labels of all nodes
    node: int 
        target node 
    heuristic: 'l1' or 'l2 
        heuristic for selecting edge perturbations (min l1 or l2 distance between (projected) node attributes implemented)
    projection: lambda function (optional)
        projection to apply to the attributes before evaluating the heuristic
    Returns
    -------
    result_dict: dictionary
        each entry corresponds to a node and includes the node's index as well as robustness w.r.t. all classes
    '''
    # init result dict
    result_dict = {'idx': node}
    #get candidate edges for all classes
    classes, candidate_edges, average_distance = get_candidate_edges(adj, attr, labels, node, heuristic=heuristic, projection=projection)
    for adv_class, edges in zip(classes, candidate_edges):
        count = evaluate_node(adj, attr, model, lp, lp_input, labels, node, edges)
        result_dict[int(adv_class)] = count
    return result_dict

def evaluate_node_minmax_classes(adj, attr, model, lp, lp_input, labels, node, heuristic='l2', projection=None):
    '''evaluate a model's robustness w.r.t a single node and edge insertions connecting this node to nodes from a different class. 
    This is repeated only for the classes having minimal and maximal distance to the node (w.r.t. node attributes) 
    Parameters
    ----------
    adj: torch_sparse.SparseTensor [n,n]
    attr: torch.Tensor [n, d]
        Dense attribute matrix.
    model: torch.nn.Module
        Model to evaluate.
    lp: src.models.LP or None
        Label propaghation (if not None)
    lp_input: dict[str] 
        keys are 'y_true' and 'mask' and provide known (training) labels and a binary mask indicating their node's location
    labels: torch.Tensor [n]
        Ground-truth labels of all nodes
    node: int 
        target node 
    heuristic: 'l1' or 'l2 
        heuristic for selecting edge perturbations and classes to evaluate (min l1 or l2 distance between (projected) node attributes implemented)
    projection: lambda function (optional)
        projection to apply to the attributes before evaluating the heuristic and selecting classes
    Returns
    -------
    result_dict: dictionary
        each entry corresponds to a node and includes the node's index as well as robustness w.r.t. the two selected classes
    '''
    # init result dict
    result_dict = {'idx': node}
    #get candidate edges for all classes
    classes, candidate_edges, average_distance = get_candidate_edges(adj, attr, labels, node, heuristic=heuristic, projection=projection)
    min_class, max_class = np.argmin(average_distance), np.argmax(average_distance)
    count_min = evaluate_node_binary(adj, attr, model, lp, lp_input, labels, node, candidate_edges[min_class])
    result_dict[int(classes[min_class])] = count_min
    count_max = evaluate_node_binary(adj, attr, model, lp, lp_input, labels, node, candidate_edges[max_class])
    result_dict[int(classes[max_class])] = count_max
    return result_dict
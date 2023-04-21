import logging
from typing import Any, Dict, Union

import numpy as np
from sacred import Experiment

import torch

from rgnn_at_scale.helper.io import Storage
from rgnn_at_scale.helper.utils import accuracy
from rgnn_at_scale.helper import utils
import torch.nn.functional as F
from tqdm.auto import tqdm
import pandas as pd
import copy

from src.models.lp import LP
from src.models.dummy_model import Dummy_model
from src.helpers import prepare_data
from src.evaluate import evaluate_node_all_classes, evaluate_node_minmax_classes

try:
    import seml
except:  # noqa: E722
    seml = None


ex = Experiment()

if seml is not None:
    seml.setup_logger(ex)


@ex.config
def config():
    overwrite = None

    if seml is not None:
        db_collection = None
        if db_collection is not None:
            ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

    # default params
    data_dir = './datasets'
    dataset = 'cora_ml'
    make_undirected = True
    binary_attr = False
    data_device = 0
    n_per_class = 30
    dense_split = True

    device = 0
    seed = 0

    artifact_dir = 'cache_debug'
    model_storage_type = 'pretrained'

    display_steps = 100
    debug_level = "info"


@ex.automain
def run(data_dir: str, dataset: str, model_label: str, binary_attr: bool, lp_params: Dict[str, Any], projection: bool,
        make_undirected: bool, seed: int, artifact_dir: str, model_storage_type: str, n_per_class: Union[list,int], metric: str,
        device: Union[str, int], data_device: Union[str, int], display_steps: int, debug_level: str, dense_split:bool):
    """
    Instantiates a sacred experiment executing 
    """
    if debug_level is not None and isinstance(debug_level, str):
        logger = logging.getLogger()
        if debug_level.lower() == "info":
            logger.setLevel(logging.INFO)
        if debug_level.lower() == "debug":
            logger.setLevel(logging.DEBUG)
        if debug_level.lower() == "critical":
            logger.setLevel(logging.CRITICAL)
        if debug_level.lower() == "error":
            logger.setLevel(logging.ERROR)

    logging.info({
        'dataset': dataset, 'model_label': model_label, 'binary_attr': binary_attr,
        'make_undirected': make_undirected, 'seed': seed, 'artifact_dir': artifact_dir,
        'model_storage_type': model_storage_type, 'device': device,
        'display_steps': display_steps, 'data_device': data_device,
        'n_per_class': n_per_class
    })

    torch.manual_seed(seed)
    np.random.seed(seed)

    # load data
    attr, adj, idx_train, idx_val, idx_test, labels = prepare_data(dataset=dataset, 
                                                                    data_device=data_device,
                                                                    data_dir= data_dir, 
                                                                    dense_split= dense_split, 
                                                                    n_per_class= n_per_class, 
                                                                    make_undirected=make_undirected, 
                                                                    binary_attr= binary_attr)

    #n_features = attr.shape[1]
    n_classes = int(labels[~labels.isnan()].max() + 1)

    logging.info("Memory Usage after loading the dataset:")
    logging.info(utils.get_max_memory_bytes() / (1024 ** 3))

    # subsample test set to reduce number of evaluated nodes on arxiv
    if dataset == 'ogbn-arxiv':
        # compute n_per_class
        n_total = 1500
        bins = labels[idx_test].bincount()
        bins_ratio = bins/bins.sum()
        n_per_class = (n_total*bins_ratio).ceil()
        # subsample test set
        idx_keep = []
        test_map = torch.zeros_like(labels)
        test_map[idx_test]=1
        for c,n in enumerate(n_per_class):
            idx = ((labels == c) & test_map).nonzero().flatten()
            idx_keep.append(idx[torch.randperm(len(idx))[:int(n)]])
        idx_test = torch.cat(idx_keep).cpu().numpy()

    # load model and label propagation
    if model_label == 'LP':
        model = Dummy_model(n_classes, device)
    else:
        store = Storage(cache_dir = artifact_dir)
        model_params = {"dataset": dataset, "binary_attr": binary_attr, "make_undirected": make_undirected, "seed": seed, "label": model_label.split('+')[0]}
        if dataset != 'ogbn-arxiv': # if not on arxiv also filter for split type
            model_params.update({"dense_split": dense_split, "n_per_class": n_per_class})
        models_and_hyperparams = store.find_models(model_storage_type, model_params)
        logging.info(model_params)
        logging.info(models_and_hyperparams)
        assert len(models_and_hyperparams)==1
        model = models_and_hyperparams[0][0].to(device)
    if 'LP' in model_label:
        logging.info('using LP')
        lp = LP(**lp_params, num_classes = n_classes)
        mask = torch.zeros_like(labels)
        mask[idx_train]=1
        lp_input = {'y_true': labels[idx_train], 'mask': mask.bool()}
    else:
        logging.info('not using LP')
        lp = None
        lp_input = None

    # forward pass for calculating model accuracy and determining all correctly classified test nodes
    model.eval()
    with torch.no_grad():
        logits = model(attr, adj)
        if not lp is None:
            logits = lp.smooth(y_soft=logits, A = adj, **lp_input)
        probs = F.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)
    acc = accuracy(logits, labels, idx_test)
    logging.info(f'model accuracy on test nodes: {acc}')

    idx_true = idx_test[(labels[idx_test] == preds[idx_test]).nonzero().cpu()].flatten()

    # define projection for determining candidate edges
    if projection:
        if 'GCN' in model_label:
            model_copy = copy.deepcopy(model)
            projection = lambda x : model_copy.layers[0].gcn_0.lin.to(0)(x)
        elif 'APPNP' in model_label:
            model_copy = copy.deepcopy(model)
            projection = lambda x : model_copy.lin1.to(0)(x)
        elif model_label == 'LP': # in case of LP use the GCN-projection by default
            store = Storage(cache_dir = artifact_dir)
            model_params = {"dataset": dataset, "binary_attr": binary_attr, "make_undirected": make_undirected, "seed": seed, "label": 'GCN'}
            if dataset != 'ogbn-arxiv': # if not on arxiv filter for split type
                model_params.update({"dense_split": dense_split, "n_per_class": n_per_class})
            models_and_hyperparams = store.find_models(model_storage_type, model_params)
            assert len(models_and_hyperparams)==1
            model_copy = models_and_hyperparams[0][0].to(device)
            projection = lambda x : model_copy.layers[0].gcn_0.lin.to(0)(x)
        else: 
            assert False
    else:
        projection = None

    # evaluate for all correctly classified test nodes
    df = pd.DataFrame()
    for node in tqdm(idx_true, desc='evaluating all correctly classified test nodes ...'):
        if dataset == 'ogbn-arxiv':
            result_dict = evaluate_node_minmax_classes(adj, attr, model, lp, lp_input, labels, node, heuristic=metric, projection=projection)
        else:
            result_dict = evaluate_node_all_classes(adj, attr, model, lp, lp_input, labels, node, heuristic=metric, projection=projection)
        df = df.append(result_dict, ignore_index=True)

    # add additional info (degree, confidence, min, max)
    df_2 = pd.DataFrame()
    df_2['degree'] = adj.sum(dim=1)[idx_true].cpu().numpy()
    df_2['idx'] = idx_true
    df_2['label'] = labels[idx_true].cpu()
    df_2['confidence'] = probs.max(dim=1).values[idx_true].detach().cpu().numpy()
    df_full = df.merge(df_2, how='left')

    return {'result_df': df_full.to_dict(), 'model_accuracy': acc}
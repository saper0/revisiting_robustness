from pymongo import MongoClient
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd


URI = "mongodb://gosl:Wuyg6KTV@fs.kdd.in.tum.de:27017/gosl?authMechanism=SCRAM-SHA-1"


class Experiment:
    """An experiment refers to the (robustness) results optained by a 
    particular model on K averaged over multiple seeds."""
    def __init__(self, result_list: List[Dict[str, Any]]):
        assert len(result_list) > 0
        self.individual_results = result_list
        print(self.individual_results[0]["result"])


class ExperimentLoader:
    """Manages a mongodb-collection-connection and access to its data.
    
    Assumes same experiments with different seeds are stored consecutively.
    """
    def __init__(self, collection="runs", uri=URI):
        """Establish connection to a given mongodb collection."""
        self.client = MongoClient(uri)
        self.db = self.client.gosl
        self.collection = self.db[collection]

    def load_experiment_dict(self, id: int) -> Dict[str, Any]:
        """Return result-dict of experiment with ID id."""
        return self.collection.find_one({'_id': id})

    def load_experiment(self, start_id: int, n_seeds: int) -> Experiment:
        """Return experiment with ID id."""
        exp_dict_l = []
        for i in range(n_seeds):
            exp_dict_l.append(self.load_experiment_dict(start_id + i))
        return Experiment(exp_dict_l)

    def load_experiments(
            self, start_id: int, end_id: int, n_seeds: int, label: str=None
        ) -> List[Experiment]:
        """Return Experiments between start_id and end_id.
        
        Assumes that one experiment consists of multiple seeds which are stored
        consecutively in the mongodb. 
        """
        experiment_ids = [i for i in range(start_id, end_id + 1, n_seeds)]
        experiments = [self.load_experiment(i, n_seeds) for i in experiment_ids]
        if label is not None:
            filtered_experiments = []
            for experiment in experiments:
                if experiment.label == label:
                    filtered_experiments.append(experiment)
            return filtered_experiments
        else:
            return experiments


class ExperimentManager:
    """Administrates access and visualization of robustness experiments."""
    def __init__(self, experiments: List[Dict[str, Any]]):
        self.experiments_dict = {}
        self.load(experiments)

    def load(self, experiments):
        """Populates experiments_dict from stored results in MongoDB."""
        exp_loader = ExperimentLoader()
        for exp_spec in experiments:
            exp_list = exp_loader.load_experiments(exp_spec["start_id"], 
                                                   exp_spec["end_id"], 
                                                   exp_spec["n_seeds"], 
                                                   exp_spec["label"])
            for exp in exp_list:
                if exp.label not in self.experiments_dict:
                    self.experiments_dict[exp.label] = {}
                self.experiments_dict[exp.label][exp.K] = exp

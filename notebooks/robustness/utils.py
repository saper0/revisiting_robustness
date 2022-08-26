from pymongo import MongoClient
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd


URI = "mongodb://gosl:Wuyg6KTV@fs.kdd.in.tum.de:27017/gosl?authMechanism=SCRAM-SHA-1"


def assert_equal_dicts(d1: Dict[str, Any], d2: Dict[str, Any]):
    """Checks if every key of d1 is in d2 and they store same value."""
    for key in d1:
        assert key in d2
        assert d1[key] == d2[key]


def append_dict(source: Dict[str, Any], target: Dict[str, Any], 
                exclude_keys: List[str] = []):
    """Appends each element in source-dict to same key-element in target. 
    
    Assumes source-dict has as element a single number or again a dict. 
    Possibility to ignore keys given in exclude_keys.
    """
    for key, item in source.items():
        if key not in exclude_keys:
            if isinstance(item, dict):
                if key not in target:
                    target[key] = {}
                append_dict(item, target[key])
            else: 
                if key not in target:
                    target[key] = []
                target[key].append(item)


def average_dict(target: Dict[str, Any]):
    """Recursively averages every value-holding element in target-dictionary. 
    
    Assumes target dictionary holds str-keys and either list or dict items. 
    Every list-item is averaged. If element is a dict, assumes it is 
    shallow only containing lists.
    """
    keys = [key for key in target]
    for key in keys:
        item = target[key]
        if isinstance(item, dict):
            avg_dict, std_dict = average_subdict(item)
            target["avg_" + key] = avg_dict
            target["std_" + key] = std_dict
        else:
            assert len(target[key]) > 0
            target["avg_" + key] = np.mean(item)
            target["std_" + key] = np.std(item)

def average_subdict(subdict: Dict[str, Any]):
    """Return a dictionary with averaged and a dictionary with standard deviation
    values for each element."""
    keys = [key for key in subdict]
    avg_dict = {}
    std_dict = {}
    for key in keys:
        item = subdict[key]
        assert not isinstance(item, dict)
        assert len(subdict[key]) > 0
        avg_dict[key] = np.mean(item)
        std_dict[key] = np.std(item)
    return avg_dict, std_dict


class Experiment:
    """An experiment refers to the (robustness) results optained by a 
    particular model on K averaged over multiple seeds."""
    def __init__(self, experiment_list: List[Dict[str, Any]]):
        assert len(experiment_list) > 0
        self.individual_experiments = experiment_list
        self.id = experiment_list[0]["_id"]
        self.hyperparameters = experiment_list[0]["config"]
        self.label = self.hyperparameters["model_params"]["label"]
        self.K = self.hyperparameters["data_params"]["K"]
        Experiment.assert_same_hyperparameters(self.individual_experiments)
        self.average_result_statistics()
        self.calculate_robustness_metrics()

    @staticmethod
    def assert_same_hyperparameters(
        individual_experiments: List[Dict[str, Any]]
    ) -> None:
        """Sanity check if all given experiments indeed have the same 
        configuration."""
        data_params_l = []
        model_params_l = []
        train_params_l = []
        for experiment in individual_experiments:
            data_params_l.append(experiment["config"]["data_params"])
            model_params_l.append(experiment["config"]["model_params"])
            train_params_l.append(experiment["config"]["train_params"])
        for i in range(1, len(individual_experiments)):
            assert_equal_dicts(data_params_l[0], data_params_l[i])
            assert_equal_dicts(model_params_l[0], model_params_l[i])
            assert_equal_dicts(train_params_l[0], train_params_l[i])
    
    def average_result_statistics(self):
        """Average prediction statistics and robustness statistics calculated 
        from the raw data for each seed."""
        self.prediction_statistics = {}
        self.robustness_statistics = {}
        final_training_loss_l = []
        final_training_accuracy_l = []
        final_validation_loss_l = []
        final_validation_accuracy_l = []
        for experiment in self.individual_experiments:
            result = experiment["result"]
            append_dict(result["prediction_statistics"], 
                        self.prediction_statistics)
            exclude_keys = ["c_bayes_robust", "c_gnn_robust", 
                            "c_gnn_wrt_bayes_robust", "c_bayes_robust_when_both",
                            "c_gnn_robust_when_both", "c_degree_total"]
            append_dict(result["robustness_statistics"], 
                        self.robustness_statistics, exclude_keys)
            final_training_loss_l.append(result["final_training_loss"])
            final_training_accuracy_l.append(result["final_training_accuracy"])
            final_validation_loss_l.append(result["final_validation_loss"])
            final_validation_accuracy_l.append(result["final_validation_accuracy"])
        average_dict(self.prediction_statistics)
        average_dict(self.robustness_statistics)
        self.avg_training_loss = np.mean(final_training_loss_l)
        self.std_training_loss = np.std(final_training_loss_l)
        self.avg_training_accuracy = np.mean(final_training_accuracy_l)
        self.std_training_accuracy = np.std(final_training_accuracy_l)
        self.avg_validation_loss = np.mean(final_validation_loss_l)
        self.std_validation_loss = np.std(final_validation_loss_l)
        self.avg_validation_accuracy = np.mean(final_validation_accuracy_l)
        self.std_validation_accuracy = np.std(final_validation_accuracy_l)
        if False:
            print(f"{self.label} on K={self.K} has {self.avg_training_accuracy*100:.1f}"
                f"+-{self.std_training_accuracy*100:.1f}% trn acc and "
                f"{self.avg_validation_accuracy*100:.1f}+-{self.std_validation_accuracy*100:.1f}%"
                f" val acc.")

                
    def calculate_robustness_metrics(self):
        self.count_zero_degree_nodes(verbose=False)
        self.measure_robustness(verbose=False)

    def count_zero_degree_nodes(self, verbose):
        """Count how many 0-degree nodes in V'.
        
        0-degree nodes have to be removes from calculation of robustness 
        metrics.
        """
        count_deg_0 = 0
        count_total = 0
        for experiment in self.individual_experiments:
            result = experiment["result"]
            robustness_stats = result["robustness_statistics"]
            for key, item in robustness_stats["c_bayes_robust_when_both"].items():
                if key == "0":
                    count_deg_0 += len(item)
                count_total += len(item)
        count_deg_0 = count_deg_0 / len(self.individual_experiments)
        count_total = count_total / len(self.individual_experiments)
        if verbose:
            print(f"{self.label} on K={self.K} has {count_deg_0}/{count_total}"
                    f" ({count_deg_0/count_total*100:.1f}%) deg 0 nodes removed"
                    f" for robustness metric calculation")

    def measure_robustness(self, verbose):
        """Calculate over-robustness metrics and average over seeds."""
        over_robustness_l = []
        adv_robustness_l = []
        for experiment in self.individual_experiments:
            result = experiment["result"]
            robustness_stats = result["robustness_statistics"]
            g_wrt_y = robustness_stats["c_bayes_robust_when_both"]
            f_wrt_y = robustness_stats["c_gnn_robust_when_both"]
            f_wrt_g = robustness_stats["c_gnn_wrt_bayes_robust"]
            over_robustness = 0
            adv_robustness = 0
            c_nodes = 0
            for deg in g_wrt_y:
                if deg == "0":
                    continue
                for g_wrt_y_i, f_wrt_y_i, f_wrt_g_i in \
                    zip(g_wrt_y[deg], f_wrt_y[deg], f_wrt_g[deg]):
                    over_robustness += (f_wrt_y_i - g_wrt_y_i) / int(deg)
                    adv_robustness += (g_wrt_y_i - f_wrt_g_i) / int(deg)
                c_nodes += len(g_wrt_y[deg])
            over_robustness_l.append(over_robustness / c_nodes)
            adv_robustness_l.append(adv_robustness / c_nodes)
        self.avg_over_robustness = np.mean(over_robustness_l)
        self.std_over_robustness = np.std(over_robustness_l)
        self.avg_adv_robustness = np.mean(adv_robustness_l)
        self.std_adv_robustness = np.std(adv_robustness_l)
        if self.hyperparameters["attack_params"]["attack"] == "random" and verbose:
            print(f"{self.label} on K={self.K} has {self.avg_over_robustness:.2f}+-"
                    f"{self.std_over_robustness:.2f} over-robustness")
        if self.hyperparameters["attack_params"]["attack"] == "l2" and verbose:
            print(f"{self.label} on K={self.K} has {self.avg_adv_robustness:.2f}+-"
                    f"{self.std_adv_robustness:.2f} adv. robustness")

class ExperimentManager:
    """Administrates access and visualization of robustness experiments.
    
    Assumes same experiments with different seeds are stored consecutively.
    """
    def __init__(self, experiments: List[Dict[str, Any]], collection="runs", 
                 uri=URI):
        """Establish connection to a given mongodb collection. 
        
        Load and administrate data from specified experiments.
        """
        self.client = MongoClient(uri)
        self.db = self.client.gosl
        self.collection = self.db[collection]
        self.load(experiments)

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
        """Return Experiments between start_id and end_id with given label.
        
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

    def load(self, experiments) -> None:
        """Populates experiments_dict from stored results in MongoDB.
        
        Experiment_dict is populated as a two-level dictionary. First level
        has the label of the experiment as key and second-level the K.
        """
        self.experiments_dict = {}
        for exp_spec in experiments:
            exp_list = self.load_experiments(exp_spec["start_id"],
                                             exp_spec["end_id"],
                                             exp_spec["n_seeds"],
                                             exp_spec["label"])
            for exp in exp_list:
                if exp.label not in self.experiments_dict:
                    self.experiments_dict[exp.label] = {}
                self.experiments_dict[exp.label][exp.K] = exp

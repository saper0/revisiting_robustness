from pymongo import MongoClient
from typing import Any, Dict, Iterator, List, Tuple, Union

from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
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


def extend_dict(source: Dict[str, Any], target: Dict[str, Any], 
                include_keys: List[str] = []):
    """Extends element in target dictionary by each elements in source-dict.
    
    Assumes source-dict has as element as list or again a dict. Only keys in
    include_keys are considered. If a sub-dictionary is included, all its keys
    will be included.
    """
    for key, item in source.items():
        if key in include_keys:
            if isinstance(item, dict):
                if key not in target:
                    target[key] = {}
                extend_dict(item, target[key], include_keys = item.keys())
            else: 
                if key not in target:
                    target[key] = []
                target[key].extend(item)


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
        self.attack = self.hyperparameters["attack_params"]["attack"]
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
            exclude_keys = exclude_keys[:-1]
            extend_dict(result["robustness_statistics"],
                        self.robustness_statistics, include_keys=exclude_keys)
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
        self.validation_loss = final_validation_loss_l
        self.std_validation_loss = np.std(final_validation_loss_l)
        self.avg_validation_accuracy = np.mean(final_validation_accuracy_l)
        self.std_validation_accuracy = np.std(final_validation_accuracy_l)
        self.validation_accuracy = final_validation_accuracy_l
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
        rob_f_wrt_y_l = []
        rob_g_wrt_y_l = []
        rob_f_wrt_g_l = []
        over_robustness_l = []
        over_robustness_v2_l = []
        min_changes_to_flip_overrob_l = []
        min_changes_to_flip_overrob_v2_l = []
        min_changes_to_flip_advrob_l = []
        min_changes_to_flip_underrob_l = []
        adv_robustness_l = []
        under_robustness_l = []
        f1_robustness_l = []
        f1_robustness_v2_l = []
        f1_min_changes_l = []
        f1_min_changes_v2_l = []
        for experiment in self.individual_experiments:
            result = experiment["result"]
            robustness_stats = result["robustness_statistics"]
            g_wrt_y = robustness_stats["c_bayes_robust_when_both"]
            f_wrt_y = robustness_stats["c_gnn_robust_when_both"]
            f_wrt_g = robustness_stats["c_gnn_wrt_bayes_robust"]
            over_robustness = 0
            #adv_robustness = 0
            rob_f_wrt_y = 0
            rob_g_wrt_y = 0
            rob_f_wrt_g = 0
            min_changes_to_flip_overrob = 0 
            min_changes_to_flip_overrob_v2 = 0
            min_changes_to_flip_advrob = 0
            min_changes_to_flip_underrob = 0
            c_nodes = 0
            for deg in g_wrt_y:
                if deg == "0":
                    continue
                for g_wrt_y_i, f_wrt_y_i, f_wrt_g_i in \
                    zip(g_wrt_y[deg], f_wrt_y[deg], f_wrt_g[deg]):
                    rob_f_wrt_y += f_wrt_y_i / int(deg)
                    rob_g_wrt_y += g_wrt_y_i / int(deg)
                    rob_f_wrt_g += f_wrt_g_i / int(deg)
                    over_robustness += (f_wrt_y_i - g_wrt_y_i) / int(deg)
                    #adv_robustness += (g_wrt_y_i - f_wrt_g_i) / int(deg)
                    min_changes_to_flip_overrob += (f_wrt_y_i + 1) / (g_wrt_y_i + 1)
                    min_changes_to_flip_overrob_v2 += 1 - (f_wrt_g_i + 1) / (f_wrt_y_i + 1)
                    min_changes_to_flip_advrob += (f_wrt_g_i + 1) / (g_wrt_y_i + 1)
                    min_changes_to_flip_underrob += 1 - min_changes_to_flip_advrob
                c_nodes += len(g_wrt_y[deg])
            over_robustness_l.append(over_robustness / c_nodes)
            over_robustness_v2_l.append(1 - rob_f_wrt_g / rob_f_wrt_y)
            adv_robustness_l.append(rob_f_wrt_g / rob_g_wrt_y)
            under_robustness_l.append(1 - rob_f_wrt_g / rob_g_wrt_y)
            rob_f_wrt_y_l.append(rob_f_wrt_y / c_nodes)
            rob_g_wrt_y_l.append(rob_g_wrt_y / c_nodes)
            rob_f_wrt_g_l.append(rob_f_wrt_g / c_nodes)
            min_changes_to_flip_overrob_l.append(min_changes_to_flip_overrob / c_nodes) 
            min_changes_to_flip_overrob_v2_l.append(min_changes_to_flip_overrob_v2 / c_nodes)
            min_changes_to_flip_advrob_l.append(min_changes_to_flip_advrob / c_nodes) 
            min_changes_to_flip_underrob_l.append(min_changes_to_flip_underrob / c_nodes)
            # f1 robustness
            Rover = min(rob_g_wrt_y_l[-1] / rob_f_wrt_y_l[-1], 1)
            Radv = rob_f_wrt_g_l[-1] / rob_g_wrt_y_l[-1]
            f1_robustness_l.append(2 * Rover * Radv / (Rover + Radv))
            # f1 robustness v2
            Rover = 1 - over_robustness_v2_l[-1]
            Radv = adv_robustness_l[-1]
            f1_robustness_v2_l.append(2 * Rover * Radv / (Rover + Radv))
            # f1 min changes
            Rover = min((rob_g_wrt_y_l[-1] + 1)/(rob_f_wrt_y_l[-1] + 1), 1)
            Radv = min_changes_to_flip_advrob_l[-1]
            f1_min_changes_l.append(2 * Rover * Radv / (Rover + Radv))
            # f1 min changes v2
            Rover = min_changes_to_flip_overrob_v2_l[-1]
            Radv = 1 - min_changes_to_flip_underrob_l[-1]
            f1_min_changes_v2_l.append(2 * Rover * Radv / (Rover + Radv))
        if self.label == "APPNP" and False:
            print(over_robustness_l)
        # All Robustness Statistics Attributes
        self.avg_robustness_f_wrt_y = np.mean(rob_f_wrt_y_l)
        self.std_robustness_f_wrt_y = np.std(rob_f_wrt_y_l)
        self.avg_robustness_g_wrt_y = np.mean(rob_g_wrt_y_l)
        self.std_robustness_g_wrt_y = np.std(rob_g_wrt_y_l)
        self.avg_robustness_f_wrt_g = np.mean(rob_f_wrt_g_l)
        self.std_robustness_f_wrt_g = np.std(rob_f_wrt_g_l)
        self.avg_over_robustness = np.mean(over_robustness_l)
        self.std_over_robustness = np.std(over_robustness_l)
        self.avg_over_robustness_v2 = np.mean(over_robustness_v2_l)
        self.std_over_robustness_v2 = np.std(over_robustness_v2_l)
        self.relative_over_robustness = self.avg_robustness_f_wrt_y / self.avg_robustness_g_wrt_y
        # see https://www.geol.lsu.edu/jlorenzo/geophysics/uncertainties/Uncertaintiespart2.html#muldiv
        self.std_relative_over_robustness = \
            (self.std_robustness_f_wrt_y / self.avg_robustness_f_wrt_y 
             + self.std_robustness_g_wrt_y / self.avg_robustness_g_wrt_y) \
                * self.relative_over_robustness 
        self.avg_adv_robustness = np.mean(adv_robustness_l)
        self.std_adv_robustness = np.std(adv_robustness_l)
        self.avg_under_robustness = np.mean(under_robustness_l)
        self.std_under_robustness = np.std(under_robustness_l)
        self.relative_adv_robustness = self.avg_robustness_f_wrt_g / self.avg_robustness_g_wrt_y
        self.std_relative_adv_robustness = \
            (self.std_robustness_f_wrt_g / self.avg_robustness_f_wrt_g
             + self.std_robustness_g_wrt_y / self.avg_robustness_g_wrt_y) \
                * self.relative_adv_robustness
        self.avg_min_changes_to_flip_overrob = np.mean(min_changes_to_flip_overrob_l)
        self.std_min_changes_to_flip_overrob = np.std(min_changes_to_flip_overrob_l)
        self.avg_min_changes_to_flip_overrob_v2 = np.mean(min_changes_to_flip_overrob_v2_l)
        self.std_min_changes_to_flip_overrob_v2 = np.std(min_changes_to_flip_overrob_v2_l)
        self.avg_min_changes_to_flip_advrob = np.mean(min_changes_to_flip_advrob_l)
        self.std_min_changes_to_flip_advrob = np.std(min_changes_to_flip_advrob_l)
        self.avg_min_changes_to_flip_underrob = np.mean(min_changes_to_flip_underrob_l)
        self.std_min_changes_to_flip_underrob = np.std(min_changes_to_flip_underrob_l)
        self.avg_f1_robustness = np.mean(f1_robustness_l)
        self.std_f1_robustness = np.std(f1_robustness_l)
        self.avg_f1_min_changes = np.mean(f1_min_changes_l)
        self.std_f1_min_changes = np.std(f1_min_changes_l)
        self.avg_f1_robustness_v2 = np.mean(f1_robustness_v2_l)
        self.std_f1_robustness_v2 = np.std(f1_robustness_v2_l)
        self.avg_f1_min_changes_v2 = np.mean(f1_min_changes_v2_l)
        self.std_f1_min_changes_v2 = np.std(f1_min_changes_v2_l)
        # Raw Robustness Metrics for each Seed:
        self.rob_f_wrt_y_l = np.array(rob_f_wrt_y_l)
        self.rob_g_wrt_y_l = np.array(rob_g_wrt_y_l)
        self.rob_f_wrt_g_l = np.array(rob_f_wrt_g_l)
        self.min_changes_to_flip_overrob_v2_l = np.array(min_changes_to_flip_overrob_v2_l)
        self.min_changes_to_flip_advrob_l = np.array(min_changes_to_flip_advrob_l)
      
    def get_measurement(self, name: str) -> Tuple[float, float]:
        """Return tuple: averaged measurement, std-measurement."""
        if name == "over-robustness":
            return self.avg_over_robustness.item(), self.std_over_robustness.item()
        if name == "over-robustness-v2":
            return self.avg_over_robustness_v2.item(), self.std_over_robustness_v2.item()
        if name == "relative-over-robustness":
            return self.relative_over_robustness.item(), self.std_relative_over_robustness.item()
        if name == "f_wrt_y":
            return self.avg_robustness_f_wrt_y.item(), self.std_robustness_f_wrt_y.item()
        if name == "g_wrt_y":
            return self.avg_robustness_g_wrt_y.item(), self.std_robustness_g_wrt_y.item()
        if name == "f_wrt_g":
            return self.avg_robustness_f_wrt_g.item(), self.std_robustness_f_wrt_g.item()
        if name == "adversarial-robustness":
            return self.avg_adv_robustness.item(), self.std_adv_robustness.item()
        if name == "relative-adversarial-robustness":
            return self.relative_adv_robustness.item(), self.std_adv_robustness.item()
        if name == "under-robustness":
            return self.avg_under_robustness.item(), self.std_under_robustness.item()
        if name == "validation-accuracy":
            return self.avg_validation_accuracy.item(), self.std_validation_accuracy.item()
        if name == "test-accuracy":
            n = self.hyperparameters["data_params"]["inductive_samples"]
            gnn_test_acc = self.prediction_statistics["avg_c_acc_gnn"] / n
            gnn_test_acc_std = self.prediction_statistics["std_c_acc_gnn"] / n
            return gnn_test_acc, gnn_test_acc_std
        if name == "test-accuracy-bayes":
            n = self.hyperparameters["data_params"]["inductive_samples"]
            bayes_test_acc = self.prediction_statistics["avg_c_acc_bayes"] / n
            bayes_test_acc_std = self.prediction_statistics["std_c_acc_bayes"] / n
            return bayes_test_acc, bayes_test_acc_std
        if name == "relative-changes-to-flip-overrobust":
            return self.avg_min_changes_to_flip_overrob.item(), self.std_min_changes_to_flip_overrob.item()
        if name == "min-changes-to-flip-overrobust-v2":
            return self.avg_min_changes_to_flip_overrob_v2.item(), self.std_min_changes_to_flip_overrob_v2.item()
        if name == "relative-changes-to-flip-advrobust":
            return self.avg_min_changes_to_flip_advrob.item(), self.std_min_changes_to_flip_advrob.item()
        if name == "min-changes-to-flip-underrobust":
            return self.avg_min_changes_to_flip_underrob.item(), self.std_min_changes_to_flip_underrob.item()
        if name == "f1-robustness":
            return self.avg_f1_robustness.item(), self.std_f1_robustness.item()
        if name == "f1-robustness-v2":
            return self.avg_f1_robustness_v2.item(), self.std_f1_robustness_v2.item()
        if name == "f1-min-changes":
            return self.avg_f1_min_changes.item(), self.std_f1_min_changes.item()
        if name == "f1-min-changes-2":
            return self.avg_f1_min_changes_v2.item(), self.std_f1_min_changes_v2.item()


class ExperimentManager:
    """Administrates access and visualization of robustness experiments.
    
    Assumes same experiments with different seeds are stored consecutively.
    """
    def __init__(self, experiments: List[Dict[str, Any]], uri=URI):
        """Establish connection to a given mongodb collection. 
        
        Load and administrate data from specified experiments.
        """
        self.client = MongoClient(uri)
        self.db = self.client.gosl
        self.load(experiments)

    def load_experiment_dict(self, id: int, collection: str) -> Dict[str, Any]:
        """Return result-dict of experiment with ID id."""
        return self.db[collection].find_one({'_id': id})

    def load_experiment(self, start_id: int, n_seeds: int, 
                        collection: str) -> Experiment:
        """Return experiment with ID id."""
        exp_dict_l = []
        for i in range(n_seeds):
            exp_dict_l.append(self.load_experiment_dict(start_id + i, 
                                                        collection))
        return Experiment(exp_dict_l)

    def load_experiments(
            self, start_id: int, end_id: int, n_seeds: int, label: str=None,
            collection: str="runs",
        ) -> List[Experiment]:
        """Return Experiments between start_id and end_id with given label.
        
        Assumes that one experiment consists of multiple seeds which are stored
        consecutively in the mongodb in a given collection.
        """
        experiment_ids = [i for i in range(start_id, end_id + 1, n_seeds)]
        experiments = [self.load_experiment(i, n_seeds, collection) 
                       for i in experiment_ids]
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
            if "collection" not in exp_spec:
                exp_spec["collection"] = "runs"
            exp_list = self.load_experiments(exp_spec["start_id"],
                                             exp_spec["end_id"],
                                             exp_spec["n_seeds"],
                                             exp_spec["label"],
                                             exp_spec["collection"])
            for exp in exp_list:
                if exp.attack not in self.experiments_dict:
                    self.experiments_dict[exp.attack] = {}
                if exp.label not in self.experiments_dict[exp.attack]:
                    self.experiments_dict[exp.attack][exp.label] = {}
                self.experiments_dict[exp.attack][exp.label][exp.K] = exp

    def get_robustness_table(self, attack: str, models: List[str], 
                             K_l: List[float]=[0.1, 0.5, 1, 1.5, 2, 5]):
        """Return data-frame with individual robustnesses as well as over-
        robustness and normal robustness metric."""
        exp_l = []
        for label, K, exp in self.experiment_iterator(attack, models, K_l):
            key = (label, str(K))
            row = [
                f"{exp.avg_robustness_f_wrt_y:.2f}+-{exp.std_robustness_f_wrt_y:.2f}",
                f"{exp.avg_robustness_g_wrt_y:.2f}+-{exp.std_robustness_g_wrt_y:.2f}",
                f"{exp.avg_over_robustness_v2:.2f}+-{exp.std_over_robustness_v2:.2f}",
                #f"{exp.avg_robustness_f_wrt_y / exp.avg_robustness_g_wrt_y * 100:.2f}",
                f"{exp.relative_over_robustness:.2f}+-{exp.std_relative_over_robustness:.2f}",
                f"{exp.avg_min_changes_to_flip_overrob_v2:.2f}+-{exp.std_min_changes_to_flip_overrob_v2:.2f}",
                f"{exp.avg_adv_robustness:.2f}+-{exp.std_adv_robustness:.2f}",
                #f"{exp.avg_robustness_f_wrt_g / exp.avg_robustness_g_wrt_y * 100:.2f}"
                f"{exp.relative_adv_robustness:.2f}+-{exp.std_relative_adv_robustness:.2f}",
                f"{exp.avg_under_robustness:.2f}+-{exp.std_under_robustness:.2f}"
            ]
            exp_l.append((key, row))
        columns = ["Rob_f|y", "Rob_g|y", "1-f|g/f|y", "f|y/g|y", "1-f|y+1/f|y+1", "Rob_f|g", "f|g/g|y", "1-f|g/g|y"]
        f = pd.DataFrame.from_dict(dict(exp_l)).T
        f.columns = columns
        return f

    def plot(self, name: str, attack: str, models: List[str], 
             errorbars: bool=True, ylabel: str=None, title: str=None,
             K_l: List[float]=[0.1, 0.5, 1, 1.5, 2, 5]):
        """Plot relative or absolute over-robustness measure.

        Args:
            name (str): What measurement to plot:
                - over-robustness
                - relative-over-robustness
                - f_wrt_y (allows for BC model)
                - adversarial-robustness
                - relative-adversarial-robustness
                - validation-accuracy
                - test-accuracy
                - f1-robustness
                - f1-min-changes
            attack (str): 
            models (List[str]): White-list
            errorbars (bool): True or False
            ylabel: Label of y-axis. If not provided it is set to "name".
            title: Title of plot. If not provided it is set to "name".
            K_l:List[float]=[0.1, 0.5, 1, 1.5, 2, 5]. White-list
        """
        fig, ax = plt.subplots()
        color_list = ['r', 'tab:green', 'b', 'lime', 'c', 'k', "antiquewhite"]
        linestyle_list = ['-', '--', ':', '-.']
        ax.set_prop_cycle(cycler('linestyle', linestyle_list)*
                          cycler('color', color_list))
        added_bayes = False
        for label, exp_by_k in self.model_iterator(attack, models):
            x = []
            y = []
            y_err = []
            y_bc = []
            y_err_bc = []
            for K, exp in exp_by_k.items():
                if K not in K_l:
                    continue
                x.append(K)
                value, std = exp.get_measurement(name)
                y.append(value)
                y_err.append(std)
                if "BC" in models and not added_bayes:
                    if name == "f_wrt_y":
                        value, std = exp.get_measurement("g_wrt_y")
                    elif name == "test-accuracy":
                        value, std = exp.get_measurement("test-accuracy-bayes")
                    else:
                        raise ValueError("BC requested but name not f_wrt_y")
                    y_bc.append(value)
                    y_err_bc.append(std)
            if errorbars:
                ax.errorbar(x, y, yerr=y_err, marker="o", label=label, capsize=3)
                if "BC" in models and not added_bayes:
                    ax.errorbar(x, y_bc, yerr=y_err_bc, fmt="s:", label="Bayes Classifier", capsize=3)
            else:
                ax.plot(x, y, marker="o", label=label)
                if "BC" in models and not added_bayes:
                    ax.plot(x, y_bc, "s:", label="Bayes Classifier")
            added_bayes = True
        if ylabel is None:
            ylabel=name
        ax.set_ylabel(ylabel)
        if title is None:
            title=name
        ax.set_title(title)
        ax.set_xticks(K_l, minor=False)
        ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
        ax.set_xlabel("K")
        ax.set_xlim(left=0.)
        ax.yaxis.grid()
        ax.legend()
        plt.show()

    def plot_wrt_degree(self, name: str, attack: str, models: List[str], 
                        K: List[float], max_degree: int=None,
                        errorbars: bool=True, ylabel: str=None, title: str=None):
        """Generate w.r.t. degree plots.

        Args:
            name (str):
                - f_wrt_y
                - f_wrt_g
                - both
            attack (str): _description_
            models (List[str]): _description_
            K (List[float]): _description_
            errorbars (bool, optional): _description_. Defaults to True.
            ylabel (str, optional): _description_. Defaults to None.
            title (str, optional): _description_. Defaults to None.
        """
        if max_degree is None:
            max_degree = 0
            for label, K, exp in self.experiment_iterator(attack, models, [K]):  
                avg_f_wrt_y = exp.robustness_statistics["avg_avg_bayes_robust_when_both"]
                max_deg_ = max([int(deg) for deg in avg_f_wrt_y.keys()])
                if max_deg_ > max_degree:
                    max_degree = max_deg_
        
        fig, axs = plt.subplots()
        color_list = ['r', 'tab:green', 'b', 'lime', 'c', 'k', "antiquewhite"]
        linestyle_list = ['-', '--', ':', '-.']
        axs.set_prop_cycle(cycler('linestyle', linestyle_list)*
                           cycler('color', color_list))
        bayes_added = False
        for label, K, exp in self.experiment_iterator(attack, models, [K]):  
            avg_f_wrt_y = exp.robustness_statistics["avg_avg_gnn_robust_when_both"]
            std_f_wrt_y = exp.robustness_statistics["std_avg_gnn_robust_when_both"]
            avg_g_wrt_y = exp.robustness_statistics["avg_avg_bayes_robust_when_both"]
            std_g_wrt_y = exp.robustness_statistics["std_avg_bayes_robust_when_both"]
            avg_f_wrt_g = exp.robustness_statistics["avg_avg_gnn_wrt_bayes_robust"]
            std_f_wrt_g = exp.robustness_statistics["std_avg_gnn_wrt_bayes_robust"]
            
            x = np.sort([int(i) for i in avg_f_wrt_y.keys()])
            x = x[x <= max_degree]
            ordered_avg_f_wrt_y = [avg_f_wrt_y[str(i)] for i in x]
            ordered_std_f_wrt_y = [std_f_wrt_y[str(i)] for i in x]
            ordered_avg_g_wrt_y = [avg_g_wrt_y[str(i)] for i in x]
            ordered_std_g_wrt_y = [std_g_wrt_y[str(i)] for i in x]
            ordered_avg_f_wrt_g = [avg_f_wrt_g[str(i)] for i in x]
            ordered_std_f_wrt_g = [std_f_wrt_g[str(i)] for i in x]
            if errorbars:
                if name == "f_wrt_y" or name == "both":
                    axs.errorbar(x, ordered_avg_f_wrt_y, 
                                yerr=ordered_std_f_wrt_y, marker="o", label=f"{label}", capsize=3)
                if not bayes_added:
                    axs.errorbar(x, ordered_avg_g_wrt_y, 
                                yerr=ordered_std_g_wrt_y, fmt="s:", label=f"Bayes", capsize=3)
                if name == "f_wrt_g" or name == "both":
                    axs.errorbar(x, ordered_avg_f_wrt_g, 
                                yerr=ordered_std_f_wrt_g, marker="o", label=f"{label} w.r.t. Bayes", capsize=3)
            else:
                if name == "f_wrt_y" or name == "both":
                    axs.plot(x, ordered_avg_f_wrt_y, marker='o', label=f"{label}")
                if not bayes_added:
                    axs.plot(x, ordered_avg_g_wrt_y, 's:', label="Bayes")
                if name == "f_wrt_g" or name == "both":
                    axs.plot(x, ordered_avg_f_wrt_g, marker='o', label=f"{label} w.r.t. Bayes")
            bayes_added = True
        if ylabel is None:
            ylabel=name
        axs.set_ylabel(ylabel)
        axs.set_xlabel("Degree")
        if title is None:
            title=name
        axs.set_title(title)
        axs.legend()
        start_x, end_x = axs.get_xlim()
        start_y, end_y = axs.get_ylim()
        axs.xaxis.set_ticks(np.arange(0, end_x, step=1))
        axs.yaxis.set_ticks(np.arange(0, end_y, step=1))
        plt.grid()
        plt.show()

    def boxplot_wrt_degree(self, name: str, attack: str, models: List[str], 
                           K: List[float], max_degree: int = None, 
                           errorbars: bool=True, ylabel: str=None, 
                           title: str=None):
        """Generate w.r.t. degree boxplots.

        Args:
            name (str):
                - f_wrt_y
                - f_wrt_g
                - both
            attack (str): _description_
            models (List[str]): _description_
            K (List[float]): _description_
            errorbars (bool, optional): _description_. Defaults to True.
            ylabel (str, optional): _description_. Defaults to None.
            title (str, optional): _description_. Defaults to None.
        """
        if max_degree is None:
            max_degree = 0
            for label, K, exp in self.experiment_iterator(attack, models, [K]):  
                avg_f_wrt_y = exp.robustness_statistics["avg_avg_bayes_robust_when_both"]
                max_deg_ = max([int(deg) for deg in avg_f_wrt_y.keys()])
                if max_deg_ > max_degree:
                    max_degree = max_deg_
        
        fig, axs = plt.subplots()
        color_list = ['r', 'tab:green', 'b', 'lime', 'c', 'k', "antiquewhite"]
        linestyle_list = ['-', '--', ':', '-.']
        axs.set_prop_cycle(cycler('linestyle', linestyle_list)*
                           cycler('color', color_list))
        bayes_added = True
        for label, K, exp in self.experiment_iterator(attack, models, [K]):  
            f_wrt_y = exp.robustness_statistics["avg_gnn_robust_when_both"]
            g_wrt_y = exp.robustness_statistics["avg_bayes_robust_when_both"]
            f_wrt_g = exp.robustness_statistics["avg_gnn_wrt_bayes_robust"]
            x = np.sort([int(i) for i in f_wrt_y.keys()])
            x = x[x <= max_degree]
            ordered_f_wrt_y = [f_wrt_y[str(i)] for i in x]
            ordered_g_wrt_y = [g_wrt_y[str(i)] for i in x]
            ordered_f_wrt_g = [f_wrt_g[str(i)] for i in x]
            if errorbars:
                if name == "f_wrt_y" or name == "both":
                    axs.boxplot(ordered_f_wrt_y, showfliers=False)
                    #axs.violinplot(ordered_f_wrt_y)
                    pass
                if not bayes_added:
                    axs.boxplot(ordered_g_wrt_y, 
                                whiskerprops={'color' : 'tab:blue'}, 
                                patch_artist=True,
                                showfliers=False)
            else:
                pass
            bayes_added = True
        if ylabel is None:
            ylabel=name
        axs.set_ylabel(ylabel)
        axs.set_xlabel("Degree")
        axs.set_xlim(left=0)
        axs.set_xticks(range(1, len(x)+1), x)
        if title is None:
            title=name
        axs.set_title(title)
        plt.grid()
        plt.show()

    def boxplot_wrt_degree_raw(self, name: str, attack: str, models: List[str], K: List[float],
                        errorbars: bool=True, ylabel: str=None, title: str=None):
        """Generate w.r.t. degree boxplots.

        Args:
            name (str):
                - f_wrt_y
                - f_wrt_g
                - both
            attack (str): _description_
            models (List[str]): _description_
            K (List[float]): _description_
            errorbars (bool, optional): _description_. Defaults to True.
            ylabel (str, optional): _description_. Defaults to None.
            title (str, optional): _description_. Defaults to None.
        """
        max_deg = 0
        for label, K, exp in self.experiment_iterator(attack, models, [K]):  
            avg_f_wrt_y = exp.robustness_statistics["avg_avg_bayes_robust_when_both"]
            max_deg_ = max([int(deg) for deg in avg_f_wrt_y.keys()])
            if max_deg_ > max_deg:
                max_deg = max_deg_
        
        fig, axs = plt.subplots()
        color_list = ['r', 'tab:green', 'b', 'lime', 'c', 'k', "antiquewhite"]
        linestyle_list = ['-', '--', ':', '-.']
        axs.set_prop_cycle(cycler('linestyle', linestyle_list)*
                           cycler('color', color_list))
        bayes_added = False
        for label, K, exp in self.experiment_iterator(attack, models, [K]):  
            f_wrt_y = exp.robustness_statistics["c_gnn_robust_when_both"]
            g_wrt_y = exp.robustness_statistics["c_bayes_robust_when_both"]
            f_wrt_g = exp.robustness_statistics["c_gnn_wrt_bayes_robust"]
            
            x = np.sort([int(i) for i in avg_f_wrt_y.keys()])
            ordered_f_wrt_y = [f_wrt_y[str(i)] for i in x]
            ordered_g_wrt_y = [g_wrt_y[str(i)] for i in x]
            ordered_f_wrt_g = [f_wrt_g[str(i)] for i in x]
            if errorbars:
                if name == "f_wrt_y" or name == "both":
                    #axs.boxplot(ordered_f_wrt_y, showfliers=False)
                    #axs.violinplot(ordered_f_wrt_y)
                    pass
                if not bayes_added:
                    print(len(ordered_g_wrt_y))
                    axs.boxplot(ordered_g_wrt_y, 
                                whiskerprops={'color' : 'tab:blue'}, 
                                patch_artist=True,
                                showfliers=False)
            else:
                pass
            bayes_added = True
        if ylabel is None:
            ylabel=name
        axs.set_ylabel(ylabel)
        axs.set_xlabel("Degree")
        axs.set_xlim(left=0)
        axs.set_xticks(range(1, len(ordered_f_wrt_y)+1), x)
        if title is None:
            title=name
        axs.set_title(title)
        plt.grid()
        plt.show()

    def plot_f1(self, name: str, attack_overrobustness: str, attack_advrobustness: str,
                models: List[str], errorbars: bool=True, 
                label: str=None, title: str=None, ylabel: str=None, 
                K_l: List[float]=[0.1, 0.5, 1, 1.5, 2, 5]):
        """Plot f1 scores to trade of between over- and adv. robustness.

        Args:
            name (str): 
                - f1-robustness
                - f1-min-changes
            attack_overrobustness (str): use lp-weak
            attack_advrobustness (str): use strong attack!
            models (List[str]): _description_
            errorbars (bool, optional): _description_. Defaults to True.
            label (str, optional): _description_. Defaults to None.
            title (str, optional): _description_. Defaults to None.
            K_l (List[float], optional): _description_. Defaults to [0.1, 0.5, 1, 1.5, 2, 5].
        """
        fig, ax = plt.subplots()
        color_list = ['r', 'tab:green', 'b', 'lime', 'c', 'k', "antiquewhite"]
        linestyle_list = ['-', '--', ':', '-.']
        ax.set_prop_cycle(cycler('linestyle', linestyle_list)*
                          cycler('color', color_list))
        # Calculate Over-Robustness Metric
        Rover_dict = {}
        for label, K, exp in self.experiment_iterator(attack_overrobustness, 
                                                      models, K_l):
            if label not in Rover_dict:
                Rover_dict[label] = {}
            if name == "f1-robustness":
                Rover_dict[label][K] = np.minimum(exp.rob_g_wrt_y_l / exp.rob_f_wrt_y_l, 1)
            elif name == "f1-min-changes":
                Rover_dict[label][K] = np.minimum((exp.rob_g_wrt_y_l + 1) / (exp.rob_f_wrt_y_l + 1), 1)
            elif name == "f1-robustness-v2":
                Rover_dict[label][K] = exp.rob_f_wrt_g_l / exp.rob_f_wrt_y_l
            elif name == "f1-min-changes-v2":
                Rover_dict[label][K] = 1 - exp.min_changes_to_flip_overrob_v2_l
            else:
                raise ValueError(f"name={name} but only f1-robustness or f1-min-changes supported.")
        # Calculate Adv-Robsutness Metric
        Radv_dict = {}
        for label, K, exp in self.experiment_iterator(attack_advrobustness,
                                                      models, K_l):
            if label not in Radv_dict:
                Radv_dict[label] = {}
            if name == "f1-robustness":
                Radv_dict[label][K] = exp.rob_f_wrt_g_l / exp.rob_g_wrt_y_l
            elif name == "f1-min-changes":
                Radv_dict[label][K] = (exp.rob_f_wrt_g_l + 1) / (exp.rob_g_wrt_y_l + 1)
            elif name == "f1-robustness-v2":
                Radv_dict[label][K] = exp.rob_f_wrt_g_l / exp.rob_g_wrt_y_l
            elif name == "f1-min-changes-v2":
                Radv_dict[label][K] = exp.avg_min_changes_to_flip_advrob
            else:
                assert False
        # Calculate F1-Score
        for label in Rover_dict:
            x = K_l
            y = []
            yerr = []
            for K in K_l:
                f1_score = 2 * Rover_dict[label][K] * Radv_dict[label][K] / (Rover_dict[label][K] + Radv_dict[label][K])
                y.append(np.mean(f1_score))
                yerr.append(np.std(f1_score))
            if errorbars:
                ax.errorbar(x, y, yerr=yerr, marker="o", label=label, capsize=3)
            else:
                ax.plot(x, y, marker="o", label=label)
        if ylabel is None:
            ylabel=name
        ax.set_ylabel(ylabel)
        if title is None:
            title=name
        ax.set_title(title)
        ax.set_xticks(K_l, minor=False)
        ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
        ax.set_xlabel("K")
        ax.set_xlim(left=0.)
        ax.yaxis.grid()
        ax.legend()
        plt.show()

    def model_iterator(
        self, attack: str, models: List[str]
    ) -> Iterator[Tuple[str, Dict[float, Experiment]]]:
        """Provide iterator over stored models of a specific attack.
        
        Returns a tuple with model-label and associated stored (K, Experiment)
        pairs.
        """
        for attack_, exp_by_label in self.experiments_dict.items():
            if attack_ != attack:
                continue
            for label, exp_by_k in exp_by_label.items():
                if label not in models:
                    continue
                yield (label, exp_by_k)

    def experiment_iterator(
        self, attack: str, models: List[str], 
        K_l: List[float]=[0.1, 0.5, 1, 1.5, 2, 5]
    ) -> Iterator[Tuple[str, float, Experiment]]:
        """Provide iterator over the requested experiments."""
        for label, exp_by_k in self.model_iterator(attack, models):
            for K, exp in exp_by_k.items():
                if float(K) not in K_l:
                    continue
                yield (label, K, exp)


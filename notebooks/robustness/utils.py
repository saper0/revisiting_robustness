from collections.abc import Iterable
import copy
from pymongo import MongoClient
from typing import Any, Dict, Iterator, List, Tuple, Union

from cycler import cycler
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import scipy.stats


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
            avg_dict, std_dict, sem_dict = average_subdict(item)
            target["sem_" + key] = sem_dict
            target["avg_" + key] = avg_dict
            target["std_" + key] = std_dict
        else:
            assert len(target[key]) > 0
            target["sem_" + key] = scipy.stats.sem(item, ddof=0)
            target["avg_" + key] = np.mean(item)
            target["std_" + key] = np.std(item)


def average_subdict(subdict: Dict[str, Any]):
    """Return a dictionary with averaged and a dictionary with standard deviation
    values and standard error of the mean for each element."""
    keys = [key for key in subdict]
    avg_dict = {}
    std_dict = {}
    sem_dict = {}
    for key in keys:
        item = subdict[key]
        assert not isinstance(item, dict)
        assert len(subdict[key]) > 0
        avg_dict[key] = np.mean(item)
        std_dict[key] = np.std(item)
        sem_dict[key] = scipy.stats.sem(item, ddof=0)
    return avg_dict, std_dict, sem_dict


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
        self.std_training_loss = scipy.stats.sem(final_training_loss_l)
        self.avg_training_accuracy = np.mean(final_training_accuracy_l)
        self.std_training_accuracy = scipy.stats.sem(final_training_accuracy_l)
        self.avg_validation_loss = np.mean(final_validation_loss_l)
        self.validation_loss = final_validation_loss_l
        self.std_validation_loss = scipy.stats.sem(final_validation_loss_l)
        self.avg_validation_accuracy = np.mean(final_validation_accuracy_l)
        self.std_validation_accuracy = scipy.stats.sem(final_validation_accuracy_l)
        self.validation_accuracy = final_validation_accuracy_l
          
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
                c_nodes += len(f_wrt_g[deg])
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
        # All Robustness Statistics Attributes
        self.avg_robustness_f_wrt_y = np.mean(rob_f_wrt_y_l)
        self.std_robustness_f_wrt_y = scipy.stats.sem(rob_f_wrt_y_l)
        self.avg_robustness_g_wrt_y = np.mean(rob_g_wrt_y_l)
        self.std_robustness_g_wrt_y = scipy.stats.sem(rob_g_wrt_y_l)
        self.avg_robustness_f_wrt_g = np.mean(rob_f_wrt_g_l)
        self.std_robustness_f_wrt_g = scipy.stats.sem(rob_f_wrt_g_l)
        self.avg_over_robustness = np.mean(over_robustness_l)
        self.std_over_robustness = scipy.stats.sem(over_robustness_l)
        self.avg_over_robustness_v2 = np.mean(over_robustness_v2_l)
        self.std_over_robustness_v2 = scipy.stats.sem(over_robustness_v2_l)
        self.relative_over_robustness = self.avg_robustness_f_wrt_y / self.avg_robustness_g_wrt_y
        # see https://www.geol.lsu.edu/jlorenzo/geophysics/uncertainties/Uncertaintiespart2.html#muldiv
        self.std_relative_over_robustness = \
            (self.std_robustness_f_wrt_y / self.avg_robustness_f_wrt_y 
             + self.std_robustness_g_wrt_y / self.avg_robustness_g_wrt_y) \
                * self.relative_over_robustness 
        self.avg_adv_robustness = np.mean(adv_robustness_l)
        self.std_adv_robustness = scipy.stats.sem(adv_robustness_l)
        self.avg_under_robustness = np.mean(under_robustness_l)
        self.std_under_robustness = scipy.stats.sem(under_robustness_l)
        self.relative_adv_robustness = self.avg_robustness_f_wrt_g / self.avg_robustness_g_wrt_y
        self.std_relative_adv_robustness = \
            (self.std_robustness_f_wrt_g / self.avg_robustness_f_wrt_g
             + self.std_robustness_g_wrt_y / self.avg_robustness_g_wrt_y) \
                * self.relative_adv_robustness
        self.avg_min_changes_to_flip_overrob = np.mean(min_changes_to_flip_overrob_l)
        self.std_min_changes_to_flip_overrob = scipy.stats.sem(min_changes_to_flip_overrob_l)
        self.avg_min_changes_to_flip_overrob_v2 = np.mean(min_changes_to_flip_overrob_v2_l)
        self.std_min_changes_to_flip_overrob_v2 = scipy.stats.sem(min_changes_to_flip_overrob_v2_l)
        self.avg_min_changes_to_flip_advrob = np.mean(min_changes_to_flip_advrob_l)
        self.std_min_changes_to_flip_advrob = scipy.stats.sem(min_changes_to_flip_advrob_l)
        self.avg_min_changes_to_flip_underrob = np.mean(min_changes_to_flip_underrob_l)
        self.std_min_changes_to_flip_underrob = scipy.stats.sem(min_changes_to_flip_underrob_l)
        self.avg_f1_robustness = np.mean(f1_robustness_l)
        self.std_f1_robustness = scipy.stats.sem(f1_robustness_l)
        self.avg_f1_min_changes = np.mean(f1_min_changes_l)
        self.std_f1_min_changes = scipy.stats.sem(f1_min_changes_l)
        self.avg_f1_robustness_v2 = np.mean(f1_robustness_v2_l)
        self.std_f1_robustness_v2 = scipy.stats.sem(f1_robustness_v2_l)
        self.avg_f1_min_changes_v2 = np.mean(f1_min_changes_v2_l)
        self.std_f1_min_changes_v2 = scipy.stats.sem(f1_min_changes_v2_l)
        # Raw Robustness Metrics for each Seed:
        self.rob_f_wrt_y_l = np.array(rob_f_wrt_y_l)
        self.rob_g_wrt_y_l = np.array(rob_g_wrt_y_l)
        self.rob_f_wrt_g_l = np.array(rob_f_wrt_g_l)
        self.min_changes_to_flip_overrob_v2_l = np.array(min_changes_to_flip_overrob_v2_l)
        self.min_changes_to_flip_advrob_l = np.array(min_changes_to_flip_advrob_l)
      
    def get_measurement(self, name: str, budget: str=None) -> Tuple[float, float]:
        """Return tuple: averaged measurement, std-measurement.
        
        Budget can be None, deg+2 or int
        """
        if name == "over-robustness":
            return self.avg_over_robustness.item(), self.std_over_robustness.item()
        if name == "over-robustness-v2":
            if budget is None:
                return self.avg_over_robustness_v2.item(), self.std_over_robustness_v2.item()
            else:
                return self.calc_robustness_measure(name, local_budget=budget)
        if name == "relative-over-robustness":
            return self.relative_over_robustness.item(), self.std_relative_over_robustness.item()
        if name == "f_wrt_y":
            if budget is None:
                return self.avg_robustness_f_wrt_y.item(), self.std_robustness_f_wrt_y.item()
            else:
                return self.calc_robustness_measure(name, local_budget=budget)
        if name == "g_wrt_y":
            if budget is None:
                return self.avg_robustness_g_wrt_y.item(), self.std_robustness_g_wrt_y.item()
            else:
                return self.calc_robustness_measure(name, local_budget=budget)
        if name == "f_wrt_g":
            return self.avg_robustness_f_wrt_g.item(), self.std_robustness_f_wrt_g.item()
        if name == "adversarial-robustness":
            if budget is None:
                return self.avg_adv_robustness.item(), self.std_adv_robustness.item()
            else:
                return self.calc_robustness_measure(name, local_budget=budget)
        if name == "relative-adversarial-robustness":
            return self.relative_adv_robustness.item(), self.std_adv_robustness.item()
        if name == "under-robustness":
            return self.avg_under_robustness.item(), self.std_under_robustness.item()
        if name == "validation-accuracy":
            return self.avg_validation_accuracy.item(), self.std_validation_accuracy.item()
        if name == "test-accuracy":
            n = self.hyperparameters["data_params"]["inductive_samples"]
            gnn_test_acc = self.prediction_statistics["avg_c_acc_gnn"] / n
            gnn_test_acc_std = self.prediction_statistics["sem_c_acc_gnn"] / n
            return gnn_test_acc, gnn_test_acc_std
        if name == "test-accuracy-bayes":
            n = self.hyperparameters["data_params"]["inductive_samples"]
            bayes_test_acc = self.prediction_statistics["avg_c_acc_bayes"] / n
            bayes_test_acc_std = self.prediction_statistics["sem_c_acc_bayes"] / n
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
            if budget is None:
                return self.avg_f1_robustness_v2.item(), self.std_f1_robustness_v2.item()
            else:
                return self.calc_robustness_measure(name, local_budget=budget)
        if name == "f1-min-changes":
            return self.avg_f1_min_changes.item(), self.std_f1_min_changes.item()
        if name == "f1-min-changes-2":
            return self.avg_f1_min_changes_v2.item(), self.std_f1_min_changes_v2.item()
        if name == "rob_f_wrt_y_l":
            if budget is None:
                return self.rob_f_wrt_y_l
            else:
                return self.calc_robustness_measure(name, local_budget=budget)
        if name == "rob_g_wrt_y_l":
            if budget is None:
                return self.rob_g_wrt_y_l
            else:
                return self.calc_robustness_measure(name, local_budget=budget)
        if name == "rob_f_wrt_g_l":
            if budget is None:
                return self.rob_f_wrt_g_l
            else:
                return self.calc_robustness_measure(name, local_budget=budget)
        if name == "adversarial_accuracy":
            if budget is None:
                assert False, "Not implemented without budget."
            else:
                return self.calc_robustness_measure(name, local_budget=budget)
        if name == "adversarial_error":
            if budget is None:
                assert False, "Not implemented without budget."
            else:
                return self.calc_robustness_measure(name, local_budget=budget)
        if name == "c_acc_bayes_feature":
            n = self.hyperparameters["data_params"]["inductive_samples"]
            return self.prediction_statistics["avg_c_acc_bayes_feature"] / n, \
                self.prediction_statistics["std_c_acc_bayes_feature"] / n
        if name == "c_acc_bayes_structure":
            n = self.hyperparameters["data_params"]["inductive_samples"]
            return self.prediction_statistics["avg_c_acc_bayes_structure"] / n, \
                self.prediction_statistics["std_c_acc_bayes_structure"] / n
        if name == "c_acc_bayes":
            n = self.hyperparameters["data_params"]["inductive_samples"]
            return self.prediction_statistics["avg_c_acc_bayes"] / n, \
                self.prediction_statistics["std_c_acc_bayes"] / n

    def calc_robustness_measure(self, name, local_budget: str):
        rob_f_wrt_y_l = []
        rob_g_wrt_y_l = []
        rob_f_wrt_g_l = []
        over_robustness_l = []
        adv_robustness_l = []
        f1_robustness_l = []
        adversarial_accuracy_l = []
        for experiment in self.individual_experiments:
            result = experiment["result"]
            robustness_statistics = result["robustness_statistics"]
            f_wrt_y = robustness_statistics["c_gnn_robust_when_both"]
            g_wrt_y = robustness_statistics["c_bayes_robust_when_both"]
            f_wrt_g = robustness_statistics["c_gnn_wrt_bayes_robust"]
            rob_f_wrt_y = 0
            rob_g_wrt_y = 0
            rob_f_wrt_g = 0
            wrong_robust_examples = 0
            c_nodes = 0
            for deg in f_wrt_y:
                if deg == "0":
                    continue
                #print(self.label, self.K)
                #print(len(f_wrt_g[deg]), len(g_wrt_y[deg]), len(f_wrt_y[deg]))
                #if self.label == "MLP":
                #    if len(f_wrt_g[deg]) < len(g_wrt_y[deg]):
                #        n = len(g_wrt_y[deg]) - len(f_wrt_g[deg])
                #        for i in range(n):
                #            f_wrt_g[deg].append(100)
                #assert len(g_wrt_y[deg]) == len(f_wrt_y[deg])
                #assert len(f_wrt_g[deg]) == len(f_wrt_y[deg])
                for g_wrt_y_i, f_wrt_y_i, f_wrt_g_i in \
                    zip(g_wrt_y[deg], f_wrt_y[deg], f_wrt_g[deg]):
                    if type(local_budget) == str:
                        budget = int(deg)+int(local_budget[4])
                    else:
                        budget = local_budget
                    if f_wrt_y_i > budget:
                        f_wrt_y_i = budget
                    if g_wrt_y_i > budget:
                        g_wrt_y_i = budget
                    if f_wrt_g_i > budget:
                        f_wrt_g_i = budget
                    if f_wrt_y_i > g_wrt_y_i:
                        wrong_robust_examples += 1
                    rob_f_wrt_y += f_wrt_y_i / int(deg)
                    rob_g_wrt_y += g_wrt_y_i / int(deg)
                    rob_f_wrt_g += f_wrt_g_i / int(deg)
                c_nodes += len(f_wrt_g[deg])
            adversarial_accuracy_l.append(wrong_robust_examples / c_nodes)
            over_robustness_l.append(1 - rob_f_wrt_g / rob_f_wrt_y)
            adv_robustness_l.append(rob_f_wrt_g / rob_g_wrt_y)
            rob_f_wrt_y_l.append(rob_f_wrt_y / c_nodes)
            rob_g_wrt_y_l.append(rob_g_wrt_y / c_nodes)
            rob_f_wrt_g_l.append(rob_f_wrt_g / c_nodes)
            # f1 robustness v2
            Rover = 1 - over_robustness_l[-1]
            Radv = adv_robustness_l[-1]
            f1_robustness_l.append(2 * Rover * Radv / (Rover + Radv))
        if name=="over-robustness-v2":
            return np.mean(over_robustness_l), scipy.stats.sem(over_robustness_l)
        if name=="adversarial-robustness":
            return np.mean(adv_robustness_l), scipy.stats.sem(adv_robustness_l)
        if name=="f1-robustness-v2":
            return np.mean(f1_robustness_l), scipy.stats.sem(f1_robustness_l)
        if name=="rob_f_wrt_y_l":
            return np.array(rob_f_wrt_y_l)
        if name=="rob_g_wrt_y_l":
            return np.array(rob_g_wrt_y_l)
        if name=="rob_f_wrt_g_l":
            return np.array(rob_f_wrt_g_l)
        if name=="f_wrt_y":
            return np.mean(rob_f_wrt_y_l), scipy.stats.sem(rob_f_wrt_y_l)
        if name=="g_wrt_y":
            return np.mean(rob_g_wrt_y_l), scipy.stats.sem(rob_g_wrt_y_l)
        if name=="adversarial_accuracy":
            return 1-np.mean(adversarial_accuracy_l), scipy.stats.sem(adversarial_accuracy_l)
        if name=="adversarial_error":
            return np.mean(adversarial_accuracy_l), scipy.stats.sem(adversarial_accuracy_l)


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
            if not isinstance(exp_spec["start_id"], Iterable):
                exp_spec["start_id"] = [exp_spec["start_id"]]
                exp_spec["end_id"] = [exp_spec["end_id"]]
            for start_id, end_id in zip(exp_spec["start_id"], exp_spec["end_id"]):
                exp_list = self.load_experiments(start_id,
                                                 end_id,
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

    def get_value_in_table(self, name: str, attack: str, models: List[str],
                           budget: str=None, use_mean=True, use_std=False,
                           K_l: List[float]=[0.1, 0.5, 1, 1.5, 2, 3, 4, 5]):
        columns = {}
        added_bayes = {}
        for label, K, exp in self.experiment_iterator(attack, models, K_l):
            key = str(K)
            if key not in columns:
                columns[key] = {}
            mean, std = exp.get_measurement(name, budget)
            if use_mean and not use_std:
                columns[key][label] = f"{mean*100:.1f}%"
            if not use_mean and use_std:
                columns[key][label] = f"{std*100:.1f}%"
            if use_mean and use_std:
                columns[key][label] = f"{mean*100:.1f}+{std*100:.1f}%"
            if "BC" in models and key not in added_bayes:
                mean, std = exp.get_measurement(name+"-bayes", budget)
                columns[key]["Bayes Classifier (BC)"] = f"{mean*100:.1f}%"
                mean, std = exp.get_measurement("c_acc_bayes_feature", budget)
                columns[key]["BC (Features Only)"] = f"{mean*100:.1f}%"
                mean, std = exp.get_measurement("c_acc_bayes_structure", budget)
                columns[key]["BC (Structure Only)"] = f"{mean*100:.1f}%"
                added_bayes[key] = ""
        f = pd.DataFrame.from_dict(columns)
        f = f.reindex(sorted(f.columns), axis=1)
        return f

    def get_bc_performance(self, name: str, attack: str, models: List[str],
                           budget: str=None, 
                           K_l: List[float]=[0.1, 0.5, 1, 1.5, 2, 3, 4, 5]):
        columns = {}
        for label in ["_feature", "_structure", ""]:
            name = "c_acc_bayes" + label
            for _, K, exp in self.experiment_iterator(attack, models, K_l):
                key = str(K)
                if key not in columns:
                    columns[key] = {}
                mean, std = exp.get_measurement(name, budget)
                columns[key][name] = f"{mean*100:.1f}+-{std*100:.1f}"
            f = pd.DataFrame.from_dict(columns)
        return f

    def get_style(self, label: str):
        color_dict = {
            "MLP": 'r',
            "GCN": 'tab:green', 
            "APPNP": 'lime', 
            "SGC": "b",
            "GAT": "slategrey",
            "GATv2": "k",
            "GraphSAGE": "lightsteelblue",
            "LP": "wheat"
        }
        linestyle_dict = {
            "LP": '--'
        }
        use_color=""
        linestyle="-"
        for key, color in color_dict.items():
            sep_labels = label.split("+")
            if sep_labels[0] == key:
                use_color = color
                if len(sep_labels) == 2 or sep_labels[0] == "LP":
                    linestyle = "--"
        return use_color, linestyle

    def plot(self, name: str, attack: str, models: List[str], 
             errorbars: bool=True, ylabel: str=None, title: str=None,
             spacing: str="normal", legend_loc="best", legend_cols: int=None,
             budget: str=None, yspacing: str="normal", width=0.86, ratio=1.618,
             titlefont=20, fontweight="bold",
             K_l: List[float]=[0.1, 0.5, 1, 1.5, 2, 3, 4, 5]):
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
        h, w = matplotlib.figure.figaspect(ratio / width)
        fig, ax = plt.subplots(figsize=(w,h))
        #color_list = ['r', 
        #              'tab:green', 
        #              'b', 
        #              'lime', 
        #              'slategrey', 
        #              'k', 
        #              "lightsteelblue",
        #              "antiquewhite",
        #              ]
        #linestyle_list = ['-', '--', ':', '-.']
        #ax.set_prop_cycle(cycler('linestyle', linestyle_list)*
        #                  cycler('color', color_list))
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
                value, std = exp.get_measurement(name, budget)
                y.append(value)
                y_err.append(std)
                if "BC" in models and not added_bayes:
                    if name == "f_wrt_y":
                        value, std = exp.get_measurement("g_wrt_y", budget)
                    elif name == "test-accuracy":
                        value, std = exp.get_measurement("test-accuracy-bayes")
                    else:
                        raise ValueError("BC requested but name not f_wrt_y")
                    y_bc.append(value)
                    y_err_bc.append(std)
            sort_ids = np.argsort(x)
            if spacing == "even":
                x = [i for i in range(len(K_l))]
            else:
                x = K_l
            y = np.array(y)[sort_ids]
            color, linestyle = self.get_style(label)
            if label == "GraphSAGE":
                label = "GraphSAGE"
            if label == "GraphSAGE+LP":
                label = "GraphSAGE+LP"
            if errorbars:
                y_err = np.array(y_err)[sort_ids]
                ax.errorbar(x, y, yerr=y_err, marker="o", color=color, linestyle=linestyle,
                            label=label, capsize=5, linewidth=2.5, markersize=8)
                if "BC" in models and not added_bayes:
                    y_bc = np.array(y_bc)[sort_ids]
                    y_err_bc = np.array(y_err_bc)[sort_ids] 
                    ax.errorbar(x, y_bc, yerr=y_err_bc, fmt="s:", label="Bayes Classifier", 
                    capsize=5, linewidth=2.5, markersize=8)
            else:
                ax.plot(x, y, marker="o",  color=color, linestyle=linestyle, 
                        label=label)
                if "BC" in models and not added_bayes:
                    y_bc = np.array(y_bc)[sort_ids]
                    ax.plot(x, y_bc, "s:", label="Bayes Classifier")
            added_bayes = True
        if ylabel is None:
            ylabel=name
        
        ax.xaxis.get_major_formatter()._usetex = False
        ax.yaxis.get_major_formatter()._usetex = False
        if title is None:
            title=name
        if yspacing == "log":
            ax.set_yscale('log')
        if spacing == "log":
            ax.set_xscale('log')
            xticks = np.sort(K_l.append([0.2, 10]))
            ax.set_xticks(xticks, minor=True)
        elif spacing == "even":
            ax.xaxis.set_ticks(x, minor=False)
            xticks = [f"{K}" for K in K_l]
            ax.xaxis.set_ticklabels(xticks, fontsize=15, fontweight="bold")
            ax.set_xlim(left=-0.3)
        else:
            ax.set_xticks(K_l, minor=False)
            ax.set_xlim(left=0.)
        ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_xlabel("K", fontsize=17, fontweight="bold")
        ax.set_title(title, fontweight=fontweight, fontsize=titlefont)
        #ax.set_yticklabels([f"{round(i, 2):.2f}" for i in ax.get_yticks()], fontsize=13)
        ax.set_yticklabels([f"{round(i, 1):.1f}" for i in ax.get_yticks()], fontsize=15, fontweight="bold")
        #ax.set_xticklabels(ax.get_xticks(), fontsize=13)
        ax.yaxis.grid()
        ax.xaxis.grid()
        if legend_cols is None:
            ax.legend(loc=legend_loc)
        else:
            box = ax.get_position()
            #w = box.width * 0.8
            #h = w / 1.618
            #width = 1.618 * box.height
            ax.set_position([box.x0, box.y0, box.width*width, box.height])
            leg = ax.legend(loc=legend_loc, ncol=legend_cols, shadow=False,
                            bbox_to_anchor=(1, -0.23), frameon=False, markerscale=1,
                            prop=dict(size=12.3, weight="bold"))
            #leg.get_lines()[0].set_linestyle
            #for i in leg.legendHandles:
            #    i.set_linestyle(":")
            #ax.set_aspect(1.618)
        fig.set_tight_layout(True)
        plt.show()

    def plot_workshop(self, name: str, attack: str, models: List[str], 
             errorbars: bool=True, ylabel: str=None, title: str=None,
             spacing: str="normal", legend_loc="best", legend_cols: int=None,
             budget: str=None, yspacing: str="normal", width=0.86, ratio=1.618,
             titlefont=20, fontweight="bold", tickfont=17, xylabelfont=20,
             legendfont=17, outside_legend=False, linewidth=2.5, markersize=8,
             ylim = None, bbox_to_anchor=(1.005, 1.011), handlelength=1.35,
             labelspacing = 0.1, borderpad=0.2, tickfontweight="bold",
             K_l: List[float]=[0.1, 0.5, 1, 1.5, 2, 3, 4, 5]):
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
        h, w = matplotlib.figure.figaspect(ratio / width)
        fig, ax = plt.subplots(figsize=(w,h))
        #color_list = ['r', 
        #              'tab:green', 
        #              'b', 
        #              'lime', 
        #              'slategrey', 
        #              'k', 
        #              "lightsteelblue",
        #              "antiquewhite",
        #              ]
        #linestyle_list = ['-', '--', ':', '-.']
        #ax.set_prop_cycle(cycler('linestyle', linestyle_list)*
        #                  cycler('color', color_list))
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
                value, std = exp.get_measurement(name, budget)
                y.append(value)
                y_err.append(std)
                if "BC" in models and not added_bayes:
                    if name == "f_wrt_y":
                        value, std = exp.get_measurement("g_wrt_y", budget)
                    elif name == "test-accuracy":
                        value, std = exp.get_measurement("test-accuracy-bayes")
                    else:
                        raise ValueError("BC requested but name not f_wrt_y")
                    y_bc.append(value)
                    y_err_bc.append(std)
            sort_ids = np.argsort(x)
            if spacing == "even":
                x = [i for i in range(len(K_l))]
            else:
                x = K_l
            y = np.array(y)[sort_ids]
            color, linestyle = self.get_style(label)
            if label == "GraphSAGE":
                label = "GraphSAGE"
            if label == "GraphSAGE+LP":
                label = "GraphSAGE+LP"
            if fontweight == "bold":
                label = f"\\textbf{{{label}}}"
            if errorbars:
                y_err = np.array(y_err)[sort_ids]
                ax.errorbar(x, y, yerr=y_err, marker="o", color=color, linestyle=linestyle,
                            label=label, capsize=5, linewidth=linewidth, markersize=markersize)
                if "BC" in models and not added_bayes:
                    y_bc = np.array(y_bc)[sort_ids]
                    y_err_bc = np.array(y_err_bc)[sort_ids] 
                    ax.errorbar(x, y_bc, yerr=y_err_bc, fmt="s:", 
                                label="Bayes Classifier", color="tab:olive",
                                capsize=5, linewidth=3, markersize=9)
            else:
                ax.plot(x, y, marker="o",  color=color, linestyle=linestyle, 
                        label=label)
                if "BC" in models and not added_bayes:
                    y_bc = np.array(y_bc)[sort_ids]
                    ax.plot(x, y_bc, "s:", label="Bayes Classifier")
            added_bayes = True
        if ylabel is None:
            ylabel=name
        
        ax.xaxis.get_major_formatter()._usetex = True
        ax.yaxis.get_major_formatter()._usetex = True
        if title is None:
            title=name
        if yspacing == "log":
            ax.set_yscale('log')
        if spacing == "log":
            ax.set_xscale('log')
            xticks = np.sort(K_l.append([0.2, 10]))
            ax.set_xticks(xticks, minor=True)
        elif spacing == "even":
            ax.xaxis.set_ticks(x, minor=False)
            if tickfontweight=="bold":
                xticks = [f"\\textbf{{{K}}}" for K in K_l]
            else:
                xticks = [f"{K}" for K in K_l]
            ax.xaxis.set_ticklabels(xticks, fontsize=tickfont)
            ax.set_xlim(left=-0.3)
            if ylim is not None:
                ax.set_ylim(top=ylim)
        else:
            ax.set_xticks(K_l, minor=False)
            ax.set_xlim(left=0.)
        ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
        ax.set_ylabel(ylabel, fontsize=xylabelfont)
        xlabel = "K"
        if fontweight == "bold":
            xlabel = f"\\textbf{{{xlabel}}}"
        ax.set_xlabel(xlabel, fontsize=xylabelfont)
        ax.set_title(title, fontweight=fontweight, fontsize=titlefont)
        if tickfontweight=="bold":
            ax.set_yticklabels([f"\\textbf{{{round(i, 2):.1f}}}" for i in ax.get_yticks()], fontsize=tickfont)
        else:
            ax.set_yticklabels([f"{round(i, 2):.1f}" for i in ax.get_yticks()], fontsize=tickfont)
        #ax.set_yticklabels([f"{round(i, 1):.1f}" for i in ax.get_yticks()], fontsize=tickfont)
        #ax.set_xticklabels(ax.get_xticks(), fontsize=13)
        ax.yaxis.grid()
        ax.xaxis.grid()
        if legend_cols is None:
            ax.legend(loc=legend_loc)
        #Titlepicture:
        #ax.legend(prop=dict(size=legendfont, weight="bold"), loc=legend_loc, 
        #          bbox_to_anchor=(1.018, 1.04), frameon=True,
        #          labelspacing=0.12, handleheight=0.5)
        if outside_legend:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width*width, box.height])
            leg = ax.legend(loc=legend_loc, ncol=legend_cols, shadow=False,
                            bbox_to_anchor=(1, 1), frameon=False, markerscale=1,
                            prop=dict(size=legendfont))
        else:
            #ax.legend(prop=dict(size=legendfont), loc=legend_loc, 
            #        bbox_to_anchor=(0.5, 1.028), frameon=True,
            #        labelspacing=0.1, handleheight=0.3, ncol=legend_cols,
            #        handlelength=1, columnspacing=0.2)
            #Titlepicture:
            #ax.legend(prop=dict(size=legendfont), loc=legend_loc, 
            #        bbox_to_anchor=bbox_to_anchor, frameon=True,
            #        labelspacing=0.1, ncol=legend_cols, handletextpad=0.4,
            #        handlelength=1.35, columnspacing=0.2, borderaxespad=0.2,
            #        borderpad=0.2)
            ax.legend(prop=dict(size=legendfont), loc=legend_loc, 
                    bbox_to_anchor=bbox_to_anchor, frameon=True,
                    labelspacing=labelspacing, ncol=legend_cols, handletextpad=0.4,
                    handlelength=handlelength, columnspacing=0.2, borderaxespad=0.2,
                    borderpad=borderpad)
        fig.set_tight_layout(True)
        plt.show()

    def plot_normal(self, name: str, attack: str, models: List[str], 
             errorbars: bool=True, ylabel: str=None, title: str=None,
             spacing: str="normal", legend_loc="best", legend_cols: int=None,
             budget: str=None, yspacing: str="normal", width=0.86, ratio=1.618,
             K_l: List[float]=[0.1, 0.5, 1, 1.5, 2, 3, 4, 5]):
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
        h, w = matplotlib.figure.figaspect(ratio / width)
        fig, ax = plt.subplots(figsize=(w,h))
        #color_list = ['r', 
        #              'tab:green', 
        #              'b', 
        #              'lime', 
        #              'slategrey', 
        #              'k', 
        #              "lightsteelblue",
        #              "antiquewhite",
        #              ]
        #linestyle_list = ['-', '--', ':', '-.']
        #ax.set_prop_cycle(cycler('linestyle', linestyle_list)*
        #                  cycler('color', color_list))
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
                value, std = exp.get_measurement(name, budget)
                y.append(value)
                y_err.append(std)
                if "BC" in models and not added_bayes:
                    if name == "f_wrt_y":
                        value, std = exp.get_measurement("g_wrt_y", budget)
                    elif name == "test-accuracy":
                        value, std = exp.get_measurement("test-accuracy-bayes")
                    else:
                        raise ValueError("BC requested but name not f_wrt_y")
                    y_bc.append(value)
                    y_err_bc.append(std)
            sort_ids = np.argsort(x)
            if spacing == "even":
                x = [i for i in range(len(K_l))]
            else:
                x = K_l
            y = np.array(y)[sort_ids]
            color, linestyle = self.get_style(label)
            if label == "GraphSAGE":
                label = "GraphSAGE"
            if label == "GraphSAGE+LP":
                label = "GraphSAGE+LP"
            if errorbars:
                y_err = np.array(y_err)[sort_ids]
                ax.errorbar(x, y, yerr=y_err, marker="o", color=color, linestyle=linestyle,
                            label=label, capsize=3)
                if "BC" in models and not added_bayes:
                    y_bc = np.array(y_bc)[sort_ids]
                    y_err_bc = np.array(y_err_bc)[sort_ids] 
                    ax.errorbar(x, y_bc, yerr=y_err_bc, fmt="s:", label="Bayes Classifier", 
                    capsize=3, color="tab:olive")
            else:
                ax.plot(x, y, marker="o",  color=color, linestyle=linestyle, 
                        label=label)
                if "BC" in models and not added_bayes:
                    y_bc = np.array(y_bc)[sort_ids]
                    ax.plot(x, y_bc, "s:", label="Bayes Classifier")
            added_bayes = True
        if ylabel is None:
            ylabel=name
        
        ax.xaxis.get_major_formatter()._usetex = False
        ax.yaxis.get_major_formatter()._usetex = False
        if title is None:
            title=name
        if yspacing == "log":
            ax.set_yscale('log')
        if spacing == "log":
            ax.set_xscale('log')
            xticks = np.sort(K_l.append([0.2, 10]))
            ax.set_xticks(xticks, minor=True)
        elif spacing == "even":
            ax.xaxis.set_ticks(x, minor=False)
            xticks = [f"{K}" for K in K_l]
            ax.xaxis.set_ticklabels(xticks)
            ax.set_xlim(left=-0.3)
        else:
            ax.set_xticks(K_l, minor=False)
            ax.set_xlim(left=0.)
        ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
        ax.set_ylabel(ylabel)
        ax.set_xlabel("K")
        ax.set_title(title, fontweight="bold", fontsize=15)
        #ax.set_yticklabels([f"{round(i, 2):.2f}" for i in ax.get_yticks()], fontsize=13)
        ax.set_yticklabels([f"{round(i, 1):.1f}" for i in ax.get_yticks()],)
        #ax.set_xticklabels(ax.get_xticks(), fontsize=13)
        ax.yaxis.grid()
        ax.xaxis.grid()
        if legend_cols is None:
            ax.legend(loc=legend_loc)
        else:
            box = ax.get_position()
            #w = box.width * 0.8
            #h = w / 1.618
            #width = 1.618 * box.height
            ax.set_position([box.x0, box.y0, box.width*width, box.height])
            ax.legend(loc=legend_loc, ncol=legend_cols, shadow=False,
                      bbox_to_anchor=(1, -0.05), frameon=False,
                      prop=dict(size=10, weight="bold"))
            #ax.set_aspect(1.618)
        fig.set_tight_layout(True)
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
        h, w = matplotlib.figure.figaspect(1.618 / 1)
        fig, axs = plt.subplots(figsize=(w,h))
        color_list = ['r', 'tab:green', 'b', 'lime', 'c', 'k', "wheat", "tab:olive"]
        linestyle_list = ['-', '--', ':', '-.']
        axs.set_prop_cycle(cycler('linestyle', linestyle_list)*
                           cycler('color', color_list))
        bayes_added = False
        for label, K, exp in self.experiment_iterator(attack, models, K):  
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
                    if K == 5.0:
                        avg_g_store = ordered_avg_g_wrt_y
                        std_g_store = ordered_std_g_wrt_y
                    else:
                        axs.errorbar(x, ordered_avg_g_wrt_y, 
                                    yerr=ordered_std_g_wrt_y, fmt="s:", label=f"K={K}", capsize=3)
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
            bayes_added = False
        axs.errorbar(x, avg_g_store, 
                    yerr=std_g_store, fmt="s:", label=f"K={5.0}", capsize=3)
        if ylabel is None:
            ylabel=name
        axs.set_ylabel(ylabel, fontsize=12)
        axs.set_xlabel("Node degree", fontsize=12)
        if title is None:
            title=name
        axs.set_title(r"\textbf{Average robustness of the Bayes classifier}", fontsize=13)
        axs.legend(loc="upper left", ncol=2, bbox_to_anchor=(-0.01, 1.02))
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
                        errorbars: bool=True, ylabel: str=None, title: str=None,
                        toplim=12):
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
        fig, axs = plt.subplots()
        max_deg = 0
        for label, K, exp in self.experiment_iterator(attack, models, [K]):  
            avg_f_wrt_y = exp.robustness_statistics["avg_avg_bayes_robust_when_both"]
            max_deg_ = max([int(deg) for deg in avg_f_wrt_y.keys()])
            if max_deg_ > max_deg:
                max_deg = max_deg_
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
            x = x[x <= 8]
            ordered_f_wrt_y = [f_wrt_y[str(i)] for i in x]
            for f_wrt_y_l in ordered_f_wrt_y:
                for i in range(len(f_wrt_y_l)):
                    if f_wrt_y_l[i] > toplim: 
                        f_wrt_y_l[i]=toplim
            ordered_g_wrt_y = [g_wrt_y[str(i)] for i in x]
            ordered_f_wrt_g = [f_wrt_g[str(i)] for i in x]
            if errorbars:
                if name == "f_wrt_y" or name == "both":
                    axs.boxplot(ordered_f_wrt_y, positions=x,showfliers=True,
                                notch=False, 
                                flierprops={'markersize':5}, 
                                medianprops={'linewidth':3, "solid_capstyle": "butt"})
                    #axs.violinplot(ordered_f_wrt_y)
                    pass
                if name == "g_wrt_y":
                    #print(len(ordered_g_wrt_y))
                    axs.boxplot(ordered_g_wrt_y, positions=x,showfliers=True,
                                notch=False, 
                                flierprops={'markersize':5}, 
                                medianprops={'linewidth':3, "solid_capstyle": "butt"})
                    pass
            else:
                pass
            bayes_added = True
        if ylabel is None:
            ylabel=name
        axs.set_ylabel(ylabel)
        axs.set_xlabel("Node degree", fontsize=15)
        axs.set_xlim(left=-0.5)
        axs.set_ylim(top=toplim)
        print(x)
        axs.set_xticks(range(0, len(ordered_f_wrt_y)), x)
        axs.set_yticks(range(0, toplim+1, 5))
        axs.set_xticklabels(axs.get_xticks(), fontsize=13)
        axs.set_xticklabels(axs.get_xticks(), fontsize=13)
        axs.yaxis.grid()
        if title is None:
            title=name
        axs.set_title(title, fontweight="bold", fontsize=15)
        plt.show()

    def plot_f1(self, name: str, attack_overrobustness: str, attack_advrobustness: str,
                models: List[str], errorbars: bool=True, spacing: str="normal",
                label: str=None, title: str=None, ylabel: str=None, budget=None,
                legend_loc="best", titlefont=20, tickfont=17, xylabelfont=20,
                legendfont=17, outside_legend=False, width=0.86,
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
        h, w = matplotlib.figure.figaspect(1.618 / width)
        fig, ax = plt.subplots(figsize=(w,h))
        color_list = ['r', 'tab:green', 'b', 'lime', 'c', 'k', "antiquewhite"]
        linestyle_list = ['-', '--', ':', '-.']
        ax.set_prop_cycle(cycler('linestyle', linestyle_list)*
                          cycler('color', color_list))
        # Calculate Over-Robustness Metric
        Rover_dict = {}
        for label, K, exp in self.experiment_iterator(attack_overrobustness, 
                                                      models, K_l):
            rob_g_wrt_y_l = exp.get_measurement("rob_g_wrt_y_l", budget)
            rob_f_wrt_y_l = exp.get_measurement("rob_f_wrt_y_l", budget)
            rob_f_wrt_g_l = exp.get_measurement("rob_f_wrt_g_l", budget)
            if label not in Rover_dict:
                Rover_dict[label] = {}
            if name == "f1-robustness":
                Rover_dict[label][K] = np.minimum(rob_g_wrt_y_l / rob_f_wrt_y_l, 1)
            elif name == "f1-min-changes":
                Rover_dict[label][K] = np.minimum((rob_g_wrt_y_l + 1) / (rob_f_wrt_y_l + 1), 1)
            elif name == "f1-robustness-v2":
                Rover_dict[label][K] = rob_f_wrt_g_l / rob_f_wrt_y_l
            elif name == "f1-min-changes-v2":
                Rover_dict[label][K] = 1 - exp.min_changes_to_flip_overrob_v2_l
            else:
                raise ValueError(f"name={name} but only f1-robustness or f1-min-changes supported.")
        # Calculate Adv-Robsutness Metric
        Radv_dict = {}
        for label, K, exp in self.experiment_iterator(attack_advrobustness,
                                                      models, K_l):
            rob_g_wrt_y_l = exp.get_measurement("rob_g_wrt_y_l", budget)
            rob_f_wrt_y_l = exp.get_measurement("rob_f_wrt_y_l", budget)
            rob_f_wrt_g_l = exp.get_measurement("rob_f_wrt_g_l", budget)
            if label not in Radv_dict:
                Radv_dict[label] = {}
            if name == "f1-robustness":
                Radv_dict[label][K] = rob_f_wrt_g_l / rob_g_wrt_y_l
            elif name == "f1-min-changes":
                Radv_dict[label][K] = (rob_f_wrt_g_l + 1) / (rob_g_wrt_y_l + 1)
            elif name == "f1-robustness-v2":
                Radv_dict[label][K] = rob_f_wrt_g_l / rob_g_wrt_y_l
            elif name == "f1-min-changes-v2":
                Radv_dict[label][K] = exp.avg_min_changes_to_flip_advrob
            else:
                assert False
        # Calculate F1-Score
        for label in Rover_dict:
            if spacing == "even":
                x = [i for i in range(len(K_l))]
            else:
                x = K_l
            y = []
            yerr = []
            for K in K_l:
                f1_score = 2 * Rover_dict[label][K] * Radv_dict[label][K] / (Rover_dict[label][K] + Radv_dict[label][K])
                y.append(np.mean(f1_score))
                yerr.append(scipy.stats.sem(f1_score))
            color, linestyle = self.get_style(label)
            if errorbars:
                ax.errorbar(x, y, yerr=yerr, marker="o", label=label, capsize=5,
                            color=color, linestyle=linestyle, linewidth=2.5,
                            markersize=8)
            else:
                ax.plot(x, y, marker="o", label=label, color=color, 
                        linestyle=linestyle)
        if ylabel is None:
            ylabel=name 
        ax.xaxis.get_major_formatter()._usetex = False
        ax.yaxis.get_major_formatter()._usetex = False
        ax.set_ylabel("", fontsize=xylabelfont)
        ax.set_xlabel("K", fontsize=xylabelfont)
        ax.set_title(title, fontsize=titlefont)
        ax.set_yticklabels([f"{round(i, 2):.2f}" for i in ax.get_yticks()], fontsize=tickfont)
        ax.set_xticklabels(ax.get_xticks(), fontsize=tickfont)
        if title is None:
            title=name
        if spacing == "log":
            ax.set_xscale('log')
            xticks = np.sort(K_l + [0.2, 10])
            # or symlog with only k=10
            ax.set_xticks(xticks, minor=True)
        elif spacing == "even":
            ax.xaxis.set_ticks(x, minor=False)
            ax.xaxis.set_ticklabels(K_l)
            ax.set_xlim(left=-0.3)
        else:
            ax.set_xticks(K_l, minor=False)
            ax.set_xlim(left=0.)
        ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.grid()
        ax.xaxis.grid()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.86, box.height])
        leg = ax.legend(loc=legend_loc, ncol=1, shadow=False,
                        bbox_to_anchor=(1, 1), frameon=False, markerscale=1,
                        prop=dict(size=legendfont))
        #leg.get_lines()[0].set_linestyle
        #for i in leg.legendHandles:
        #    i.set_linestyle(":")
        #ax.set_aspect(1.618)
        fig.set_tight_layout(True)
        plt.show()

    def starplot(self, name: str, attack: str, models: List[str], 
                 K: List[float], max_degree: int=None, logplot: bool=False,
                 errorbars: bool=True, ylabel: str=None, title: str=None,
                 bayes_label="Bayes"):
        """Generate starplot (w.r.t. degree plots).

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
        #h, w = matplotlib.figure.figaspect(1.618 / 1.15)
        #fig, axs = plt.subplots(figsize=(w,h))
        fig, axs = plt.subplots()
        color_list = ['r', 'tab:green', 'b', 'lime', 'c', 'k', "antiquewhite"]
        linestyle_list = ['-', '--', ':', '-.']
        axs.set_prop_cycle(cycler('linestyle', linestyle_list)*
                           cycler('color', color_list))
        bayes_added = False
        for label, K, exp in self.experiment_iterator(attack, models, [K]):  
            avg_f_wrt_y = exp.robustness_statistics["avg_avg_gnn_robust_when_both"]
            std_f_wrt_y = exp.robustness_statistics["sem_avg_gnn_robust_when_both"]
            avg_g_wrt_y = exp.robustness_statistics["avg_avg_bayes_robust_when_both"]
            std_g_wrt_y = exp.robustness_statistics["sem_avg_bayes_robust_when_both"]
            avg_f_wrt_g = exp.robustness_statistics["avg_avg_gnn_wrt_bayes_robust"]
            std_f_wrt_g = exp.robustness_statistics["sem_avg_gnn_wrt_bayes_robust"]
            
            x = np.sort([int(i) for i in avg_f_wrt_y.keys()])
            x = x[x <= max_degree]
            ordered_avg_f_wrt_y = [avg_f_wrt_y[str(i)] for i in x]
            ordered_std_f_wrt_y = [std_f_wrt_y[str(i)] for i in x]
            ordered_avg_g_wrt_y = [avg_g_wrt_y[str(i)] for i in x]
            ordered_std_g_wrt_y = [std_g_wrt_y[str(i)] for i in x]
            ordered_avg_f_wrt_g = [avg_f_wrt_g[str(i)] for i in x]
            ordered_std_f_wrt_g = [std_f_wrt_g[str(i)] for i in x]
            color, linestyle = self.get_style(label)
            if errorbars:
                if not bayes_added:
                    axs.errorbar(x, ordered_avg_g_wrt_y, 
                                yerr=ordered_std_g_wrt_y, fmt="s:", label=f"{bayes_label}", capsize=3,
                                color="tab:olive")
                if name == "f_wrt_y" or name == "both":
                    axs.errorbar(x, ordered_avg_f_wrt_y,  
                                yerr=ordered_std_f_wrt_y, marker="o", label=f"{label}", capsize=3,
                                color=color, linestyle=linestyle)
                if name == "f_wrt_g" or name == "both":
                    axs.errorbar(ordered_avg_f_wrt_g, x, 
                                yerr=ordered_std_f_wrt_g, marker="o", label=f"{label} w.r.t. Bayes", capsize=3)
            else:
                if name == "f_wrt_y" or name == "both":
                    axs.plot(ordered_avg_f_wrt_y, x, marker='o', label=f"{label}")
                if not bayes_added:
                    axs.plot(ordered_avg_g_wrt_y, x, 's:', label=bayes_label)
                if name == "f_wrt_g" or name == "both":
                    axs.plot(ordered_avg_f_wrt_g, x, marker='o', label=f"{label} w.r.t. Bayes")
            bayes_added = True
        if ylabel is None:
            ylabel=name
        axs.set_ylabel(ylabel, fontsize=13)
        axs.set_xlabel("Node Degree", fontsize=13)
        if logplot:
            axs.set_yscale('log')
        if title is None:
            title=name
        axs.set_title(title, fontweight="bold", fontsize=15)
        start_x, end_x = axs.get_xlim()
        start_y, end_y = axs.get_ylim()
        # filling:
        for label, K, exp in self.experiment_iterator(attack, models, [K]):  
            avg_g_wrt_y = exp.robustness_statistics["avg_avg_bayes_robust_when_both"]
            x = np.sort([int(i) for i in avg_f_wrt_y.keys()])
            x = x[x <= max_degree]
            ordered_avg_g_wrt_y = [avg_g_wrt_y[str(i)] for i in x]
            start_y, end_y = axs.get_ylim()
            axs.fill_between(x, start_y, ordered_avg_g_wrt_y, interpolate=True, 
                                color='tab:olive', alpha=0.1)
            axs.fill_between(x, ordered_avg_g_wrt_y, end_y, 
                                interpolate=True, color='red', alpha=0.1)
            break

        #axs.xaxis.set_ticks(np.arange(0, end_x, step=1))
        axs.yaxis.set_ticks([1, 10, 100], ["1", "10", "100"])
        axs.set_xticklabels([int(i) for i in axs.get_xticks()], fontsize=10)
        axs.set_yticklabels(axs.get_yticks(), fontsize=10)
        #axs.xaxis.set_ticks_position('top')
        #axs.xaxis.set_label_position('top')
        box = axs.get_position()
        axs.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.9])
        axs.legend(loc="center left", ncol=1, shadow=False,
                    bbox_to_anchor=(1, 0.5), frameon=False, fontsize=12)
        axs.invert_yaxis()
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0)
        t = axs.text(
            4, 2, "Preserved\nSemantics", 
            rotation=0, size=14, color="tab:olive", fontstyle="oblique",
            fontweight="bold",
            bbox=bbox_props)
        t = axs.text(
            3, 38, "Changed\nSemantics", 
            rotation=0, size=14, color="r", fontstyle="oblique",
            fontweight="bold",
            bbox=bbox_props)
        #plt.grid(axis="both")
        plt.show()

    def starplot_workshop(self, name: str, attack: str, models: List[str], 
                 K: List[float], max_degree: int=None, logplot: bool=False,
                 errorbars: bool=True, ylabel: str=None, title: str=None,
                 bayes_label="Bayes", xyfontsize=13, ticksize=10, legendsize=12,
                 ratio=1.618, weight=1.5, titlefont=15, invert_axis=True,
                 linewidth=1, markersize=5):
        
        """Generate starplot (w.r.t. degree plots).

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
        #h, w = matplotlib.figure.figaspect(ratio / weight)
        #fig, axs = plt.subplots(figsize=(w,h))
        fig, axs = plt.subplots()
        color_list = ['r', 'tab:green', 'b', 'lime', 'c', 'k', "antiquewhite"]
        linestyle_list = ['-', '--', ':', '-.']
        axs.set_prop_cycle(cycler('linestyle', linestyle_list)*
                           cycler('color', color_list))
        bayes_added = False
        for label, K, exp in self.experiment_iterator(attack, models, [K]):  
            avg_f_wrt_y = exp.robustness_statistics["avg_avg_gnn_robust_when_both"]
            std_f_wrt_y = exp.robustness_statistics["sem_avg_gnn_robust_when_both"]
            avg_g_wrt_y = exp.robustness_statistics["avg_avg_bayes_robust_when_both"]
            std_g_wrt_y = exp.robustness_statistics["sem_avg_bayes_robust_when_both"]
            avg_f_wrt_g = exp.robustness_statistics["avg_avg_gnn_wrt_bayes_robust"]
            std_f_wrt_g = exp.robustness_statistics["sem_avg_gnn_wrt_bayes_robust"]
            
            x = np.sort([int(i) for i in avg_f_wrt_y.keys()])
            x = x[x <= max_degree]
            ordered_avg_f_wrt_y = [avg_f_wrt_y[str(i)] for i in x]
            ordered_std_f_wrt_y = [std_f_wrt_y[str(i)] for i in x]
            ordered_avg_g_wrt_y = [avg_g_wrt_y[str(i)] for i in x]
            ordered_std_g_wrt_y = [std_g_wrt_y[str(i)] for i in x]
            ordered_avg_f_wrt_g = [avg_f_wrt_g[str(i)] for i in x]
            ordered_std_f_wrt_g = [std_f_wrt_g[str(i)] for i in x]
            color, linestyle = self.get_style(label)
            if errorbars:
                if not bayes_added:
                    axs.errorbar(x, ordered_avg_g_wrt_y, 
                                yerr=ordered_std_g_wrt_y, fmt="s:", label=bayes_label, capsize=3,
                                color="tab:olive", linewidth=linewidth, markersize=markersize)
                if name == "f_wrt_y" or name == "both":
                    axs.errorbar(x, ordered_avg_f_wrt_y,  
                                yerr=ordered_std_f_wrt_y, marker="o", label=f"\\textbf{{{label}}}", capsize=3,
                                color=color, linestyle=linestyle, linewidth=linewidth, markersize=markersize)
                if name == "f_wrt_g" or name == "both":
                    axs.errorbar(ordered_avg_f_wrt_g, x, 
                                yerr=ordered_std_f_wrt_g, marker="o", label=f"{label} w.r.t. Bayes", capsize=3)
            else:
                if name == "f_wrt_y" or name == "both":
                    axs.plot(ordered_avg_f_wrt_y, x, marker='o', label=f"{label}")
                if not bayes_added:
                    axs.plot(ordered_avg_g_wrt_y, x, 's:', label=bayes_label)
                if name == "f_wrt_g" or name == "both":
                    axs.plot(ordered_avg_f_wrt_g, x, marker='o', label=f"{label} w.r.t. Bayes")
            bayes_added = True
        if ylabel is None:
            ylabel=name
        axs.set_ylabel(ylabel, fontsize=xyfontsize, fontweight="bold")
        axs.set_xlabel(r"\textbf{Node Degree}", fontsize=xyfontsize, fontweight="bold")
        if logplot:
            axs.set_yscale('log')
        if title is None:
            title=name
        axs.set_title(title, fontweight="bold", fontsize=titlefont)
        start_x, end_x = axs.get_xlim()
        start_y, end_y = axs.get_ylim()
        # filling:
        for label, K, exp in self.experiment_iterator(attack, models, [K]):  
            avg_g_wrt_y = exp.robustness_statistics["avg_avg_bayes_robust_when_both"]
            x = np.sort([int(i) for i in avg_f_wrt_y.keys()])
            x = x[x <= max_degree]
            ordered_avg_g_wrt_y = [avg_g_wrt_y[str(i)] for i in x]
            start_y, end_y = axs.get_ylim()
            axs.fill_between(x, start_y, ordered_avg_g_wrt_y, interpolate=True, 
                                color='tab:olive', alpha=0.1)
            axs.fill_between(x, ordered_avg_g_wrt_y, end_y, 
                                interpolate=True, color='red', alpha=0.1)
            break

        #axs.xaxis.set_ticks(np.arange(0, end_x, step=1))
        axs.yaxis.set_ticks([1, 10, 100], ["1", "10", "100"])
        axs.set_xticklabels([int(i) for i in axs.get_xticks()], 
                            fontsize=ticksize)
        axs.set_yticklabels(axs.get_yticks(), fontsize=ticksize)
        #axs.xaxis.set_ticks_position('top')
        #axs.xaxis.set_label_position('top')
        box = axs.get_position()
        axs.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.7])
        axs.legend(loc="center left", ncol=1, shadow=False,
                    bbox_to_anchor=(1, 0.5), frameon=False, fontsize=legendsize)
        if invert_axis:
            axs.invert_yaxis()
            bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0)
            t = axs.text(
                4.2, 1.4, r"\begin{center}\textbf{\textit{Preserved}}\\\textbf{\textit{semantics}}\end{center}", 
                rotation=0, size=15, color="tab:olive", fontstyle="oblique",
                fontweight="bold",
                bbox=bbox_props)
            t = axs.text(
                3, 24, r"\begin{center}\textbf{\textit{Changed}}\\\textbf{\textit{semantics}}\end{center}", 
                rotation=0, size=15, color="r", fontstyle="oblique",
                fontweight="bold",
                bbox=bbox_props)
        else:
            bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0)
            t = axs.text(
                4, 1.6, r"\begin{center}\textbf{Preserved}\\\textbf{semantics}\end{center}", 
                rotation=0, size=15, color="tab:olive", fontstyle="oblique",
                fontweight="bold",
                bbox=bbox_props)
            t = axs.text(
                3, 32, r"\begin{center}\textbf{Changed}\\\textbf{semantics}\end{center}", 
                rotation=0, size=15, color="r", fontstyle="oblique",
                fontweight="bold",
                bbox=bbox_props)
        #plt.grid(axis="both")
        fig.set_tight_layout(True)
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


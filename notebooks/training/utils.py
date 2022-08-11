from pymongo import MongoClient
from typing import Any, Dict, List

import numpy as np
import pandas as pd


URI = "mongodb://gosl:Wuyg6KTV@fs.kdd.in.tum.de:27017/gosl?authMechanism=SCRAM-SHA-1"


def assert_equal_dicts(d1: Dict[str, Any], d2: Dict[str, Any]):
    """Checks if every key of d1 is in d2 and they store same value."""
    for key in d1:
        assert key in d2
        assert d1[key] == d2[key]
        

class Experiment:
    """An experiment refers to the results optained by a particular 
    hyperparameter setting averaged over multiple seeds.
    
    Allows comparison of experiments by average validation loss. """
    def __init__(self, experiment_list: List[Dict[str, Any]]):
        assert len(experiment_list) >= 0
        self.individual_experiments = experiment_list
        self.id = experiment_list[0]["_id"]
        self.hyperparameters = experiment_list[0]["config"]
        self.assert_same_hyperparameters(self.individual_experiments)
        self.extract_results(self.individual_experiments)

    def assert_same_hyperparameters(self, experiment_list: List[Dict[str, Any]]) -> None:
        """Sanity check if all experiments summarized by an instance of this 
        class indeed have the same configuration."""
        data_params_l = []
        model_params_l = []
        train_params_l = []
        for experiment in experiment_list:
            data_params_l.append(experiment["config"]["data_params"])
            model_params_l.append(experiment["config"]["model_params"])
            train_params_l.append(experiment["config"]["train_params"])
        for i in range(1, len(experiment_list)):
            assert_equal_dicts(data_params_l[0], data_params_l[i])
            assert_equal_dicts(model_params_l[0], model_params_l[i])
            assert_equal_dicts(train_params_l[0], train_params_l[i])

    def extract_results(self, experiment_list: List[Dict[str, Any]]):
        """Collect important training results into a list."""
        self.validation_loss_l = []
        self.validation_acc_l = []
        self.training_loss_l = []
        self.training_acc_l = []
        self.best_epoch_l = []
        self.training_epoch_l = []
        for experiment in experiment_list:
            results = experiment["result"]
            self.validation_loss_l.append(results["best_validation_loss"])
            self.validation_acc_l.append(results["best_validation_accuracy"])
            self.training_loss_l.append(results["best_training_loss"])
            self.training_acc_l.append(results["best_training_accuracy"])
            self.best_epoch_l.append(results["best_epoch"])
            self.training_epoch_l.append(results["training_epochs"])

    @property
    def validation_loss(self):
        return float(np.mean(self.validation_loss_l))

    @property
    def validation_loss_std(self):
        return float(np.std(self.validation_loss_l))

    @property
    def training_loss(self):
        return float(np.mean(self.training_loss_l))

    @property
    def training_loss_std(self):
        return float(np.std(self.training_loss_l))

    @property
    def validation_accuracy(self):
        return float(np.mean(self.validation_acc_l))

    @property
    def validation_accuracy_std(self):
        return float(np.std(self.validation_acc_l))

    @property
    def training_accuracy(self):
        return float(np.mean(self.training_acc_l))

    @property
    def training_accuracy_std(self):
        return float(np.std(self.training_acc_l))
    
    @property
    def best_epoch(self):
        return float(np.mean(self.best_epoch_l))

    @property
    def best_epoch_std(self):
        return float(np.std(self.best_epoch_l))

    @property
    def training_epochs(self):
        return float(np.mean(self.training_epoch_l))

    @property
    def training_epochs_std(self):
        return float(np.std(self.training_epoch_l))

    @property
    def K(self):
        return self.hyperparameters["data_params"]["K"]
            
    # The six rich comparison methods
    def __lt__(self, other):
        return self.validation_loss < other.validation_loss
    
    def __le__(self, other):
        return self.validation_loss <= other.validation_loss
    
    def __eq__(self, other):
        return self.validation_loss == other.validation_loss
    
    def __ne__(self, other):
        return self.validation_loss != other.validation_loss
    
    def __gt__(self, other):
        return self.validation_loss > other.validation_loss
    
    def __ge__(self, other):
        return self.validation_loss >= other.validation_loss

    def __str__(self):
        print_str = "ID: " + str(self.individual_experiments[0]["_id"])
        print_str += f" ValLoss: {self.validation_loss:.4f} +- {self.validation_loss_std:.4f}"
        print_str += f" ValAcc: {self.validation_accuracy*100:.2f} +- {self.validation_accuracy_std*100:.2f}"
        return print_str


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

    def load_experiments(self, start_id: int, end_id: int, n_seeds: int, 
                         K: float) -> List[Experiment]:
        """Return Experiments between start_id and end_id with K=K.
        
        Assumes that one experiment consists of multiple seeds which are stored
        consecutively in the mongodb. 
        """
        experiment_ids = [i for i in range(start_id, end_id + 1, n_seeds)]
        experiments = [self.load_experiment(i, n_seeds) for i in experiment_ids]
        experiments_with_K = []
        for experiment in experiments:
            if experiment.K == K:
                experiments_with_K.append(experiment)
        experiments_with_K.sort()
        return experiments_with_K


#def experiment_list_to_dataframe(experiments: List[Experiment]) ->
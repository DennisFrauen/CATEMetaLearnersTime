import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from data.simulations import Simulator
import logging

logger = logging.getLogger(__name__)


class SyntheticDataset():
    def __init__(self, X: np.array, A: np.array, Y: np.array, active_entries: np.array, tau: int, targets: np.array, weights: np.array,
                 subset_name: str):
        self.subset_name = subset_name
        T = X.shape[1]
        user_sizes = np.squeeze(active_entries.sum(1))
        treatments = A.astype(float)
        future_treatments = treatments[:, 1:T-tau, :]
        if tau > 0:
            for i in range(tau):
                future_treatments = np.concatenate((future_treatments, treatments[:, 2+i:T-tau+i+1, :]), axis=2)
        if targets is None:
            # Targets are tau+1 step ahead outcomes
            targets = Y[:, tau+1:, :]
        if weights is None:
            weights = np.ones_like(Y[:, tau+1:, :])


        self.data = {
            'sequence_lengths': user_sizes - tau - 1,
            'prev_treatments': treatments[:, :-1-tau, :],
            'covariates': X[:, 1:T-tau, :],
            'future_treatments': future_treatments,
            'active_entries': active_entries[:, 1:T-tau, :],
            'targets': targets,
            'prev_outcomes': Y[:, :-1-tau, :],
            'weights': weights,
        }

    def get_pytorch_data(self):
        dataset = torch.utils.data.TensorDataset(torch.tensor(self.data['targets'], dtype=torch.float32),
                                                 torch.tensor(self.data['covariates'], dtype=torch.float32),
                                                 torch.tensor(self.data['prev_outcomes'], dtype=torch.float32),
                                                 torch.tensor(self.data['prev_treatments'], dtype=torch.float32),
                                                 torch.tensor(self.data['future_treatments'], dtype=torch.float32),
                                                 torch.tensor(self.data['active_entries'], dtype=torch.float32),
                                                 torch.tensor(self.data['weights'], dtype=torch.float32))
        return dataset


class SyntheticDataLoader(L.LightningDataModule):
    def __init__(self, simulator: Simulator, num_patients: dict, seed: int, a_int: list, batch_size: dict,
                 standardize=True, **kwargs):
        """
        Args:
        :param simulator: Simulator object
        :param num_patients: Number of patients. Values: {'train': int, 'val': int, 'test': int}
        :param a_int: intervention sequence. Values: list
        """

        super().__init__()
        self.seed = seed
        self.batch_size = batch_size
        np.random.seed(seed)
        self.simulator = simulator
        self.num_patients = num_patients
        self.a_int = a_int
        self.train_raw, self.val_raw, self.test_raw = self.simulate()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.x_dim = self.simulator.config["p"]
        self.T_dim = self.simulator.config["T"]
        self.standardized = False
        if standardize:
            self.scaling_params = {"y_mean": self.train_raw['Y'].mean(), "y_std": self.train_raw['Y'].std(),
                                   "x_mean": self.train_raw['X'].mean((0, 1)), "x_std": self.train_raw['X'].std((0, 1))}
            self.setup("init")
        else:
            self.scaling_params = None
    def simulate(self):
        X, A, Y, active_entries = self.simulator.simulate_factual(n=self.num_patients['train'])
        train_f_raw = {"X": X, "A": A, "Y": Y, "active_entries": active_entries}

        X, A, Y, active_entries = self.simulator.simulate_factual(n=self.num_patients['val'])
        val_f_raw = {"X": X, "A": A, "Y": Y, "active_entries": active_entries}

        X, A, Y, active_entries = self.simulator.simulate_intervention(n=self.num_patients['test'], a_int=self.a_int)
        test_int_raw = {"X": X, "A": A, "Y": Y, "active_entries": active_entries}
        return train_f_raw, val_f_raw, test_int_raw

    def setup(self, stage: str):
        # Scale data
        if self.scaling_params is not None and not self.standardized:
            logger.info(f'Processing datasets before training')
            self.train_raw["Y"] = (self.train_raw["Y"] - self.scaling_params["y_mean"]) / self.scaling_params["y_std"]
            self.train_raw["X"] = (self.train_raw["X"] - self.scaling_params["x_mean"]) / self.scaling_params["x_std"]
            self.val_raw["Y"] = (self.val_raw["Y"] - self.scaling_params["y_mean"]) / self.scaling_params["y_std"]
            self.val_raw["X"] = (self.val_raw["X"] - self.scaling_params["x_mean"]) / self.scaling_params["x_std"]
            self.test_raw["Y"] = (self.test_raw["Y"] - self.scaling_params["y_mean"]) / self.scaling_params["y_std"]
            self.test_raw["X"] = (self.test_raw["X"] - self.scaling_params["x_mean"]) / self.scaling_params["x_std"]
            self.standardized = True

    def prepare_training(self, tau, targets=None, weights=None):
        # Create Dataset objects
        # tau = forecast horizon, 0 corresponds to 1-step ahead prediction
        # targets = new targets variables, should be of shape (batch size, T - tau - 1, 1)
        if targets is None:
            targets = {"train": None, "val": None, "test": None}
        if "test" not in targets:
            targets["test"] = None
        if weights is None:
            weights = {"train": None, "val": None, "test": None}
        if "test" not in weights:
            weights["test"] = None
        self.train_dataset = SyntheticDataset(self.train_raw["X"], self.train_raw["A"], self.train_raw["Y"],
                                              self.train_raw["active_entries"], tau, targets["train"], weights["train"], 'train')
        self.val_dataset = SyntheticDataset(self.val_raw["X"], self.val_raw["A"], self.val_raw["Y"],
                                            self.val_raw["active_entries"], tau, targets["val"], weights["val"], 'val')
        self.test_dataset = SyntheticDataset(self.test_raw["X"], self.test_raw["A"], self.test_raw["Y"],
                                             self.test_raw["active_entries"], tau, targets["test"], weights["test"], 'test')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset.get_pytorch_data(),
            batch_size=self.batch_size['train'],
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset.get_pytorch_data(),
            batch_size=self.batch_size['val'],
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset.get_pytorch_data(),
            batch_size=self.batch_size['test'],
            shuffle=False
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size['test'],
            shuffle=False
        )

    def get_propensities(self):
        propensities = {}
        data_indicator = ['train', 'val', 'test']
        data_raw_unscaled = self.get_unscaled_data()
        for i, data in enumerate(data_raw_unscaled):
            propensity = np.zeros_like(data['A'], dtype=float)
            propensity[:, 0, :] = self.simulator.propensity(np.transpose(data["X"][:, 0:1, :], (0, 2, 1)), None, None)
            for t in range(1, data['A'].shape[1]):
                propensity[:, t, :] = self.simulator.propensity(np.transpose(data['X'][:, 0:t+1, :], (0, 2, 1)), np.transpose(data['A'][:, 0:t, :], (0, 2, 1)), np.transpose(data['Y'][:, 0:t, :], (0, 2, 1)))
            propensities[data_indicator[i]] = propensity[:, 1:, :]
        return propensities

    def get_y_means(self):
        y_means = {}
        data_indicator = ['train', 'val', 'test']
        data_raw_unscaled = self.get_unscaled_data()
        for i, data in enumerate(data_raw_unscaled):
            y_mean = np.zeros_like(data['Y'], dtype=float)
            y_mean[:, 0, :] = self.simulator.y_mean(np.transpose(data["X"][:, 0:1, :], (0, 2, 1)), np.transpose(data["A"][:, 0:1, :], (0, 2, 1)), None)
            for t in range(1, data['Y'].shape[1]):
                y_mean[:, t, :] = self.simulator.y_mean(np.transpose(data['X'][:, 0:t+1, :], (0, 2, 1)), np.transpose(data['A'][:, 0:t+1, :], (0, 2, 1)), np.transpose(data['Y'][:, 0:t, :], (0, 2, 1)))
            y_means[data_indicator[i]] = (y_mean[:, 1:, :] - self.scaling_params["y_mean"]) / self.scaling_params["y_std"]
        return y_means

    def get_unscaled_data(self):
        data_indicator = ['train', 'val', 'test']
        data_raw = [self.train_raw, self.val_raw, self.test_raw]
        data_raw_unscaled = []
        for i, data in enumerate(data_raw):
            X_unscaled = data['X'] * self.scaling_params['x_std'] + self.scaling_params['x_mean']
            Y_unscaled = data['Y'] * self.scaling_params['y_std'] + self.scaling_params['y_mean']
            data_raw_unscaled.append({"X": X_unscaled, "A": data['A'], "Y": Y_unscaled, "active_entries": data['active_entries']})
        return data_raw_unscaled
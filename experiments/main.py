import numpy as np
import torch

import utils.utils as utils
import wandb
import random
from data.simulations import Sim_Autoregressive, Sim_Autoregressive_Propensity, Sim_Autoregressive_Overlap
from data.dataset import SyntheticDataLoader
from models.base_transformer import Transformer_History, Transformer_History_Treatment
from lightning import Trainer


def run_experiment(config, exp_function, end_function=None):
    if config["run"]["logging"]:
        raise ValueError("Logging needs to be turned off")
    seed = config["run"]["seed"]
    result_path = utils.get_project_path() + "/experiments/" + config["run"]["name"] + "/results/"
    results = []

    for i in range(config["run"]["n_runs"]):
        print("Starting run " + str(i + 1) + " of " + str(config["run"]["n_runs"]))
        utils.set_seed(seed)
        seed = random.randint(0, 1000000)
        if config["run"]["run_start_index"] <= i + 1:
            # Load data
            data_loaders = select_dataset(config, seed)
            # Create model config
            config_nuisance, config_meta = create_model_configs(config, data_loaders[0])
            # Train or load models
            nuisance_models, meta_learners = train_models(config["run"], config_nuisance, config_meta, data_loaders[0],
                                                          run=i,
                                                          seed=seed)
            # Run experiment, result should be dictionary of pandas dataframes
            result = exp_function(config["run"], data_loaders, nuisance_models, meta_learners)
            # Save results
            if "save" in config["run"]:
                if config["run"]["save"]:
                    if result is not None:
                        for key in result.keys():
                            result[key].to_pickle(result_path + key + "_run_" + str(i) + ".pkl")
            results.append(result)
    if end_function is not None:
        end_function(config, results)
    print("Experiment finished")


def eval_test(config_run, data_loaders, nuisance_models, meta_learners):
    predictions = test_predictions(config_run, data_loaders[0], nuisance_models, meta_learners)
    test_results = {}
    data_loaders[0].prepare_training(tau=0)
    outcomes_test = data_loaders[0].test_dataset.get_pytorch_data().tensors[0]
    if len(config_run["interventions"]) == 1:
        if "piha" in predictions.keys():
            predictions["piha"] = predictions["piha"][list(predictions["piha"].keys())[0]]
        if "pira" in predictions.keys():
            predictions["pira"] = predictions["pira"][list(predictions["pira"].keys())[0]]
    elif len(config_run["interventions"]) == 2:
        data_loaders[1].prepare_training(tau=0)
        outcomes_test -= data_loaders[1].test_dataset.get_pytorch_data().tensors[0]
        if "piha" in predictions.keys():
            predictions["piha"] = predictions["piha"][list(predictions["piha"].keys())[0]] - predictions["piha"][
                list(predictions["piha"].keys())[1]]
        if "pira" in predictions.keys():
            predictions["pira"] = predictions["pira"][list(predictions["pira"].keys())[0]] - \
                                    predictions["pira"][list(predictions["pira"].keys())[1]]
    else:
        raise ValueError("Only 1 or 2 intervention sequences are supported for evaluation on test data")
    # Calculate RMSE for all learners
    for learner in config_run["learners"]:
        test_results[learner] = np.sqrt(
            np.mean((predictions[learner][:, -1, 0].detach().numpy() - outcomes_test[:, -1, 0].detach().numpy()) ** 2))
    return test_results


def test_predictions(config_run, data_loader, nuisance_models, meta_learners):
    tau = len(config_run["interventions"]["a"]) - 1
    learner_keys = config_run["learners"]
    interventions = config_run["interventions"]
    predictions = {}
    if "piha" in learner_keys:
        data_loader.prepare_training(tau=tau)
        hist_adj = {}
        for int_seq in interventions:
            hist_adj[int_seq] = nuisance_models["ha"]["ha_" + int_seq].predict_step(data_loader.test_dataset.get_pytorch_data().tensors)
        predictions["piha"] = hist_adj
    if "pira" in learner_keys:
        mu = {}
        for int_seq in interventions:
            model_name = "mu_0_" + int_seq
            data_loader.prepare_training(tau=tau)
            mu[int_seq] = nuisance_models["mu"][model_name].predict_step(
                data_loader.test_dataset.get_pytorch_data().tensors)
        predictions["pira"] = mu
    if "ha" in learner_keys:
        data_loader.prepare_training(tau=tau, targets=None)
        predictions["ha"] = meta_learners["ha"].predict_step(data_loader.test_dataset.get_pytorch_data().tensors)
    if "ra" in learner_keys:
        data_loader.prepare_training(tau=tau, targets=None)
        predictions["ra"] = meta_learners["ra"].predict_step(data_loader.test_dataset.get_pytorch_data().tensors)
    if "ipw" in learner_keys:
        data_loader.prepare_training(tau=tau, targets=None)
        predictions["ipw"] = meta_learners["ipw"].predict_step(data_loader.test_dataset.get_pytorch_data().tensors)
    if "dr" in learner_keys:
        data_loader.prepare_training(tau=tau, targets=None)
        predictions["dr"] = meta_learners["dr"].predict_step(data_loader.test_dataset.get_pytorch_data().tensors)
    if "ivwdr" in learner_keys:
        data_loader.prepare_training(tau=tau, targets=None)
        predictions["ivwdr"] = meta_learners["ivwdr"].predict_step(data_loader.test_dataset.get_pytorch_data().tensors)
    return predictions


def train_models(config_run, config_nuisance, config_meta, data_loader, run=0, seed=0):
    savepath = "/experiments/" + config_run["name"] + "/saved_models/run_" + str(run) + "/"
    tau = len(config_run["interventions"]["a"]) - 1

    # Propensity model
    utils.set_seed(seed)
    if config_nuisance["propensity"] is not None:
        model_propensity = Transformer_History(config_nuisance["propensity"])
        if config_run["train_propensity"]:
            print("Fitting propensity model")
            data_loader.prepare_training(tau=0)
            model_propensity.fit(data_loader, "propensity", logging=config_run["logging"])
            utils.save_pytorch_model(savepath + "nuisance_models/propensity", model_propensity)
        else:
            print("Load propensity model")
            model_propensity = utils.load_pytorch_model(savepath + "nuisance_models/propensity", model_propensity)
    else:
        model_propensity = None

    # History adjustment model
    models_ha = {}
    utils.set_seed(seed)
    if config_nuisance["ha"] is not None:
        n_train = data_loader.train_raw["A"].shape[0]
        n_val = data_loader.val_raw["A"].shape[0]
        T = data_loader.train_raw["X"].shape[1]
        # Create models
        for int_seq in config_run["interventions"]:
            models_ha["ha_" + int_seq] = Transformer_History(config_nuisance["ha"])
        if config_run["train_history_adjustments"]:
            print("Fitting history adjustment models")
            for int_seq in config_run["interventions"]:
                a_int = config_run["interventions"][int_seq]
                # Calculate indicators to filter data (T-type learner)
                treatments = {"train": torch.zeros((n_train, T - 1 - tau, tau + 1)),
                              "val": torch.zeros((n_val, T - 1 - tau, tau + 1))}
                data_loader.prepare_training(tau=0)
                treatments_train = data_loader.train_dataset.get_pytorch_data().tensors[4]
                treatments_val = data_loader.val_dataset.get_pytorch_data().tensors[4]
                for t in range(T - 1 - tau):
                    treatments["train"][:, t, :] = treatments_train[:, t:t + tau + 1, 0]
                    treatments["val"][:, t, :] = treatments_val[:, t:t + tau + 1, 0]
                indicators = {"train": (
                        treatments["train"] == torch.tensor(a_int, dtype=torch.int).repeat(n_train, T - 1 - tau,
                                                                                           1)).long().prod(dim=-1, keepdim=True),
                              "val": (treatments["val"] == torch.tensor(a_int, dtype=torch.int).repeat(n_val,
                                                                                                       T - 1 - tau,
                                                                                                       1)).long().prod(dim=-1, keepdim=True)}
                utils.set_seed(seed)
                data_loader.prepare_training(tau=tau, weights=indicators)
                current_model_name = "ha_" + int_seq
                models_ha[current_model_name].fit(data_loader, current_model_name,
                                                  logging=config_run["logging"])
                utils.save_pytorch_model(savepath + "nuisance_models/" + current_model_name,
                                         models_ha[current_model_name])
        else:
            print("Load history adjustment model")
            for key in models_ha.keys():
                models_ha[key] = utils.load_pytorch_model(savepath + "nuisance_models/" + key, models_ha[key])
    else:
        models_ha = None

    # Response function models
    models_mu = {}
    utils.set_seed(seed)
    if config_nuisance["mu"] is not None:
        # Create models
        for int_seq in config_run["interventions"]:
            for t in range(tau+1):
                utils.set_seed(seed)
                config_mu_t = config_nuisance["mu"].copy()
                config_mu_t["time_dim"] = config_nuisance["mu"]["time_dim"] - tau + t
                models_mu["mu_" + str(t) + "_" + int_seq] = Transformer_History(config_mu_t)

        if config_run["train_response_functions"]:
            print("Fitting response function models")
            # Fit remaining response function models
            T = data_loader.train_raw["X"].shape[1]
            for int_seq in config_run["interventions"]:
                for t in reversed(range(tau+1)):
                    utils.set_seed(seed)
                    a_int = [config_run["interventions"][int_seq][t]]
                    treat_indicator = {"train": (torch.tensor(data_loader.train_raw["A"], dtype=torch.int) == torch.tensor(a_int, dtype=torch.int).repeat(
                        data_loader.train_raw["A"].shape[0], data_loader.train_raw["A"].shape[1], 1)).long()[:, 1:T-(tau-t), :],
                                    "val": (torch.tensor(data_loader.val_raw["A"], dtype=torch.int) == torch.tensor(a_int, dtype=torch.int).repeat(
                                        data_loader.val_raw["A"].shape[0], data_loader.val_raw["A"].shape[1], 1)).long()[:, 1:T-(tau-t), :]}
                    if t < tau:
                        # Define parent model (previous response function)
                        prev_model_name = "mu_" + str(t + 1) + "_" + int_seq
                        # Predictions of parent model
                        data_loader.prepare_training(tau=tau - t - 1)
                        pred_train = models_mu[prev_model_name].predict_step(
                            data_loader.train_dataset.get_pytorch_data().tensors)
                        pred_val = models_mu[prev_model_name].predict_step(
                            data_loader.val_dataset.get_pytorch_data().tensors)
                        # Prepare data for training, replace outcomes with predictions
                        new_targets = {"train": pred_train[:, 1:, :], "val": pred_val[:, 1:, :],
                                       "test": None}
                        data_loader.prepare_training(tau=tau - t, targets=new_targets, weights=treat_indicator)
                    else:
                        data_loader.prepare_training(tau=0, weights=treat_indicator)
                    # Fit model
                    current_model_name = "mu_" + str(t) + "_" + int_seq
                    models_mu[current_model_name].fit(data_loader, current_model_name,
                                                      logging=config_run["logging"])
                    utils.save_pytorch_model(savepath + "nuisance_models/" + current_model_name,
                                             models_mu[current_model_name])

        else:
            print("Loading response function models")
            for key in models_mu.keys():
                models_mu[key] = utils.load_pytorch_model(savepath + "nuisance_models/" + key, models_mu[key])
    else:
        models_mu = None

    nuisance_models = {"propensity": model_propensity, "mu": models_mu, "ha": models_ha}

    # Construct pseudo-outcomes
    pseudo_outcomes = get_pseudo_outcomes(config_run["interventions"], data_loader, nuisance_models,
                                          config_run["learners"])
    if len(config_run["interventions"]) == 1:
        pseudo_outcomes = pseudo_outcomes[list(config_run["interventions"].keys())[0]]
        # Calculate stabilized inverse variance weights
        pseudo_outcomes["ivw"]["train"] = 1 / pseudo_outcomes["ivw"]["train"]
        pseudo_outcomes["ivw"]["train"] /= pseudo_outcomes["ivw"]["train"].mean()
        pseudo_outcomes["ivw"]["val"] = 1 / pseudo_outcomes["ivw"]["val"]
        pseudo_outcomes["ivw"]["val"] /= pseudo_outcomes["ivw"]["val"].mean()
    elif len(config_run["interventions"]) == 2:
        pseudo_outcomes_a = pseudo_outcomes[list(config_run["interventions"].keys())[0]]
        pseudo_outcomes_b = pseudo_outcomes[list(config_run["interventions"].keys())[1]]
        pseudo_outcomes = {}
        for key in pseudo_outcomes_a.keys():
            pseudo_outcomes[key] = {}
            for data in ["train", "val"]:
                if key != "ivw":
                    pseudo_outcomes[key][data] = pseudo_outcomes_a[key][data] - pseudo_outcomes_b[key][data]
                else:
                    # Calculate stabilized inverse variance weights
                    pseudo_outcomes[key][data] = 1 / (pseudo_outcomes_a[key][data] + pseudo_outcomes_b[key][data])
                    pseudo_outcomes[key][data] /= pseudo_outcomes[key][data].mean()
    else:
        raise ValueError("Only 1 or 2 intervention sequences are supported for meta-learners")

    learner_keys = [value for value in config_run["learners"] if value in ["ha", "ra", "ipw", "dr", "ivwdr"]]
    print("Fitting meta-learners")
    meta_learners = {}
    for learner_key in learner_keys:
        if config_meta[learner_key] is not None:
            model = Transformer_History(config_meta[learner_key])
            if config_run["train_meta_learners"]:
                print("Fitting " + learner_key)
                if learner_key != "ivwdr":
                    data_loader.prepare_training(tau=tau, targets=pseudo_outcomes[learner_key])
                else:
                    data_loader.prepare_training(tau=tau, targets=pseudo_outcomes["dr"], weights=pseudo_outcomes["ivw"])
                model.fit(data_loader, learner_key, logging=config_run["logging"])
                utils.save_pytorch_model(savepath + "meta_learners/" + learner_key, model)
            else:
                print("Loading " + learner_key)
                model = utils.load_pytorch_model(savepath + "meta_learners/" + learner_key, model)
            meta_learners[learner_key] = model
        else:
            raise ValueError("Meta-learner config not found")

    return nuisance_models, meta_learners


def get_pseudo_outcomes(interventions, data_loader, nuisance_models, keys):
    pseudo_outcomes = {}
    n_train = data_loader.train_raw["X"].shape[0]
    n_val = data_loader.val_raw["X"].shape[0]
    T = data_loader.train_raw["X"].shape[1]
    tau = len(interventions["a"]) - 1
    for int_seq in interventions:
        a_int = interventions[int_seq]
        pseudo_outcomes[int_seq] = {}
        # Observed outcomes
        if len({"ha", "ra", "ipw", "dr", "ivwdr"}.intersection(set(keys))) > 0:
            outcomes = {"train": torch.zeros((n_train, T - 1 - tau, tau + 1)),
                        "val": torch.zeros((n_val, T - 1 - tau, tau + 1))}
            data_loader.prepare_training(tau=0)
            outcomes_train = data_loader.train_dataset.get_pytorch_data().tensors[0]
            outcomes_val = data_loader.val_dataset.get_pytorch_data().tensors[0]
            for t in range(T - 1 - tau):
                outcomes["train"][:, t, :] = outcomes_train[:, t:t + tau + 1, 0]
                outcomes["val"][:, t, :] = outcomes_val[:, t:t + tau + 1, 0]
        # Observed treatments
        if len({"ha", "ra", "ipw", "dr", "ivwdr"}.intersection(set(keys))) > 0:
            treatments = {"train": torch.zeros((n_train, T - 1 - tau, tau + 1)),
                          "val": torch.zeros((n_val, T - 1 - tau, tau + 1))}
            data_loader.prepare_training(tau=0)
            treatments_train = data_loader.train_dataset.get_pytorch_data().tensors[4]
            treatments_val = data_loader.val_dataset.get_pytorch_data().tensors[4]
            for t in range(T - 1 - tau):
                treatments["train"][:, t, :] = treatments_train[:, t:t + tau + 1, 0]
                treatments["val"][:, t, :] = treatments_val[:, t:t + tau + 1, 0]
        # Nuisance model predictions
        # Propenstiy score
        if len({"ipw", "dr", "ivwdr"}.intersection(set(keys))) > 0:
            if nuisance_models["propensity"] is None:
                raise ValueError("Propensity model not available")
            propensity = {"train": torch.zeros((n_train, T - 1 - tau, tau + 1)),
                          "val": torch.zeros((n_val, T - 1 - tau, tau + 1))}
            data_loader.prepare_training(tau=0)
            propensity_train = nuisance_models["propensity"].predict_step(
                data_loader.train_dataset.get_pytorch_data().tensors)
            propensity_val = nuisance_models["propensity"].predict_step(
                data_loader.val_dataset.get_pytorch_data().tensors)
            for t in range(T - 1 - tau):
                propensity["train"][:, t, :] = propensity_train[:, t:t + tau + 1, 0] * torch.tensor(a_int,
                                                                                                    dtype=torch.float32).repeat(
                    n_train, 1) + (1 - propensity_train[:, t:t + tau + 1, 0]) * (
                                                       1 - torch.tensor(a_int, dtype=torch.float32).repeat(n_train,
                                                                                                           1))
                propensity["val"][:, t, :] = propensity_val[:, t:t + tau + 1, 0] * torch.tensor(a_int,
                                                                                                dtype=torch.float32).repeat(
                    n_val, 1) + (1 - propensity_val[:, t:t + tau + 1, 0]) * (
                                                     1 - torch.tensor(a_int, dtype=torch.float32).repeat(n_val,
                                                                                                         1))
        # History adjustments
        if len({"piha", "ha"}.intersection(set(keys))) > 0:
            if nuisance_models["ha"] is None:
                raise ValueError("History adjustment model not available")
            hist_adj = {}
            data_loader.prepare_training(tau=tau)
            hist_adj["train"] = nuisance_models["ha"]["ha_" + int_seq].predict_step(data_loader.train_dataset.get_pytorch_data().tensors)
            hist_adj["val"] = nuisance_models["ha"]["ha_" + int_seq].predict_step(data_loader.val_dataset.get_pytorch_data().tensors)

        # Response functions
        if len({"pira", "ra", "dr", "ivwdr"}.intersection(set(keys))) > 0:
            if nuisance_models["mu"] is None:
                raise ValueError("Response function models not available")
            mu = {"train": torch.zeros((n_train, T - 1 - tau, tau + 1)),
                  "val": torch.zeros((n_val, T - 1 - tau, tau + 1))}
            for t in reversed(range(tau + 1)):
                model_name = "mu_" + str(t) + "_" + int_seq
                # Predictions of response function model
                data_loader.prepare_training(tau=tau - t)
                mu_train = nuisance_models["mu"][model_name].predict_step(
                    data_loader.train_dataset.get_pytorch_data().tensors)
                mu_val = nuisance_models["mu"][model_name].predict_step(
                    data_loader.val_dataset.get_pytorch_data().tensors)
                mu["train"][:, :, t] = mu_train[:, t:, 0]
                mu["val"][:, :, t] = mu_val[:, t:, 0]

        # Calculate pseudo-outcomes
        if len({"piha", "ha"}.intersection(set(keys))) > 0:
            pseudo_outcomes[int_seq]["piha"] = hist_adj
        if "pira" in keys:
            pseudo_outcomes[int_seq]["pira"] = {"train": mu["train"][:, :, 0:1], "val": mu["val"][:, :, 0:1]}
        if "ra" in keys:
            a_t = interventions[int_seq][0]
            pseudo_outcomes[int_seq]["ra"] = {}
            for data in ["train", "val"]:
                indicator = (treatments[data][:, :, 0:1] == a_t).long()
                if tau > 0:
                    y_pseudo = indicator * mu[data][:, :, 1:2] + (1 - indicator) * mu[data][:, :, 0:1]
                else:
                    y_pseudo = indicator * outcomes[data] + (1 - indicator) * mu[data][:, :, 0:1]
                pseudo_outcomes[int_seq]["ra"][data] = y_pseudo
        if len({"ipw", "dr", "ivwdr"}.intersection(set(keys))) > 0:
            # IPW pseudo-outcomes
            pseudo_outcomes[int_seq]["ipw"] = {}
            indicators = {"train": (
                    treatments["train"] == torch.tensor(a_int, dtype=torch.int).repeat(n_train, T - 1 - tau,
                                                                                       1)).long(),
                          "val": (treatments["val"] == torch.tensor(a_int, dtype=torch.int).repeat(n_val, T - 1 - tau,
                                                                                                   1)).long()}
            for data in ["train", "val"]:
                y_pseudo = outcomes[data][:, :, -1:] * torch.prod(indicators[data], dim=-1, keepdim=True) / torch.prod(
                    propensity[data], dim=-1, keepdim=True)
                pseudo_outcomes[int_seq]["ipw"][data] = y_pseudo
            # DR pseudo-outcomes
            if len({"dr", "ivwdr"}.intersection(set(keys))) > 0:
                pseudo_outcomes[int_seq]["dr"] = {}
                for data in ["train", "val"]:
                    y_pseudo = pseudo_outcomes[int_seq]["ipw"][data]
                    for t in range(tau + 1):
                        if t > 0:
                            y_pseudo += + (
                                    1 - (indicators[data][:, :, t:(t + 1)] / propensity[data][:, :, t:(t + 1)])) * \
                                        mu[data][:, :, t:(t + 1)] * (torch.prod(indicators[data][:, :, :t], dim=-1,
                                                                                keepdim=True) / torch.prod(
                                propensity[data][:, :, :t], dim=-1, keepdim=True))
                        else:
                            y_pseudo += + (
                                    1 - (indicators[data][:, :, t:(t + 1)] / propensity[data][:, :, t:(t + 1)])) * \
                                        mu[data][:, :, t:(t + 1)]
                    pseudo_outcomes[int_seq]["dr"][data] = y_pseudo

            if "ivwdr" in keys:
                # Calculate inverse variance weights
                pseudo_outcomes[int_seq]["ivw"] = {}
                for data in ["train", "val"]:
                    ivw = 1 / propensity[data][:, :, 0:1]
                    for t in range(1, tau + 1):
                        ivw += torch.prod(indicators[data][:, :, :t + 1], dim=-1, keepdim=True) / (
                                torch.prod(propensity[data][:, :, :t + 1], dim=-1, keepdim=True) ** 2)
                    pseudo_outcomes[int_seq]["ivw"][data] = ivw
    if "ha" in keys:
        raise ValueError("Pseudo-outcomes for history adjustment are not supported")
        pseudo_outcomes["a"]["ha"] = {}
        pseudo_outcomes["b"]["ha"] = {}
        indicators_a = {"train": (
                treatments["train"] == torch.tensor(interventions["a"], dtype=torch.int).repeat(n_train, T - 1 - tau,
                                                                                 1)).long(),
                        "val": (treatments["val"] == torch.tensor(interventions["a"], dtype=torch.int).repeat(n_val, T - 1 - tau,
                                                                                               1)).long()}
        indicators_b = {"train": (
                treatments["train"] == torch.tensor(interventions["b"], dtype=torch.int).repeat(n_train, T - 1 - tau,
                                                                                 1)).long(),
                        "val": (treatments["val"] == torch.tensor(interventions["b"], dtype=torch.int).repeat(n_val, T - 1 - tau,
                                                                                               1)).long()}
        for data in ["train", "val"]:
            pseudo_outcomes["a"]["ha"][data] = (torch.prod(indicators_a[data], dim=-1, keepdim=True) * outcomes[data][:,
                                                                                                      :, -1:]) + \
                                               (torch.prod(indicators_b[data], dim=-1, keepdim=True) * \
                                               pseudo_outcomes["a"]["piha"][data][:, :, 0:1])
            pseudo_outcomes["b"]["ha"][data] = (torch.prod(indicators_b[data], dim=-1, keepdim=True) * outcomes[data][:,
                                                                                                      :, -1:]) + \
                                               (torch.prod(indicators_a[data], dim=-1, keepdim=True) * \
                                               pseudo_outcomes["b"]["piha"][data][:, :, 0:1])
            pseudo_outcomes["a"]["ha"][data] += (1 - torch.prod(indicators_a[data], dim=-1, keepdim=True)) * (
                        1 - torch.prod(indicators_b[data], dim=-1, keepdim=True)) * (
                                                            pseudo_outcomes["a"]["piha"][data][:, :, 0:1] -
                                                            pseudo_outcomes["a"]["piha"][data][:,
                                                            :, 0:1])

    return pseudo_outcomes


def create_model_configs(config, data_loader):
    config_path = "/experiments/" + config["run"]["name"] + "/model_configs/"

    # Base config for transformer models

    config_base = {"input_dim": data_loader.x_dim + 2, "time_dim": data_loader.T_dim - 1}
    # Propensity config
    if len({"ipw", "dr", "ivwdr"}.intersection(set(config["run"]["learners"]))) > 0:
        config_propensity = utils.load_yaml(config_path + "nuisance_models/propensity") | config_base | {
            "output_type": 'classification', "tau": 0}
    else:
        config_propensity = None

    # Response function config
    if len({"pira", "ra", "dr", "ivwdr"}.intersection(set(config["run"]["learners"]))) > 0:
        config_mu = utils.load_yaml(config_path + "nuisance_models/mu") | config_base | {
                "output_type": 'weighted_regression', "tau": 0}
    else:
        config_mu = None

    # History adjustment config
    tau = len(config["run"]["interventions"]["a"]) - 1
    if len({"piha", "ha"}.intersection(set(config["run"]["learners"]))) > 0:
        config_ha = utils.load_yaml(config_path + "nuisance_models/ha") | config_base | {
            "output_type": 'weighted_regression', "tau": tau}
    else:
        config_ha = None

    # Nuisance config
    config_nuisance = {"propensity": config_propensity, "mu": config_mu, "ha": config_ha}

    config_meta = {}
    for learner in config["run"]["learners"]:
        if learner in ["ha", "ra", "ipw", "dr", "ivwdr"]:
            config_meta[learner] = utils.load_yaml(config_path + "meta_learners/" + learner) | config_base | {
                "output_type": 'regression', "tau": tau}
            if learner == "ivwdr":
                config_meta[learner]["output_type"] = "weighted_regression"
    return config_nuisance, config_meta


def select_dataset(config, seed):
    config_data = config["data"]
    if config_data["name"] in ["sim_response1", "sim_response2", "sim_response0"]:
        data_loaders = []
        simulator_ar = Sim_Autoregressive(config_data)
        for int_seq in config["run"]["interventions"]:
            utils.set_seed(seed)
            data_loaders.append(
                SyntheticDataLoader(simulator=simulator_ar, num_patients={"train": config_data["n_train"],
                                                                          "val": config_data["n_val"],
                                                                          "test": config_data["n_test"]},
                                    seed=seed, a_int=config["run"]["interventions"][int_seq],
                                    batch_size={"train": 32, "val": 32, "test": 32}))
        return data_loaders
    if config_data["name"] in ["sim_propensity1", "sim_propensity2", "sim_propensity0"]:
        data_loaders = []
        simulator_ar = Sim_Autoregressive_Propensity(config_data)
        for int_seq in config["run"]["interventions"]:
            utils.set_seed(seed)
            data_loaders.append(
                SyntheticDataLoader(simulator=simulator_ar, num_patients={"train": config_data["n_train"],
                                                                          "val": config_data["n_val"],
                                                                          "test": config_data["n_test"]},
                                    seed=seed, a_int=config["run"]["interventions"][int_seq],
                                    batch_size={"train": 32, "val": 32, "test": 32}))
        return data_loaders
    if config_data["name"] in ["sim_overlap_4", "sim_overlap_3", "sim_overlap_3.5", "sim_overlap_2.5"]:
        data_loaders = []
        simulator_ar = Sim_Autoregressive_Overlap(config_data)
        for int_seq in config["run"]["interventions"]:
            utils.set_seed(seed)
            data_loaders.append(
                SyntheticDataLoader(simulator=simulator_ar, num_patients={"train": config_data["n_train"],
                                                                          "val": config_data["n_val"],
                                                                          "test": config_data["n_test"]},
                                    seed=seed, a_int=config["run"]["interventions"][int_seq],
                                    batch_size={"train": 32, "val": 32, "test": 32}))
        return data_loaders
    else:
        raise ValueError("Dataset not found")

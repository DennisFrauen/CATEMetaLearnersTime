import numpy as np

import utils.utils as utils  # Import module
import utils.plotting as plotting
from experiments.main import run_experiment, eval_test, get_pseudo_outcomes


# Function called in each experiment run
def exp_function(config_run, data_loaders, nuisance_models, meta_learners):
    # Evaluation on test data
    test_results = eval_test(config_run, data_loaders, nuisance_models, meta_learners)
    print("Results on test data:" + str(test_results))

    #pseudo_outcomes = get_pseudo_outcomes(config_run["interventions"], data_loaders[0], nuisance_models,
    #                                      config_run["learners"])
    #ivws_val = (pseudo_outcomes["a"]["ivw"]["val"])[:, :, 0].detach().numpy()
    #propensities_val = data_loaders[0].get_propensities()["val"][:, :, 0]
    #a_val = data_loaders[0].val_raw["A"][:, 1:, 0]
    #data_loaders[0].prepare_training(tau=0)
    #propensities_est = nuisance_models["propensity"].predict_step(
    #    data_loaders[0].val_dataset.get_pytorch_data().tensors).detach().numpy()[:, :, 0]

    if config_run["plotting"]:
        # Visualize propensity score fit
        plotting.plot_propensity_fit(data_loaders, nuisance_models)
        # Visualize 1-step ahead prediction
        #plotting.plot_1step_mu(data_loaders, nuisance_models, y_lim=[-2, 2])

    return test_results


# Function called after all experiment runs
def end_function(config_run, results):
    # Computate averages and standard deviations
    learners = list(results[0].keys())
    results_summary = {}
    for learner in learners:
        results_summary[learner] = {}
        results_learner = [results[i][learner] for i in range(len(results))]
        results_summary[learner]["mean"] = np.mean(results_learner)
        results_summary[learner]["std"] = np.std(results_learner)
    print("Result summary over all runs:" + str(results_summary))



if __name__ == "__main__":
    config_run = utils.load_yaml("/experiments/sim_overlap_2.5/config")
    run_experiment(config_run, exp_function=exp_function, end_function=end_function)
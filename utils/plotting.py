import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import utils.utils as utils


def plot_trajectories(predictions, targets, y_lim=[0, 1]):
    # Targets, predictions are np arrays of shape (n, T, 1)
    n, T = predictions.shape
    for i in range(n):
        if T > 1:
            plt.plot(range(T), predictions[i, :], label="Prediction")
            plt.plot(range(T), targets[i, :], label="Target")
            # fix y-axis to [0, 1]
            plt.ylim(y_lim[0], y_lim[1])
            plt.legend()
            plt.show()
        else:
            # Scatter
            plt.scatter(0, predictions[i, :], label="Prediction")
            plt.scatter(0, targets[i, :], label="Target")
            plt.ylim(y_lim[0], y_lim[1])
            plt.legend()
            plt.show()


def plot_propensity_fit(data_loaders, nuisance_models, y_lim=[0, 1]):
    if nuisance_models[("propensity")] is not None:
        propensities = data_loaders[0].get_propensities()["test"][:, :, 0]
        propensities_est = nuisance_models["propensity"].predict_step(
            data_loaders[0].test_dataset.get_pytorch_data().tensors).detach().numpy()[:, :, 0]
        plot_trajectories(propensities_est[0:3, :], propensities[0:3, :], y_lim=y_lim)


def plot_1step_ha(data_loaders, nuisance_models, y_lim = [-1, 1]):
    if nuisance_models["ha"]is not None:
        y_means = data_loaders[0].get_y_means()["test"][:, :, 0]
        y_mean_est = nuisance_models["ha"].predict_step(
                    data_loaders[0].test_dataset.get_pytorch_data().tensors, a_int=None).detach().numpy()[:, :, 0]
        plot_trajectories(y_mean_est[15:18, :], y_means[15:18, :], y_lim=y_lim)
        print("MSE of y_mean: " + str(((y_mean_est - y_means) ** 2).mean()))

def plot_1step_mu(data_loaders, nuisance_models, y_lim = [-1, 1]):
    if nuisance_models["mu"] is not None:
        y_means = data_loaders[0].get_y_means()["test"][:, :, 0]
        y_mean_est_1 = nuisance_models["mu"]["mu_0_a"].predict_step(
                    data_loaders[0].test_dataset.get_pytorch_data().tensors).detach().numpy()[:, :, 0]
        y_mean_est_0 = nuisance_models["mu"]["mu_0_b"].predict_step(
                    data_loaders[0].test_dataset.get_pytorch_data().tensors).detach().numpy()[:, :, 0]
        treat_test = data_loaders[0].test_dataset.get_pytorch_data().tensors[4][:, :, 0]
        y_mean_est = treat_test * y_mean_est_1 + (1 - treat_test) * y_mean_est_0
        plot_trajectories(y_mean_est[15:18, :], y_means[15:18, :], y_lim=y_lim)
        print("MSE of y_mean: " + str(((y_mean_est - y_means) ** 2).mean()))



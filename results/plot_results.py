import seaborn as sns
import matplotlib.pyplot as plt
import utils.utils as utils
import numpy as np
def plot_results():
    dr_means = [0.2206, 0.2182, 0.3275, 0.4313]
    dr_std = [0.1043, 0.1006, 0.1120, 0.0581]
    ivw_means = [0.1124, 0.1769, 0.1713, 0.1455]
    ivw_std = [0.0821, 0.0574, 0.0573, 0.0422]

    gammas = [2.5, 3, 3.5, 4]

    palette = ["darkblue", "darkred"]
    sns.set(font_scale=1.25)
    sns.set_style("whitegrid")
    # Create seaborn plot with mean lineplots and shaded standard deviation areas
    sns.lineplot(x=gammas, y=dr_means, label="DR", color=palette[0])
    sns.lineplot(x=gammas, y=ivw_means, label="IVW", color=palette[1])
    # Now add the shaded area
    plt.fill_between(gammas, np.array(dr_means) - np.array(dr_std), np.array(dr_means) + np.array(dr_std),
                     color=palette[0], alpha=0.2)
    plt.fill_between(gammas, np.array(ivw_means) - np.array(ivw_std), np.array(ivw_means) + np.array(ivw_std),
                        color=palette[1], alpha=0.2)

    plt.xlabel(r"$\gamma$")
    plt.ylabel("RMSE")
    plt.legend(loc="upper left").set_visible(True)
    plt.savefig(utils.get_project_path() + "/results/plt_overlap.pdf", bbox_inches='tight')
    plt.show()







if __name__ == "__main__":
    plot_results()
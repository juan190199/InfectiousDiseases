import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from sklearn.metrics import r2_score


def plot_estimation_scatter(theta, predictions, parameters, figsize=(20, 5)):
    number_parameters = int(len(parameters))
    fig, ax = plt.subplots(1, number_parameters, figsize=figsize)

    for i in range(len(parameters)):
        ax[i].scatter(predictions[:, i], theta[:, i], color='black', alpha=0.3)
        if i == 0:
            ax[i].set_ylabel('True')

        ax[i].set_xlabel('Estimated')
        ax[i].set_title(parameters[i])
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].plot(ax[i].get_xlim(), ax[i].get_xlim(), '--', color='black')
        ax[i].set_aspect('equal')

    plt.show()


def plot_est_scatter(theta, pred_means, pred_var, parameters, n=1, figsize=(20, 5)):
    number_parameters = int(len(parameters))
    fig, ax = plt.subplots(1, number_parameters, figsize=figsize)

    # Select n random predictions
    sel_idx = np.random.randint(0, theta.shape[0], size=n)
    sel_theta = theta[sel_idx, :]
    sel_pred_means = tf.gather(pred_means, sel_idx)
    sel_pred_var = tf.gather(pred_var, sel_idx)

    print(sel_pred_means)
    print(sel_pred_var)

    for i in range(len(parameters)):
        ax[i].scatter(pred_means[:, i], theta[:, i], color='black', alpha=0.3)
        ax[i].errorbar(sel_theta[:, i],
                       sel_pred_means[:, i],
                       yerr=sel_pred_var[:, i],
                       fmt='.k')

        if i == 0:
            ax[i].set_ylabel('True')

        ax[i].set_xlabel('Estimated')
        ax[i].set_title(parameters[i])
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].plot(ax[i].get_xlim(), ax[i].get_xlim(), '--', color='black')
        ax[i].set_aspect('equal')

    plt.show()


def plot_true_est_scatter(theta_approx_means, theta_test, param_names, font_size_metric=10,
                          figsize=(20, 4), show=True, filename=None, font_size=12):
    """
    Plots a scatter plot with abline of the estimated posterior means vs true values.
    """
    # Plot settings
    plt.rcParams['font.size'] = font_size

    # Convert true parameters to numpy
    # theta_test = theta_test.numpy()

    # Determine figure layout
    if len(param_names) < 20:
        if len(param_names) >= 6:
            n_col = int(np.ceil(len(param_names) / 2))
            n_row = 2
        else:
            n_col = int(len(param_names))
            n_row = 1
    else:
        n_col = int(np.ceil(np.sqrt(len(param_names))))
        n_row = int(np.floor(np.sqrt(len(param_names))))

    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row > 1:
        axarr = axarr.flat

    # --- Plot true vs estimated posterior means on a single row --- #
    for j in range(len(param_names)):

        # Plot analytic vs estimated
        axarr[j].scatter(theta_approx_means[:, j], theta_test[:, j], color='black', alpha=0.4)

        # get axis limits and set equal x and y limits
        lower_lim = min(axarr[j].get_xlim()[0], axarr[j].get_ylim()[0])
        upper_lim = max(axarr[j].get_xlim()[1], axarr[j].get_ylim()[1])
        axarr[j].set_xlim((lower_lim, upper_lim))
        axarr[j].set_ylim((lower_lim, upper_lim))
        axarr[j].plot(axarr[j].get_xlim(), axarr[j].get_xlim(), '--', color='black')

        # Compute R2
        r2 = r2_score(theta_test[:, j], theta_approx_means[:, j])
        axarr[j].text(0.1, 0.9, '$R^2$={:.3f}'.format(r2),
                      horizontalalignment='left',
                      verticalalignment='center',
                      transform=axarr[j].transAxes,
                      size=font_size_metric)

        if j == 0:
            # Label plot
            axarr[j].set_xlabel('Estimated')
            axarr[j].set_ylabel('True')
        axarr[j].set_title(param_names[j])
        axarr[j].spines['right'].set_visible(False)
        axarr[j].spines['top'].set_visible(False)

    # Adjust spaces
    f.tight_layout()

    if show:
        plt.show()

    # Save if specified
    # if filename is not None:
    #     f.savefig("figures/{}_{}n_scatter.png".format(filename, X_test.shape[1]), dpi=600)


def plot_hist_parameters(predictions, parameters, figsize=(12, 5)):
    number_parameters = int(len(parameters))
    fig, ax = plt.subplots(1, number_parameters, figsize=figsize)
    max_n = np.empty(len(parameters))
    max_bin_cut = np.empty(len(parameters))
    for i in range(len(parameters)):
        bin_size = (max(predictions[:, i]) - min(predictions[:, i])) / 50
        sequence = np.arange(min(predictions[:, i]), max(predictions[:, i]), bin_size)
        n, bin_cuts, patches = ax[i].hist(predictions[:, i], bins=sequence)
        max_n[i] = max(n)
        argmax_n = np.argmax(n)
        max_bin_cut[i] = bin_cuts[argmax_n]

        if i == 0:
            ax[i].set_ylabel('Frequency')

        ax[i].set_xlabel('Estimated')
        ax[i].set_title(parameters[i])
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)

    print("Vertical line at: {}".format(max_bin_cut))
    plt.show()


def plot_bw_time_series(t, compartments, best_results, worst_results, figsize=(15, 5)):
    number_compartments = int(len(compartments))
    # Number of best/worst time series for comparison
    n_worst = worst_results.shape[0]
    step_worst = int(n_worst / 5)
    iter_worst = np.arange(start=0, stop=n_worst, step=step_worst)
    n_best = best_results.shape[0]
    step_best = int(n_best / 5)
    iter_best = np.arange(start=0, stop=n_best, step=step_best)

    fig, ax = plt.subplots(1, number_compartments, figsize=figsize)

    parameter = 0
    for i in iter_worst:
        for j in range(step_worst):
            a = worst_results[i + j, :, parameter]
            ax[parameter].plot(t, worst_results[i + j, :, parameter], 'r-', linewidth=0.75)
        parameter += 1

    parameter = 0
    for i in iter_best:
        for j in range(step_best):
            ax[parameter].plot(t, best_results[i + j, :, parameter], 'b-', linewidth=0.75)
        parameter += 1

    for i in range(len(compartments)):
        ax[i].set_title(compartments[i])
        ax[i].set_xlabel(r'Time')
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)

    ax[0].set_ylabel('Population')
    plt.show()

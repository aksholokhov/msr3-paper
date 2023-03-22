from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import seaborn as sns

from pysr3.lme.problems import LMEProblem
from pysr3.lme.oracles import LinearLMEOracle, LinearLMEOracleSR3
from pysr3.lme.models import SimpleLMEModel, L0LmeModel, L0LmeModelSR3

# np.seterr(all='raise', invalid='raise')

# based on the expert's prior knowledge and their experiences
# during previous rounds of GBD
historic_significance = {
    "cv_symptoms": 0,
    "cv_unadjusted": 1,
    "cv_b_parent_only": 1,
    "cv_or": 0,
    "cv_multi_reg": 1,
    "cv_low_bullying": 1,
    "cv_anx": 1,
    "percent_female": 1,
    "cv_selection_bias": 1,
    "cv_child_baseline": 0,
    "intercept": 1,
    "time": 1,
    "cv_baseline_adjust": 0
}


def generate_bullying_experiment(dataset_path, figures_directory):
    data = pd.read_csv(dataset_path)
    data = data.rename({"cv_low_threshold_bullying": "cv_low_bullying"}, axis=1)
    # intercept does not need to be in the dataset
    # data = data.drop("intercept", axis=1)

    col_target = "log_effect_size"
    col_se = "log_effect_size_se"
    col_group = "cohort"
    categorical_features_columns = [col for col in data.columns if col.startswith("cv_")] + [
        "percent_female",
    ]

    # Switching group indicator from 'cohort' (strings) to 'group' (int)
    group_to_id = {g: i for i, g in enumerate(data[col_group].unique())}
    data["group"] = data[col_group]
    data["group"] = data["group"].replace(group_to_id)
    col_group = "group"
    data["variance"] = data[col_se] ** 2

    # Fitting the model
    problem = LMEProblem.from_dataframe(data=data,
                                        fixed_effects=["intercept", "time"],
                                        random_effects=["intercept"],
                                        groups="group",
                                        variance="variance",
                                        target=col_target,
                                        not_regularized_fe=[],
                                        not_regularized_re=[]
                                        )

    model = SimpleLMEModel()
    model.fit_problem(problem)

    # plot solution in the original space
    figure = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(nrows=1, ncols=1)
    # plot solutions
    solution_plot = figure.add_subplot(grid[0, 0])
    colors = sns.color_palette("husl", problem.num_groups)
    max_time = data["time"].max()
    for i, (coef, (x, y, z, l)) in enumerate(zip(model.coef_["per_group_coefficients"], problem)):
        group_id = problem.group_labels[i]
        time = x[:, 1]
        color = to_hex(colors[i])
        solution_plot.scatter(time, y, label=group_id, c=color)
        solution_plot.errorbar(time, y, yerr=np.sqrt(l), c=color, fmt="none")
        solution_plot.plot([0, max_time], [coef[0], coef[0] + coef[2] * max_time], c=color)
    # solution_plot.legend()
    solution_plot.set_title(
        "Solution: " + r"$\beta$" + f" = [{model.coef_['beta'][0]:.2f}, {model.coef_['beta'][1]:.2f}], " + r"$\gamma$" + f" = [{model.coef_['gamma'][0]:.2f}]")
    solution_plot.set_xlabel("Time")
    solution_plot.set_ylabel("Target")

    plt.savefig(figures_directory / f"{dataset_path.name}_intercept_only.jpg")
    plt.close()

    # plot coefficients trajectory
    figure1 = plt.figure(figsize=(13, 12))
    grid1 = plt.GridSpec(nrows=3, ncols=2)

    figure2 = plt.figure(figsize=(13, 12))
    grid2 = plt.GridSpec(nrows=2, ncols=2)

    loss_plot = figure1.add_subplot(grid1[0, 0])
    aics_plot = figure1.add_subplot(grid1[0, 1])
    betas_plot = figure1.add_subplot(grid1[1, :])
    gammas_plot = figure1.add_subplot(grid1[2, :])

    inclusion_betas_plot = figure2.add_subplot(grid2[0, :])
    inclusion_gammas_plot = figure2.add_subplot(grid2[1, :])

    figure3 = plt.figure(figsize=(16, 5))
    grid3 = plt.GridSpec(nrows=1, ncols=2)
    beta_coefs_plot = figure3.add_subplot(grid3[:, 0])
    beta_assessment_plot = figure3.add_subplot(grid3[:, 1])

    problem = LMEProblem.from_dataframe(data=data,
                                        fixed_effects=["intercept", "time"] + categorical_features_columns,
                                        random_effects=["intercept"] + categorical_features_columns,
                                        groups=col_group,
                                        variance="variance",
                                        target=col_target,
                                        not_regularized_fe=["intercept", "time"],
                                        not_regularized_re=["intercept"],
                                        )

    tbetas = np.zeros((problem.num_fixed_features - 1, problem.num_fixed_features))
    tgammas = np.zeros((problem.num_random_features - 0, problem.num_random_features))
    losses = []
    selection_aics = []

    tol = 1e-4
    for nnz_tbeta in tqdm(range(len(categorical_features_columns) + 2, 1, -1)):
        nnz_tgamma = nnz_tbeta - 1
        model = L0LmeModelSR3(nnz_tbeta=nnz_tbeta,
                              nnz_tgamma=nnz_tgamma,
                              max_iter_solver=30000,  # 2000
                              max_iter_oracle=10000,  # 20
                              ell=50,  # 1000 # 50 was good
                              initializer="None",  # EM
                              tol_solver=tol,
                              tol_oracle=1e-5,
                              practical=True
                              )
        model.fit_problem(problem)

        tbetas[nnz_tbeta - 2, :] = model.coef_["beta"]
        tgammas[nnz_tgamma - 1, :] = model.coef_["gamma"]
        oracle = LinearLMEOracle(problem)
        losses.append(oracle.loss(beta=model.coef_["beta"],
                                  gamma=model.coef_["gamma"]))
        selection_aics.append(oracle.jones2010bic(beta=model.coef_["beta"],
                                                  gamma=model.coef_["gamma"]))

    colors = sns.color_palette("husl", problem.num_fixed_features)

    nnz_tbetas = np.array(range(2, len(categorical_features_columns) + 3, 1))
    beta_features_labels = []
    for i, feature in enumerate(["intercept", "time"] + categorical_features_columns):
        betas_plot.plot(nnz_tbetas, tbetas[:, i], label=feature, color=to_hex(colors[i]))
        inclusion_betas = np.copy(tbetas[:, i])
        idx_zero_betas = np.abs(inclusion_betas) < np.sqrt(tol)
        inclusion_betas[idx_zero_betas] = None
        inclusion_betas[~idx_zero_betas] = i
        inclusion_betas_plot.plot(nnz_tbetas, inclusion_betas, color=to_hex(colors[i]))
        beta_features_labels.append(feature)

    betas_plot.set_xticks(nnz_tbetas)
    inclusion_betas_plot.set_xticks(nnz_tbetas)
    inclusion_betas_plot.set_yticks(range(len(beta_features_labels)))
    inclusion_betas_plot.set_yticklabels(beta_features_labels)

    betas_plot.legend()
    inclusion_betas_plot.set_xlabel(r"$\|\beta\|_0$: maximum number of non-zero fixed effects allowed in the model.")
    betas_plot.set_xlabel(r"$\|\beta\|_0$: maximum number of non-zero fixed effects allowed in the model.")
    betas_plot.set_ylabel(r"$\beta$: fixed effects")
    betas_plot.set_title(
        f"{dataset_path.name}: optimal coefficients for fixed effects depending on maximum non-zero coefficients allowed.")
    # plot gammas trajectory

    # plot loss function and aics
    loss_plot.set_title("Loss")
    loss_plot.plot(nnz_tbetas, losses[::-1], label="Loss (R2)")
    loss_plot.set_xlabel(r"$\|\beta\|_0$ -- number of NNZ coefficients")
    loss_plot.set_ylabel("Loss")
    loss_plot.legend()
    loss_plot.set_xticks(nnz_tbetas)

    selection_aics = np.array(selection_aics[::-1])
    argmin_aic = np.argmin(selection_aics[:-1])
    aics_plot.plot(nnz_tbetas, selection_aics, label="AIC (R2)")
    aics_plot.scatter(nnz_tbetas[argmin_aic], selection_aics[argmin_aic], s=80, facecolors='none', edgecolors='r')
    aics_plot.set_xlabel(r"$\|\beta\|_0$ -- number of NNZ coefficients")
    aics_plot.set_ylabel("AIC")
    aics_plot.legend()
    aics_plot.set_xticks(nnz_tbetas)
    aics_plot.set_title("AIC")

    beta_historic_significance = np.array([bool(historic_significance[feature]) for feature in beta_features_labels])
    beta_predicted_significance = np.array([np.abs(tbetas[argmin_aic, i]) >= np.sqrt(tol) for i, f in
                                            enumerate(beta_features_labels)])
    plot_selection(inclusion_betas_plot, nnz_tbetas[argmin_aic], beta_historic_significance,
                   beta_predicted_significance)

    figure1.savefig(figures_directory / f"{dataset_path.stem}_fixed_feature_selection.jpg")
    print(
        f"Random feature selection saved as as {figures_directory / f'{dataset_path.stem}_random_feature_selection.jpg'}")
    ## Random feature selection plot



    nnz_tgammas = np.array(range(1, len(categorical_features_columns) + 2, 1))
    gamma_features_labels = []
    for i, feature in enumerate(["intercept"] + categorical_features_columns):
        color = to_hex(colors[i + 1]) if i > 0 else to_hex(colors[i])
        gammas_plot.plot(nnz_tgammas, tgammas[:, i], '--', label=feature, color=color)
        inclusion_gammas = np.copy(tgammas[:, i])
        idx_zero_gammas = np.abs(inclusion_gammas) < np.sqrt(tol)
        inclusion_gammas[idx_zero_gammas] = None
        inclusion_gammas[~idx_zero_gammas] = i
        inclusion_gammas_plot.plot(nnz_tgammas, inclusion_gammas, '--', color=color)
        gamma_features_labels.append(feature)

    gammas_plot.legend()
    gammas_plot.set_xlabel(r"$\|\gamma\|_0$: maximum number of non-zero random effects allowed in the model.")
    gammas_plot.set_ylabel(r"$\gamma$: variance of random effects")
    gammas_plot.set_title(
        f"{dataset_path.name}: optimal variances of random effects depending on maximum non-zero coefficients allowed.")

    gammas_plot.set_xticks(nnz_tgammas)
    inclusion_gammas_plot.set_xlabel(
        r"$\|\gamma\|_0$: maximum number of non-zero random effects allowed in the model." + "\n *the time is constrained to be always included as a fixed effect, so the algorithm choses $i$ fixed effects and $i-1$ random effects (out of those $i$ fixed effects).")

    inclusion_gammas_plot.set_xticks(nnz_tgammas)
    inclusion_gammas_plot.set_yticks(range(len(gamma_features_labels)))
    inclusion_gammas_plot.set_yticklabels(gamma_features_labels)

    gamma_historic_significance = np.array([bool(historic_significance[feature]) for feature in gamma_features_labels])
    gamma_predicted_significance = np.array([np.abs(tgammas[argmin_aic, i]) >= np.sqrt(tol) for i, f in
                                             enumerate(gamma_features_labels)])
    plot_selection(inclusion_gammas_plot, nnz_tgammas[argmin_aic], gamma_historic_significance,
                   gamma_predicted_significance)

    figure2.savefig(figures_directory / f"{dataset_path.stem}_random_feature_selection.jpg")
    print(f"Random feature selection saved as as {figures_directory / f'{dataset_path.stem}_random_feature_selection.jpg'}")

    # plot the same data on two separate plots
    for i, feature in enumerate(["intercept", "time"] + categorical_features_columns):
        beta_coefs_plot.plot(nnz_tbetas, tbetas[:, i], label=feature, color=to_hex(colors[i]))
    beta_coefs_plot.legend(bbox_to_anchor=(-.63, 1.0), loc='upper left')
    beta_coefs_plot.set_aspect('auto', adjustable='box')
    beta_coefs_plot.set_xticks(nnz_tbetas)
    beta_coefs_plot.set_xlabel("NNZ: maximum number of non-zero covariates allowed in the model")
    beta_coefs_plot.set_ylabel("Values of the model coefficients")

    beta_assessment_plot.set_yticks(range(len(beta_features_labels)))
    beta_assessment_plot.set_yticklabels(beta_features_labels)
    beta_historic_significance = np.array([bool(historic_significance[feature]) for feature in beta_features_labels])
    accuracies = []
    for j in range(len(nnz_tbetas)):
        beta_predicted_significance = np.array([np.abs(tbetas[j, i]) >= np.sqrt(tol) for i, f in
                                                enumerate(beta_features_labels)])
        tp, tn, fp, fn = plot_selection(beta_assessment_plot, nnz_tbetas[j], beta_historic_significance,
                       beta_predicted_significance, add_labels= j == 0)
        accuracies.append((tp+tn)/(tp+tn+fp+fn))

    beta_assessment_plot.set_xticks(nnz_tbetas)
    beta_assessment_plot.set_xticklabels([f'{a}\n\n{b:.2f}' for a, b in zip(nnz_tbetas, accuracies)])
    beta_assessment_plot.set_aspect('auto', adjustable='box')
    beta_assessment_plot.text(-2.0, -1.26, "NNZ Covariates")
    beta_assessment_plot.text(-.68, -2.23, "Accuracy")
    from matplotlib.patches import FancyBboxPatch, Patch
    rect = FancyBboxPatch((8.75, -.20), 0.5, 12.25, boxstyle="Round", linewidth=1, edgecolor='orange', facecolor='none')
    beta_assessment_plot.add_patch(rect)
    handles, labels = beta_assessment_plot.get_legend_handles_labels()
    beta_assessment_plot.legend(handles + [Patch(facecolor='none', edgecolor='orange')], labels + ["Chosen by BIC"], bbox_to_anchor=(1.4, 1.0), loc='upper right')

    figure3.tight_layout()
    figure3.savefig(figures_directory / f"{dataset_path.stem}_assessment_selection.jpg")
    print(
        f'Assessment saved as as {figures_directory / f"{dataset_path.stem}_assessment_selection.jpg"}')
    plt.close()


def plot_selection(ax, x, y_true, y_pred, add_labels=True):
    true_pos = (y_true == True) & (y_pred == True)
    true_neg = (y_true == False) & (y_pred == False)
    false_pos = (y_true == False) & (y_pred == True)
    false_neg = (y_true == True) & (y_pred == False)

    ax.scatter([x] * len(y_true),
               [None if t == False else i for i, t in
                enumerate(true_pos)], s=80, facecolors='none', edgecolors='g', label="True Positive" if add_labels else None)
    ax.scatter([x] * len(y_true),
               [None if t == False else i for i, t in
                enumerate(false_pos)], s=80, facecolors='none', edgecolors='r', label="False Positive" if add_labels else None)
    ax.scatter([x] * len(y_true),
               [None if t == False else i for i, t in
                enumerate(true_neg)], marker='X', s=80, facecolors='none', edgecolors='g', label="True Negative" if add_labels else None)
    ax.scatter([x] * len(y_true),
               [None if t == False else i for i, t in
                enumerate(false_neg)], marker='X', s=80, facecolors='none', edgecolors='r', label="False Negative" if add_labels else None)
    ax.legend()
    return sum(true_pos), sum(true_neg), sum(false_pos), sum(false_neg)


if __name__ == "__main__":
    base_directory = Path("/Users/aksh/Storage/repos/skmixed-experiments/paper_sum2021")
    dataset_path = base_directory / "bullying_data.csv"
    figures_directory = base_directory / "figures"
    generate_bullying_experiment(dataset_path, figures_directory)

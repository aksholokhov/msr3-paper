import numpy as np
import datetime
from matplotlib import pyplot as plt
import pickle
from pathlib import Path

from pysr3.lme.models import L1LmeModelSR3, L1LmeModel
from pysr3.lme.problems import LMEProblem
from pysr3.lme.oracles import LinearLMEOracle, LinearLMEOracleSR3
from pysr3.regularizers import L1Regularizer

from tqdm import tqdm


def get_levels(A, levelset=(  # 1e-5, 1e-4, 1e-3, 5e-3, 1e-2,
        # 5e-2, 1e-1,,
        5e-1, 1e0, 5e0, 1e1, 1.5e1, 2e1, 3e1, 4e1, 5e1, 1e2)):
    return A.min() + np.array(levelset)


def run_intuition_experiment(seed, num_covariates, model_parameters, problem_parameters, lam, initial_parameters,
                             beta_lims=(-3, 3), gamma_lims=(0, 3), grid_dim=100,
                             logs_folder="."):
    logger_keys = {"loss", "x", "converged", "iteration"}

    np.random.seed(seed)
    true_beta = np.array([1] + [2, 0] * int(num_covariates / 2))
    true_gamma = np.array([2, 0] * int(num_covariates / 2))

    problem, true_model_parameters = LMEProblem.generate(**problem_parameters,
                                                         beta=true_beta,
                                                         gamma=true_gamma,
                                                         seed=seed)
    # x, y = problem.to_x_y()

    regularizer = L1Regularizer(lam=lam)
    beta_span = np.linspace(beta_lims[0], beta_lims[1], grid_dim)
    gamma_span = np.linspace(gamma_lims[0], gamma_lims[1], grid_dim)

    l1_model = L1LmeModel(**model_parameters,
                          stepping="line-search",
                          lam=lam,
                          tol_solver=1e-7,
                          logger_keys=logger_keys)
    l1_model.fit_problem(problem,
                         initial_parameters=initial_parameters,
                         )
    l1_steps = np.array(l1_model.logger_.get("x"))

    l1_SR3_model = L1LmeModelSR3(**model_parameters,
                                 stepping="fixed",
                                 lam=lam,
                                 tol_solver=1e-7,
                                 logger_keys=logger_keys)
    l1_SR3_model.fit_problem(problem,
                             initial_parameters=initial_parameters,
                             )
    l1_sr3_steps = np.array(l1_SR3_model.logger_.get("x"))

    print(f"On a 2D sample L1 problem PGD took {len(l1_steps)} steps to converge"
          f" and MSR3 took {len(l1_sr3_steps)} steps to converge.")

    # building levelset
    oracle_normal = LinearLMEOracle(problem)
    oracle_sr3 = LinearLMEOracleSR3(problem, lb=model_parameters['ell'], lg=model_parameters['ell'])
    beta_grid_normal = np.zeros((grid_dim, grid_dim))
    beta_grid_sr3 = np.zeros((grid_dim, grid_dim))
    gamma_grid_normal = np.zeros((grid_dim, grid_dim))
    gamma_grid_sr3 = np.zeros((grid_dim, grid_dim))
    for i, beta_1 in enumerate(tqdm(beta_span, desc="Evaluating value function for betas on a grid for contour plots")):
        for j, beta_2 in enumerate(beta_span):
            x = np.array([1, beta_1, beta_2, *true_gamma])
            normal_loss = oracle_normal.value_function(x) + regularizer.value(x)
            sr3_loss = oracle_sr3.value_function(x) + regularizer.value(x)
            beta_grid_normal[i, j] = normal_loss
            beta_grid_sr3[i, j] = sr3_loss
    for i, gamma_1 in enumerate(tqdm(gamma_span, desc="Evaluating value function for gammas on a grid for contour plots")):
        for j, gamma_2 in enumerate(gamma_span):
            x = np.array([*true_beta, gamma_1, gamma_2])
            normal_loss = oracle_normal.value_function(x) + regularizer.value(x)
            sr3_loss = oracle_sr3.value_function(x) + regularizer.value(x)
            gamma_grid_normal[i, j] = normal_loss
            gamma_grid_sr3[i, j] = sr3_loss

    beta_grid_normal = beta_grid_normal.T
    gamma_grid_normal = gamma_grid_normal.T
    beta_grid_sr3 = beta_grid_sr3.T
    gamma_grid_sr3 = gamma_grid_sr3.T

    result = {
        "beta_grid_normal": beta_grid_normal,
        "gamma_grid_normal": gamma_grid_normal,
        "beta_grid_sr3": beta_grid_sr3,
        "gamma_grid_sr3": gamma_grid_sr3,
        "l1_steps": l1_steps,
        "l1_sr3_steps": l1_sr3_steps
    }
    params = {
        "seed": seed,
        "num_covariates": num_covariates,
        "grid_dim": grid_dim,
        "model_parameters": model_parameters,
        "problem_parameters": problem_parameters,
        "lam": lam,
        "initial_parameters": initial_parameters,
    }
    with open(Path(logs_folder) / f"log_intuition.pickle", 'wb') as f:
        pickle.dump(result, file=f)
    with open(Path(logs_folder) / f"params_intuition.pickle", 'wb') as f:
        pickle.dump(params, file=f)
    print(f"Intuition data saved as {Path(logs_folder) / f'log_intuition.pickle'}")
    return result, None


def plot_intuition_picture(data, figures_folder=".",
                           beta_lims=(-3, 3), gamma_lims=(0, 3), grid_dim=100, ):
    beta_grid_normal = data["beta_grid_normal"]
    gamma_grid_normal = data["gamma_grid_normal"]
    beta_grid_sr3 = data["beta_grid_sr3"]
    gamma_grid_sr3 = data["gamma_grid_sr3"]
    l1_steps = data["l1_steps"]
    l1_sr3_steps = data["l1_sr3_steps"]

    beta_span = np.linspace(beta_lims[0], beta_lims[1], grid_dim)
    gamma_span = np.linspace(gamma_lims[0], gamma_lims[1], grid_dim)

    fig, ((ax_beta_normal, ax_beta_sr3), (ax_gamma_normal, ax_gamma_sr3)) = plt.subplots(nrows=2, ncols=2,
                                                                                         figsize=(20, 10))
    CS = ax_beta_normal.contour(beta_span, beta_span, beta_grid_normal, levels=get_levels(beta_grid_normal))
    #plt.clabel(CS, inline=1, fontsize=10, fmt=lambda x: f"{x - beta_grid_normal.min():1.1e}")
    ax_beta_normal.plot(l1_steps[:, 1], l1_steps[:, 2], 'r-o', label=f"Proximal Gradient Descent ({l1_steps.shape[0]} it.)")
    # ax_beta_normal.plot([-lam, 0, lam, 0, -lam], [0, lam, 0, -lam, 0], c='b')
    ax_beta_normal.set_title(r"Original Likelihood for $\beta$ (fixed effects)", fontsize=22)
    ax_beta_normal.legend(bbox_to_anchor=(0.9, 1), fontsize=20)

    CS = ax_beta_sr3.contour(beta_span, beta_span, beta_grid_sr3, levels=get_levels(beta_grid_sr3))
    #plt.clabel(CS, inline=1, fontsize=10, fmt=lambda x: f"{x - beta_grid_sr3.min():1.1e}")
    ax_beta_sr3.plot(l1_sr3_steps[:, 1], l1_sr3_steps[:, 2], 'r-o', label=f"Proximal Gradient Descent ({l1_sr3_steps.shape[0]} it.)")
    ax_beta_sr3.set_title(r"MSR3-relaxation for $\beta$ (fixed effects)", fontsize=22)
    # ax_beta_sr3.plot([-lam, 0, lam, 0, -lam], [0, lam, 0, -lam, 0], c='b')
    ax_beta_sr3.legend(bbox_to_anchor=(0.85, 1), fontsize=20)

    CS = ax_gamma_normal.contour(gamma_span, gamma_span, gamma_grid_normal, levels=get_levels(gamma_grid_normal))
    #plt.clabel(CS, inline=1, fontsize=10, fmt=lambda x: f"{x - gamma_grid_normal.min():1.1e}")
    ax_gamma_normal.plot(l1_steps[:, 3], l1_steps[:, 4], 'r-o', label=f"Proximal Gradient Descent ({l1_steps.shape[0]} iterations)")
    ax_gamma_normal.set_title(r"Original Likelihood for $\gamma$ (random effects)", fontsize=22)
    # ax_gamma_normal.plot([0, lam], [lam, 0], c='b')
    ax_gamma_normal.legend(fontsize=20)

    CS = ax_gamma_sr3.contour(gamma_span, gamma_span, gamma_grid_sr3, levels=get_levels(gamma_grid_sr3))
    #plt.clabel(CS, inline=1, fontsize=10, fmt=lambda x: f"{x - gamma_grid_sr3.min():1.1e}")
    ax_gamma_sr3.plot(l1_sr3_steps[:, 3], l1_sr3_steps[:, 4], 'r-o', label=f"Proximal Gradient Descent, ({l1_sr3_steps.shape[0]} iterations)")
    ax_gamma_sr3.set_title("MSR3-relaxation for $\gamma$ (random effects)", fontsize=22)
    # ax_gamma_sr3.plot([0, lam], [lam, 0], c='b')
    ax_gamma_sr3.legend(fontsize=20)
    filename = Path(figures_folder) / f"intuition.jpg"
    plt.savefig(filename)
    print(f"Intuition figure (Figure 2) saved as {filename}")

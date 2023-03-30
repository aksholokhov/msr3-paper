import numpy as np
import argparse
from pathlib import Path
import pickle

from sklearn.utils.fixes import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix

from pysr3.lme.problems import LMEProblem, FIXED_RANDOM
from pysr3.lme.models import L1LmeModelSR3, L1LmeModel

import time

parser = argparse.ArgumentParser('scalability')
parser.add_argument('--trial', type=int, default=1, help='Each "trial" represents one experiment')
parser.add_argument('--base_folder', type=str, default=".", help='Path to the base folder (where this file is).')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--experiment_name', type=str, default="scaling", help='Experiment name')
args = parser.parse_args()

base_folder = Path(args.base_folder)

problem_multipliers = [1, 2, 5, 10, 20, 50, 100]

if __name__ == "__main__":
    if args.trial > 700:
        algo = "pgd"
    else:
        algo = "msr3-fast"

    trial = (args.trial % 700) % 100
    multiplier = problem_multipliers[(args.trial % 700) // 100]

    experiment_folder = base_folder / "results" / f"{args.experiment_name}"
    logs_folder = experiment_folder / "logs"

    print(f"Creating a folder for the experiments' outputs: {experiment_folder}")
    experiment_folder.mkdir(parents=True, exist_ok=True)
    logs_folder.mkdir(parents=True, exist_ok=True)

    # Here we generate a random linear mixed-effects problem.
    # To use your own dataset check LMEProblem.from_dataframe and LMEProblem.from_x_y
    problem, true_parameters = LMEProblem.generate(
        groups_sizes=[10 * multiplier] * 8,  # 8 groups, 10 objects each
        features_labels=["fixed+random"] * (20 * multiplier),
        # 20 features, each one having both fixed and random components
        beta=np.array([0, 1] * (10 * multiplier)),  # True beta (fixed effects) has every other coefficient active
        gamma=np.array([0, 0, 0, 1] * (5 * multiplier)),
        # True gamma (variances of random effects) has every fourth coefficient active
        obs_var=0.1,  # The errors have standard errors of sqrt(0.1) ~= 0.33
        seed=1000*args.seed+trial  # random seed, for reproducibility
    )
    # LMEProblem provides a very convenient representation
    # of the problem. See the documentation for more details.

    # It also can be converted to a more familiar representation
    x, y, columns_labels = problem.to_x_y()
    # columns_labels describe the roles of the columns in x:
    # fixed effect, random effect, or both of those, as well as groups labels and observation standard deviation.

    # You can also convert it to pandas dataframe if you'd like.
    pandas_dataframe = problem.to_dataframe()

    # We use SR3-empowered LASSO model, but many other popular models are also available.
    # See the glossary of models for more details.
    if algo == "msr3-fast":
        model = L1LmeModelSR3(practical=True, tol_solver=1e-4 * multiplier, tol_oracle=1e-5 * multiplier, ell=20, lam=1e-3)
    elif algo == "pgd":
        model = L1LmeModel(tol_solver=1e-4 * multiplier, lam=1, stepping='line-search')
    else:
        raise ValueError(f"Unknown algo: {algo}")

    start = time.time()
    model.fit(x, y, columns_labels=columns_labels)
    end = time.time()
    best_model = model

    maybe_beta = best_model.coef_["beta"]
    maybe_gamma = best_model.coef_["gamma"]

    # Since the solver stops witin sqrt(tol) from the minimum, we use it as a criterion for whether the feature
    # is selected or not
    ftn, ffp, ffn, ftp = confusion_matrix(y_true=true_parameters["beta"],
                                          y_pred=abs(maybe_beta) > np.sqrt(best_model.tol_solver)
                                          ).ravel()
    rtn, rfp, rfn, rtp = confusion_matrix(y_true=true_parameters["gamma"],
                                          y_pred=abs(maybe_gamma) > np.sqrt(best_model.tol_solver)
                                          ).ravel()
    logs = {
        "algo": algo,
        "ftn": ftn,
        "ffp": ffp,
        "ffn": ffn,
        "ftp": ftp,
        "time": end - start
    }
    with open(logs_folder / f"log_{args.trial}.pickle", "wb") as f:
        pickle.dump(logs, f)






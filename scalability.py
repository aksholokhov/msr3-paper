import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import pickle

from sklearn.utils.fixes import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix

from pysr3.lme.problems import LMEProblem, FIXED_RANDOM, LMEStratifiedShuffleSplit
from pysr3.lme.models import L1LmeModelSR3, L1LmeModel

import time

parser = argparse.ArgumentParser('scalability')
parser.add_argument('--trial', type=int, default=1, help='Each "trial" represents one experiment')
parser.add_argument('--base_folder', type=str, default=".", help='Path to the base folder (where this file is).')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--experiment_name', type=str, default="scaling", help='Experiment name')
parser.add_argument('--n_gridsearch_lambda', type=int, default=10)
parser.add_argument('--ic', type=str, default="jones_bic")
args = parser.parse_args()

base_folder = Path(args.base_folder)

possible_etas = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1]
problem_multipliers = [1, 2, 5, 10, 20, 50, 100]

if __name__ == "__main__":
    multiplier = problem_multipliers[args.trial // 1000]
    trial = args.trial % 100

    if (args.trial % 1000) < 800:
        algo = "MSR3-Fast"
        eta = possible_etas[(args.trial % 1000) // 100]
        model = L1LmeModelSR3(practical=True, tol_solver=1e-4 * multiplier, tol_oracle=1e-5 * multiplier, ell=eta)
    elif 800 <= (args.trial % 1000) < 900:
        algo = "PGD-LineSearch"
        model = L1LmeModel(tol_solver=1e-4 * multiplier, stepping='line-search')
    elif 900 <= (args.trial % 1000) < 1000:
        algo = "PGD"
        model = L1LmeModel(tol_solver=1e-4 * multiplier, stepping='fixed', fixed_step_len=1e-3)
    else:
        raise ValueError(f"Unknown trial num")

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

    params = {
        "lam": loguniform(1e-4, 1e2),
    }
    # We use standard functionality of sklearn to perform grid-search.
    selector = RandomizedSearchCV(estimator=model,
                                  param_distributions=params,
                                  n_iter=args.n_gridsearch_lambda,  # number of points from parameters space to sample
                                  # the class below implements CV-splits for LME models
                                  cv=LMEStratifiedShuffleSplit(n_splits=2, test_size=0.5,
                                                               random_state=args.seed,
                                                               columns_labels=columns_labels),
                                  # The function below will evaluate the information criterion
                                  # on the test-sets during cross-validation.
                                  # We use cAIC from Vaida, but other options (BIC, Muller's IC) are also available
                                  scoring=lambda clf, x, y: -clf.get_information_criterion(x,
                                                                                           y,
                                                                                           columns_labels=columns_labels,
                                                                                           ic=args.ic),
                                  random_state=args.seed,
                                  n_jobs=1
                                  )
    start = time.time()
    selector.fit(x, y, columns_labels=columns_labels)
    end = time.time()
    exec_time = (end - start) / args.n_gridsearch_lambda
    best_model = selector.best_estimator_

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
        "rtn": rtn,
        'rfp': rfp,
        'rfn': rfn,
        'rtp': rtp,
        "time": exec_time
    }
    with open(logs_folder / f"log_{args.trial}.pickle", "wb") as f:
        pickle.dump(logs, f)


def process_results(logs_folder: Path):
    logs = []
    for file in logs_folder.iterdir():
        trial_num = file.name.split(".")[0].split("_")[1]
        with open(file, 'rb') as f:
            logs.append({"trial": int(trial_num), **pickle.load(f)})
    data = pd.DataFrame.from_records(logs).sort_values(by='trial')
    data['n_features'] = data['ftn'] + data['ffp'] + data['ffn'] + data['ftp']
    data['accuracy'] = (data["ftp"] + data["ftn"]) / (data['n_features'])
    data['time'] = data['time'].round(2)

    groups = data[['algo', 'n_features', 'time']].pivot_table(index='n_features', columns='algo', values='time')
    pass
